from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from speculators.models.retrace.target import LocalTarget, RemoteTarget
from tests.integration.models.test_retrace_reference import models


@pytest.fixture(autouse=True)
def no_network_client(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace())


def test_remote_checks_actual_tokens_normalizes_once_and_deletes(tmp_path, monkeypatch):
    from speculators.data_generation import vllm_client

    model, draft = models()
    payload = tmp_path / "states.safetensors"
    tokens = [1, 2, 3, 4, 7, 8]
    # Capture pre-final-norm exactly as the native hidden-states connector does.
    captured = []
    handle = model.model.norm.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach())
    )
    with torch.no_grad():
        outputs = model.model(torch.tensor([tokens]), output_hidden_states=True)
    handle.remove()
    hs = torch.stack(
        [outputs.hidden_states[1][0], outputs.hidden_states[2][0], captured[0][0]], 1
    )
    save_file(
        {"token_ids": torch.tensor(tokens), "hidden_states": hs.contiguous()},
        str(payload),
    )

    def request(client, name, item, **kwargs):
        assert item == {"input_ids": tokens}
        assert name == "retrace-target"
        return str(payload)

    monkeypatch.setattr(vllm_client, "generate_hidden_states", request)
    remote = RemoteTarget(draft, "http://127.0.0.1:9999/v1", "retrace-target", 2)
    actual = remote.states(tokens)
    expected = LocalTarget(model, draft.target_layer_ids).states(tokens)
    torch.testing.assert_close(actual.auxiliary, expected.auxiliary)
    torch.testing.assert_close(actual.scoring, expected.scoring)
    assert not payload.exists()
    assert remote.states(tokens) is actual
    assert remote.calls == 1


@pytest.mark.parametrize("bad", ["tokens", "nan", "layers"])
def test_remote_rejects_bad_payload_and_cleans_up(tmp_path, monkeypatch, bad):
    from speculators.data_generation import vllm_client

    _, draft = models()
    payload = tmp_path / "bad.safetensors"
    ids = torch.tensor([1, 2, 3])
    states = torch.randn(3, 3, draft.hidden_size)
    if bad == "tokens":
        ids[0] = 4
    if bad == "nan":
        states[0, 0, 0] = float("nan")
    if bad == "layers":
        states = states[:, :2].contiguous()
    save_file({"token_ids": ids, "hidden_states": states}, str(payload))
    monkeypatch.setattr(
        vllm_client, "generate_hidden_states", lambda *a, **k: str(payload)
    )
    remote = RemoteTarget(draft, "http://127.0.0.1:9999/v1", "target", 2)
    with pytest.raises(ValueError):
        remote.states([1, 2, 3])
    assert not payload.exists()
