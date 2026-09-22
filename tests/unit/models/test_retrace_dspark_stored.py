import copy
import json
from concurrent.futures import Future

import pytest
import torch
from datasets import Dataset

from speculators.models.retrace_dspark.core import (
    ReTraceDSparkDraftModel as RuntimeModel,
)
from speculators.models.retrace_dspark.pretrained import load_dspark
from speculators.models.retrace_dspark.stored_data import prepare, stored_row
from speculators.models.retrace_dspark.stored_target import BranchTarget
from speculators.models.retrace_dspark.stored_training import (
    ReTraceDSparkDraftModel,
    verified_pair,
)
from tests.unit.models.test_retrace_dspark_pretrained import fixture


class LocalPrefillServer:
    """Real tiny target, exposing the file connector's PRE-norm convention."""

    def __init__(self, target, layer_ids):
        self.target, self.layer_ids = target, layer_ids
        self.calls = []

    @torch.no_grad()
    def prefill(self, tokens):
        captured = []
        hook = self.target.model.norm.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        result = self.target.model(
            torch.tensor([tokens]), use_cache=False, output_hidden_states=True
        )
        hook.remove()
        auxiliary = torch.cat([result.hidden_states[i] for i in self.layer_ids], -1)
        return auxiliary, captured[0]

    def submit(self, tokens, anchor, proposals):
        self.calls.append((list(tokens), anchor, proposals))
        _, pre_norm = self.prefill(tokens)
        result = Future()
        result.set_result(pre_norm[0, anchor : anchor + proposals])
        return result


def train_fixture(tmp_path):
    target, source, _ = fixture(tmp_path)
    original, _ = load_dspark(source, str(tmp_path / "target"), dtype=torch.float32)
    adapter = ReTraceDSparkDraftModel(copy.deepcopy(original.config))
    adapter.load_state_dict(original.state_dict())
    adapter.load_verifier_weights()
    server = LocalPrefillServer(target, adapter.target_layer_ids)
    adapter.configure_stored(server, pairs_per_batch=4, blocks_per_forward=2)
    tokens = [1, 2, 3]
    with torch.no_grad():
        for _ in range(35):
            result = target(torch.tensor([tokens]), use_cache=False)
            tokens.append(int(result.logits[0, -1].argmax()))
    auxiliary, final = server.prefill(tokens)
    return (
        target,
        adapter,
        server,
        {
            "input_ids": torch.tensor([tokens]),
            "hidden_states": auxiliary,
            "verifier_last_hidden_states": final,
            "loss_mask": torch.tensor([[False] * 3 + [True] * 35]),
            "document_ids": torch.zeros(1, len(tokens), dtype=torch.long),
        },
    )


def test_stored_data_preserves_response_and_mask_and_respects_turns(tmp_path):
    row = {
        "input_ids": list(range(1, 18)),
        "loss_mask": [False] * 3 + [True] * 8 + [False] * 2 + [True] * 4,
        "seq_len": 17,
    }
    selected, boundary = stored_row(row, 50, 5, 6, 5)
    assert boundary == 3
    assert selected["input_ids"] == row["input_ids"][:9]
    assert selected["loss_mask"] == row["loss_mask"][:9]
    assert stored_row(row, 50, 2, 6, 5) is None
    with pytest.raises(ValueError, match="Invalid"):
        stored_row({**row, "seq_len": 30}, 50, 5, 6, 5)
    source = tmp_path / "source"
    Dataset.from_list([row]).save_to_disk(source)
    manifest = prepare(
        source,
        tmp_path / "selected",
        count=1,
        vocabulary=50,
        block_size=5,
        prompt_limit=5,
        response_limit=6,
    )
    assert manifest["tokens_preserved"]
    assert manifest["total_tokens"] == 9


def test_verified_pair_discards_rejection_and_requires_matching_committed_prefix():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden = torch.arange(12.0).reshape(1, 4, 3).requires_grad_()
    scores = (hidden + 30).clone()
    desired = torch.tensor([[3, 4, 8, 9]])
    pair = verified_pair(tokens, 1, [3, 9, 9, 9], hidden, scores, desired)
    assert pair.accepted == 1 and pair.next_anchor == 3 and pair.aligned
    assert pair.memory.valid.tolist() == [[True, True, False, False]]
    torch.testing.assert_close(pair.memory.draft[:, :2], hidden[:, 2:].half())
    torch.testing.assert_close(pair.memory.target[:, :2], scores[:, 2:].half())
    assert not pair.memory.draft.requires_grad and not pair.memory.target.requires_grad
    tokens[3] = 20
    assert not verified_pair(tokens, 1, [3, 9, 9, 9], hidden, scores, desired).aligned
    accepted = verified_pair(tokens, 1, [3, 4, 8, 9], hidden, scores, desired)
    assert not accepted.memory.valid.any()


def test_training_uses_actual_branch_states_and_trains_conditioner(tmp_path):
    target, adapter, server, batch = train_fixture(tmp_path)
    adapter.train()
    optimizer = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad], lr=1e-3
    )
    captures = []
    original = adapter.training_blocks

    def recording(context, anchors, positions, memory, beta, **kwargs):
        if torch.is_grad_enabled() and memory is not None:
            captures.append((positions.tolist(), memory))
        return original(context, anchors, positions, memory, beta, **kwargs)

    adapter.training_blocks = recording
    _, loss, metrics = adapter(**batch)
    assert len(server.calls) == 4
    assert all(k == 7 for _, _, k in server.calls)
    assert metrics["conditioned_pair_fraction_sum"] > 0
    assert metrics["conditioned_positions_sum"] > 0
    assert torch.isfinite(loss)
    # A branch query contains the draft's rejected tokens, not only clean tokens.
    tokens = batch["input_ids"][0].tolist()
    assert any(query != tokens[: len(query)] for query, _, _ in server.calls)
    assert any(memory.valid.any() for _, memory in captures)
    loss.backward()
    assert adapter.retrace.value.weight.grad.abs().sum() > 0
    assert adapter.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    for name, parameter in adapter.named_parameters():
        if name.startswith(("markov_head.", "confidence_head.")):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in target.parameters())
    assert adapter.embed_tokens.weight.grad is None
    assert adapter.lm_head.weight.grad is None
    assert all(
        not m.draft.requires_grad and not m.target.requires_grad for _, m in captures
    )
    # Beta uses restored optimizer update count rather than restarting on resume.
    # The zero-initialized residual must learn first before correction/gate
    # receive nonzero gradients.
    optimizer.step()
    adapter.zero_grad(set_to_none=True)
    adapter.step_provider = lambda: 60
    _, loss, metrics = adapter(**batch)
    assert metrics["beta_sum"] == 1
    loss.backward()
    assert adapter.retrace.correction.weight.grad.abs().sum() > 0
    assert adapter.retrace.gate.weight.grad.abs().sum() > 0


def test_document_isolation_sampling_and_runtime_checkpoint_compatibility(tmp_path):
    _, adapter, server, batch = train_fixture(tmp_path)
    packed = {k: torch.cat((v, v), 1) for k, v in batch.items()}
    length = batch["input_ids"].shape[1]
    packed["document_ids"][:, length:] = 1
    documents, candidates = adapter._documents(**packed)
    assert len(documents) == 2
    assert all(1 <= pos < length for _, pos in candidates)
    _, _, metrics = adapter(**packed)
    assert metrics["source_pairs_sum"] == 4  # budget per batch, not per document
    assert all(len(tokens) <= length for tokens, _, _ in server.calls)
    adapter.save_pretrained(tmp_path / "checkpoint")
    config = json.loads((tmp_path / "checkpoint/config.json").read_text())
    assert config["architectures"] == ["ReTraceDSparkDraftModel"]
    assert config["training_objective"] == "dspark_stored_pair"
    runtime = RuntimeModel.from_pretrained(tmp_path / "checkpoint", dtype=torch.float32)
    for key, value in adapter.state_dict().items():
        torch.testing.assert_close(value, runtime.state_dict()[key])


def test_no_memory_batch_has_all_conditioner_gradients(tmp_path):
    _, adapter, _, batch = train_fixture(tmp_path)
    # Force clean-response disagreement at every stored successor; teacher
    # features stay the original fixture to exercise the no-pair path only.
    batch["input_ids"][:, 3:] = 47
    _, loss, metrics = adapter(**batch)
    loss.backward()
    assert all(p.grad is not None for p in adapter.retrace.parameters())
    assert torch.isfinite(loss)


@pytest.mark.parametrize("force_unaligned", [False, True])
def test_bf16_verification_does_not_detach_training_backbone(tmp_path, force_unaligned):
    """The production outer autocast scope must not reuse no-grad weight casts."""
    from speculators.models.retrace_dspark.stored_train import load_model

    _, original, server, batch = train_fixture(tmp_path)
    with torch.no_grad():
        original.retrace.value.weight.zero_()
    original.save_pretrained(tmp_path / "initial")
    adapter = load_model(tmp_path / "initial")
    adapter.configure_stored(server, pairs_per_batch=4, blocks_per_forward=2)
    batch = {
        key: value.to(torch.bfloat16) if value.is_floating_point() else value
        for key, value in batch.items()
    }
    if force_unaligned:
        batch["input_ids"][:, 3:] = 47
    # These matrices are used first under no_grad, then for training, inside
    # the SAME outer BF16 scope used by the stock Trainer.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss, metrics = adapter(**batch)
    if force_unaligned:
        assert metrics["conditioned_pair_fraction_sum"] == 0
    else:
        assert metrics["conditioned_pair_fraction_sum"] > 0
    loss.backward()
    missing = [
        name
        for name, param in adapter.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []
    for name in (
        "fc.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    ):
        grad = dict(adapter.named_parameters())[name].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(
        param.grad is None for param in adapter.parameters() if not param.requires_grad
    )


def test_dspark_sdpa_replay_matches_native_slots_and_blocks_clean_future(tmp_path):
    from speculators.models.retrace_dspark.memory import ReTraceMemory

    _, adapter, _, batch = train_fixture(tmp_path)
    context = batch["hidden_states"]
    tokens = batch["input_ids"][0]
    positions = torch.tensor([3, 6])
    previous = tokens[positions[:, None] + torch.arange(7)]
    h = torch.randn(2, 7, 32)
    memory = ReTraceMemory(h, h / 2, torch.ones(2, 7, dtype=torch.bool))
    with torch.no_grad():
        adapter.retrace.value.weight.normal_(std=0.01)
    result = adapter.training_blocks(
        context, tokens[positions], positions, memory, previous_token_ids=previous
    )
    native = RuntimeModel.training_blocks(
        adapter,
        context,
        tokens[positions],
        positions,
        memory,
        previous_token_ids=previous,
    )
    torch.testing.assert_close(result.logits, native.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        result.confidence_logits, native.confidence_logits, rtol=1e-5, atol=1e-6
    )
    assert result.logits.shape[1] == 7  # anchor predicts proposal zero too
    changed = context.clone()
    changed[:, 6:] += 100
    altered = adapter.training_blocks(
        changed, tokens[positions], positions, memory, previous_token_ids=previous
    )
    torch.testing.assert_close(result.logits, altered.logits, rtol=1e-5, atol=1e-6)
    # Memory zero maps to the unchanged anchor input, not the first mask.
    original = result.logits.detach().clone()
    memory.draft[:, 0] += 100
    memory.target[:, 0] += 100
    altered = adapter.training_blocks(
        context, tokens[positions], positions, memory, previous_token_ids=previous
    )
    torch.testing.assert_close(original, altered.logits)


def test_stored_loss_uses_clean_markov_inputs_and_target_argmax(tmp_path, monkeypatch):
    from speculators.models.retrace_dspark import stored_training

    _, adapter, _, batch = train_fixture(tmp_path)
    # Deliberately non-greedy stored tokens exercise the distinction between
    # previous-token conditioning and the native target-argmax CE label.
    batch["input_ids"][:, 3:] = 47
    native_loss = stored_training.block_loss_sum
    calls = []

    def checked_loss(logits, targets, confidence, labels, **kwargs):
        assert torch.equal(labels[labels != -100], targets.argmax(-1)[labels != -100])
        assert kwargs == {
            "ce_weight": 0.1,
            "tv_weight": 0.9,
            "confidence_alpha": 1.0,
            "gamma": 4.0,
        }
        calls.append(labels)
        return native_loss(logits, targets, confidence, labels, **kwargs)

    block = adapter.training_blocks
    seen_previous = []

    def checked_blocks(
        context, anchors, positions, memory, beta, previous_token_ids=None
    ):
        if torch.is_grad_enabled():
            assert previous_token_ids is not None
            ix = (positions[:, None] + torch.arange(7)).clamp(
                max=batch["input_ids"].shape[1] - 1
            )
            assert torch.equal(previous_token_ids, batch["input_ids"][0, ix])
            seen_previous.append(previous_token_ids)
        else:
            assert previous_token_ids is None  # real sequential Markov samples
        return block(context, anchors, positions, memory, beta, previous_token_ids)

    monkeypatch.setattr(stored_training, "block_loss_sum", checked_loss)
    adapter.training_blocks = checked_blocks
    _, loss, _ = adapter(**batch)
    loss.backward()
    assert calls and seen_previous


def test_remote_causal_slice_token_validation_and_file_cleanup(monkeypatch):
    from speculators.models.retrace_dspark import stored_target

    tokens = [1, 2, 3, 4, 5, 6]
    state = torch.randn(6, 3, 7)

    class Transfer:
        deleted = []

        def get_generated(self, handle):
            return {"hidden_states": state, "token_ids": torch.tensor(tokens)}

        def delete(self, handle):
            self.deleted.append(handle)

    class Client:
        def close(self):
            pass

    calls = []

    def generate(client, model, item, **kwargs):
        calls.append(item)
        return "branch-file"

    monkeypatch.setattr(stored_target, "generate_hidden_states", generate)
    transfer = Transfer()
    remote = BranchTarget("unused", "target", 2, 7, transfer=transfer, client=Client())
    try:
        result = remote.submit(tokens, 2, 3).result()
        torch.testing.assert_close(result, state[2:5, -1])
        assert calls == [{"input_ids": tokens}]
        assert transfer.deleted == ["branch-file"]
        state = torch.randn(6, 2, 7)
        with pytest.raises(ValueError, match="layer mismatch"):
            remote.submit(tokens, 2, 3).result()
        assert transfer.deleted == ["branch-file", "branch-file"]
    finally:
        remote.close()


def test_stock_arrow_loader_epoch_checkpoint_and_resume(tmp_path, monkeypatch):
    """Exercise the actual shared Trainer; only device selection is adapted to CPU."""
    from hs_connectors.transfer import FileTransfer
    from safetensors.torch import save_file

    from speculators.models.retrace_dspark.stored_train import StoredTrainer, load_model
    from speculators.train import checkpointer
    from speculators.train import trainer as trainer_module
    from speculators.train.data import ArrowDataset
    from speculators.train.dataloader import _setup_dataloader
    from speculators.train.trainer import TrainerConfig

    _, adapter, server, batch = train_fixture(tmp_path)
    with torch.no_grad():
        adapter.retrace.value.weight.zero_()
    adapter.save_pretrained(tmp_path / "initial")
    adapter = load_model(tmp_path / "initial")
    adapter.configure_stored(server, pairs_per_batch=4, blocks_per_forward=2)
    assert adapter.layers[0].self_attn.config._attn_implementation == "sdpa"
    ids = batch["input_ids"][0].tolist()
    mask = batch["loss_mask"][0].tolist()
    dataset_dir = tmp_path / "arrow"
    Dataset.from_list(
        [{"input_ids": ids, "loss_mask": mask, "seq_len": len(ids)}] * 2
    ).save_to_disk(dataset_dir)
    cache = tmp_path / "clean"
    cache.mkdir()
    aux = batch["hidden_states"][0].reshape(len(ids), 2, 32)
    full = torch.cat((aux, batch["verifier_last_hidden_states"][0, :, None]), dim=1)
    for index in range(2):
        save_file(
            {"hidden_states": full.contiguous(), "token_ids": torch.tensor(ids)},
            str(cache / f"hs_{index}.safetensors"),
        )
    dataset = ArrowDataset(
        max_len=len(ids),
        datapath=dataset_dir,
        transfer=FileTransfer(cache),
        on_missing="raise",
        hidden_states_dtype=torch.float32,
    )
    dataset.data.set_format("torch")
    loader = _setup_dataloader(
        dataset, len(ids), hidden_size=32, num_target_layers=2, num_workers=0
    )
    loader.pin_memory = False  # CPU has no pinned accelerator-memory allocator.
    monkeypatch.setattr(trainer_module, "get_local_rank", lambda: "cpu")
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda *a, **k: torch.device("cpu")
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(checkpointer, "get_current_device", lambda: "cpu")
    output = tmp_path / "checkpoints"
    config = TrainerConfig(
        lr=1e-4,
        num_epochs=1,
        save_path=str(output),
        checkpoint_freq=1,
        hidden_states_dtype=torch.bfloat16,
        scheduler_type="cosine",
        scheduler_total_steps=6,
    )
    run = StoredTrainer(adapter, config, loader)
    adapter.step_provider = lambda: run.global_step
    hook = adapter.register_forward_hook(
        lambda m, a, result: setattr(
            run,
            "coverage",
            run.coverage + int(result[2]["conditioned_pair_fraction_sum"]),
        )
    )
    run.run_training()
    hook.remove()
    assert (output / "0/model.safetensors").is_file()
    assert (output / "0/optimizer_state_dict.pt").is_file()
    assert (output / "epoch0_end").is_symlink()
    state = json.loads((output / "0/training_state.json").read_text())
    assert state == {"epoch": 0, "local_step": 0, "global_step": 2}
    adapter2 = load_model(output / "0")
    adapter2.configure_stored(server, pairs_per_batch=4, blocks_per_forward=2)
    resumed = StoredTrainer(
        adapter2, config._replace(num_epochs=2, resume_from_checkpoint=True), loader
    )
    assert resumed.global_step == 2 and resumed.current_epoch == 1
    adapter2.step_provider = lambda: resumed.global_step
    hook = adapter2.register_forward_hook(
        lambda m, a, result: setattr(
            resumed,
            "coverage",
            resumed.coverage + int(result[2]["conditioned_pair_fraction_sum"]),
        )
    )
    resumed.run_training()
    hook.remove()
    assert (output / "1/model.safetensors").is_file()
    assert (output / "epoch1_end").is_symlink()
    assert (
        json.loads((output / "1/training_state.json").read_text())["global_step"] == 4
    )
