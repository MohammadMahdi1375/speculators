"""CPU protocol/measurement tests. Graph mocks do not certify NPU execution."""

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.models.dflash_prefix import compiled_profile as profile
from speculators.models.dflash_prefix import compiled_inference as inference
from speculators.models.dflash_prefix.compiled_selector import CompiledPrefixSelector


class CpuStage:
    """Simulate persistent graph output addresses on the CPU."""

    def __init__(self, operation, backend):
        self.operation = operation
        self.output = operation()

    def replay(self):
        value = self.operation()
        if isinstance(value, torch.Tensor):
            self.output.copy_(value)
        else:
            for left, right in zip(self.output, value):
                if isinstance(left, torch.Tensor):
                    left.copy_(right)
        return self.output


class CpuProductionGraph:
    def __init__(self, head, inputs, embedding):
        self.head, self.embedding = head, embedding
        self.static = tuple(t.clone() for t in inputs)
        self.sync_count = 0
        self.backend = SimpleNamespace(synchronize=self.synchronize)
        self.outputs = {False: self.operation()}
        self.graphs = {False: SimpleNamespace(replay=self.replay)}

    def synchronize(self):
        self.sync_count += 1

    def operation(self):
        return self.head.walk(self.head.prepare_tables(*self.static, embedding_weight=self.embedding))

    def replay(self):
        self.outputs[False].copy_(self.operation())

    def select(self, *inputs, embedding_weight):
        assert embedding_weight is self.embedding
        for left, right in zip(self.static, inputs):
            left.copy_(right)
        self.replay()
        return self.outputs[False].clone()


@pytest.fixture
def case(monkeypatch, tmp_path):
    torch.manual_seed(101)
    head = CompiledPrefixSelector(29, 12, rank=8, heads=2, top_k=4, max_proposals=4).eval()
    with torch.no_grad():
        head.successor_projection.weight.normal_(std=.1)
        head.head_weights.fill_(.3)
    inputs = (torch.randn(1, 4, 12), torch.arange(16).reshape(1, 4, 4),
              torch.randn(1, 4, 4), torch.tensor([28]))
    embedding = torch.randn(29, 12)
    graph = CpuProductionGraph(head, inputs, embedding)
    monkeypatch.setattr(profile, "CapturedStage", CpuStage)
    profiler = profile.CompiledSelectorProfiler(
        head, graph, logging.getLogger("profile_test"), steps=2, output_dir=tmp_path, repeats=1)
    return head, inputs, embedding, graph, profiler


@torch.no_grad()
def test_profile_is_unarmed_until_http_warmup_then_bounded_and_preserves_policy(case, tmp_path):
    head, inputs, embedding, graph, profiler = case
    tokens = graph.select(*inputs, embedding_weight=embedding)
    profiler.observe(inputs, embedding, tokens)
    assert not profiler.rows and profiler.preparation is None and graph.sync_count == 0
    (tmp_path / "ARM").touch()
    weights = {name: value.clone() for name, value in head.state_dict().items()}
    for index in range(2):
        # Different real-input stand-ins must refresh stage and production graphs.
        current = (inputs[0] + index, inputs[1], inputs[2] - index, inputs[3])
        saved = tuple(value.clone() for value in current)
        tokens = graph.select(*current, embedding_weight=embedding)
        owned_tokens = tokens.clone()
        profiler.observe(current, embedding, tokens)
        assert torch.equal(tokens, owned_tokens)
        assert all(torch.equal(left, right) for left, right in zip(current, saved))
    assert all(torch.equal(weights[name], value) for name, value in head.state_dict().items())
    assert profiler.done and profiler.preparation is None and profiler.walk is None
    report = json.loads(profiler.report_path.read_text())
    assert report["status"] == "complete" and report["observed_tokens_equal"]
    assert report["blocks"] == 2 and report["max_table_score_error"] == 0
    assert report["table_logical_bytes_total"] > 0
    assert "target verification" in report["excludes"]
    assert all(row["samples"] == 2 for row in report["timings_ms"].values())
    synchronized = graph.sync_count
    profiler.observe(inputs, embedding, tokens)
    assert graph.sync_count == synchronized


@torch.no_grad()
def test_diagnostic_detects_mismatched_tokens_and_saves_failure(case, tmp_path):
    _, inputs, embedding, graph, profiler = case
    (tmp_path / "ARM").touch()
    tokens = graph.select(*inputs, embedding_weight=embedding)
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        profiler.observe(inputs, embedding, tokens + 1)
    report = json.loads(profiler.report_path.read_text())
    assert report["status"] == "failed" and not report["observed_tokens_equal"]
    assert "changes selected tokens" in report["error"]
    assert profiler.done


def test_measure_syncs_before_clock_and_after_operation(monkeypatch):
    events = []
    clocks = iter([10.0, 10.025])
    def clock():
        events.append("clock")
        return next(clocks)
    monkeypatch.setattr(profile.time, "perf_counter", clock)
    def operation():
        events.append("operation")
        return "kept_alive"
    result, milliseconds = profile.measure(operation, lambda: events.append("sync"))
    assert result == "kept_alive" and milliseconds == pytest.approx(25)
    assert events == ["sync", "clock", "operation", "sync", "clock"]


def test_runtime_profiles_only_after_production_gate(monkeypatch, case):
    head, inputs, embedding, graph, profiler = case
    for name in list(inference.os.environ):
        if name.startswith("DFLASH_PREFIX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DFLASH_PREFIX_INFERENCE", "fast")
    monkeypatch.setenv("DFLASH_PREFIX_NPU_GRAPH", "1")
    monkeypatch.setenv("DFLASH_PREFIX_VALIDATE_STEPS", "2")
    monkeypatch.setenv("DFLASH_PREFIX_SELECTOR_PROFILE_STEPS", "2")
    monkeypatch.setenv("DFLASH_PREFIX_PROFILE_DIR", str(profiler.directory))
    class Graph:
        def __init__(self, *args):
            pass
        def select(self, *args, embedding_weight, return_scores=False):
            return head.walk(head.prepare_tables(*args, embedding_weight=embedding_weight), return_scores=return_scores)
    monkeypatch.setattr(inference, "CompiledPrefixGraph", Graph)
    calls = []
    monkeypatch.setattr(profile, "CompiledSelectorProfiler", lambda *a, **kw: SimpleNamespace(
        done=False, observe=lambda *a: calls.append(a)))
    runtime = inference.CompiledPrefixInferenceRuntime(head, logging.getLogger("test"))
    for _ in range(2):
        runtime.select(*inputs, embedding_weight=embedding)
    assert not calls
    runtime.select(*inputs, embedding_weight=embedding)
    assert len(calls) == 1


def load_client():
    path = Path(__file__).resolve().parents[2] / "scripts/dflash_prefix_compiled/profile_server.py"
    spec = importlib.util.spec_from_file_location("profile_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_manifest_rejects_wrong_target_before_launch(tmp_path):
    client = load_client()
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text('{"model_type":"qwen3"}')
    manifest = tmp_path / "manifest.json"
    document = {"thinking": False, "target_config_sha256": "wrong", "datasets": {
        "gsm8k": {"results": [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3, 4]}]}}}
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="configuration differs"):
        client.frozen_prompts(manifest, target)
    document["target_config_sha256"] = client.digest(target / "config.json")
    manifest.write_text(json.dumps(document))
    assert client.frozen_prompts(manifest, target) == [[1, 2], [3, 4]]


def test_cleanup_targets_only_owned_process_group(monkeypatch):
    client = load_client()
    signals = []
    monkeypatch.setattr(client.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    waits = []
    process = SimpleNamespace(pid=123456, wait=lambda timeout: waits.append(timeout))
    client.stop_owned_server(process)
    assert signals == [(123456, client.signal.SIGTERM), (123456, client.signal.SIGKILL)]
    assert waits == [15, 15]


@pytest.mark.parametrize("failed_probe", [False, True])
def test_driver_arms_after_warmup_collects_report_and_closes_own_child(monkeypatch, tmp_path, failed_probe):
    client = load_client()
    draft, target = tmp_path / "draft", tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    (draft / "config.json").write_text(json.dumps({"prefix_selector_kind": "compiled_prefix_v3", "block_size": 16}))
    (target / "config.json").write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"thinking": False, "target_config_sha256": client.digest(target / "config.json"),
        "datasets": {"gsm8k": {"results": [{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}]}}}))
    output = tmp_path / "diagnostic"
    args = SimpleNamespace(spec_main=tmp_path, draft=draft, target=target, manifest=manifest,
                           output=output, steps=8, port=8213, npu="7", model="qwen3-4b-dflash")
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): assert address == ("127.0.0.1", 8213)
    monkeypatch.setattr(client.socket, "socket", Socket)
    monkeypatch.setattr(client, "collect_sources", lambda root: {"test_fixture": "sha"})
    process = SimpleNamespace(pid=123456, poll=lambda: None)
    def launch(command, *, env, stdout, stderr, start_new_session):
        assert start_new_session and env["MODE"] == "fast"
        assert env["DFLASH_PREFIX_PROFILE_DIR"] == str(output)
        stdout.write("COMPILED_PREFIX_PROFILE_READY\n")
        stdout.flush()
        return process
    monkeypatch.setattr(client.subprocess, "Popen", launch)
    monkeypatch.setattr(client, "read_json", lambda *a, **kw: {"data": [{"id": args.model}]})
    stopped = []
    monkeypatch.setattr(client, "stop_owned_server", lambda child: stopped.append(child))
    calls = []
    def complete(opener, base, model, ids, tokens):
        calls.append(ids)
        if len(calls) == 1:
            assert not (output / "ARM").exists()
            with (output / "server.log").open("a") as log:
                log.write("COMPILED_PREFIX_PARITY_PASS; steady-state fast path enabled\n")
        else:
            assert (output / "ARM").exists()
            if failed_probe:
                raise RuntimeError("fixture HTTP failure")
            (output / "selector_profile_123456.json").write_text(json.dumps({
                "status": "complete", "blocks": 8, "timings_ms": {"fixture": {"mean": 1.0}}}))
        return {"completion_tokens": tokens, "diagnostic_only": True}
    monkeypatch.setattr(client, "completion", complete)
    assert client.run(args) == int(failed_probe)
    assert calls == [[1], [2]] and stopped == [process]
    assert not (output / "ARM").exists()
    record = json.loads((output / "run.json").read_text())
    assert record["complete"] == (not failed_probe)
    assert (tmp_path / "diagnostic.zip").is_file()
