"""CPU protocol tests only: these do not validate NPU graph capture or speed."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.models.dflash_prefix import compiled_inference as runtime_module
from speculators.models.dflash_prefix.compiled_selector import CompiledPrefixSelector


@pytest.fixture
def case(monkeypatch):
    for key in list(runtime_module.os.environ):
        if key.startswith("DFLASH_PREFIX_"):
            monkeypatch.delenv(key)
    torch.manual_seed(90)
    head = CompiledPrefixSelector(29, 12, rank=8, top_k=4,
                                  max_proposals=4, heads=2).eval()
    with torch.no_grad():
        head.head_weights.fill_(0.3)
        head.successor_projection.weight.normal_(std=0.2)
    hidden = torch.randn(1, 4, 12)
    ids = torch.arange(16).reshape(1, 4, 4)
    unary = torch.randn(1, 4, 4)
    anchors = torch.tensor([22])
    embedding = torch.randn(29, 12)
    return head, (hidden, ids, unary, anchors), embedding


class MockGraph:
    """Exercises control flow and faults; replays are ordinary CPU operations."""
    fault = None

    def __init__(self, head, logger):
        self.head = head
        self.calls = []

    def select(self, *inputs, embedding_weight, return_scores=False):
        self.calls.append(return_scores)
        tables = self.head.prepare_tables(*inputs, embedding_weight=embedding_weight)
        result = self.head.walk(tables, return_scores=return_scores)
        if self.fault == "production_tokens" and not return_scores:
            return result + 1
        if self.fault == "scored_tokens" and return_scores:
            return result[0] + 1, result[1]
        if self.fault == "score_drift" and return_scores:
            return result[0], result[1] + 0.1
        if self.fault == "nonfinite" and return_scores:
            return result[0], result[1] * float("nan")
        return result


def make_runtime(monkeypatch, head, *, mode="fast", check_steps=2):
    monkeypatch.setenv("DFLASH_PREFIX_INFERENCE", mode)
    monkeypatch.setenv("DFLASH_PREFIX_NPU_GRAPH", "1")
    monkeypatch.setenv("DFLASH_PREFIX_VALIDATE_STEPS", str(check_steps))
    monkeypatch.setattr(runtime_module, "CompiledPrefixGraph", MockGraph)
    return runtime_module.CompiledPrefixInferenceRuntime(head, logging.getLogger("compiled_test"))


def test_reference_ignores_graph_flag_and_retains_eager_tokens(case, monkeypatch):
    head, inputs, embedding = case
    runtime = make_runtime(monkeypatch, head, mode="reference")
    expected = head.greedy(*inputs, embedding_weight=embedding)
    assert runtime.graph is None
    assert torch.equal(runtime.select(*inputs, embedding_weight=embedding), expected)


def test_fast_checks_both_graph_outputs_then_stops_reference_execution(case, monkeypatch, caplog):
    head, inputs, embedding = case
    runtime = make_runtime(monkeypatch, head)
    with caplog.at_level(logging.INFO):
        first = runtime.select(*inputs, embedding_weight=embedding)
        second = runtime.select(*inputs, embedding_weight=embedding)
    assert torch.equal(first, second)
    assert runtime.checked == 2
    assert "COMPILED_PREFIX_PARITY_PASS" in caplog.text
    assert "steady-state fast path enabled" in caplog.text
    monkeypatch.setattr(runtime, "_eager", lambda *a, **k: pytest.fail("reference ran after warmup"))
    third = runtime.select(*inputs, embedding_weight=embedding)
    assert torch.equal(first, third)
    assert runtime.graph.calls == [True, False, True, False, False]


def test_validate_keeps_checking_after_initial_threshold(case, monkeypatch, caplog):
    head, inputs, embedding = case
    runtime = make_runtime(monkeypatch, head, mode="validate", check_steps=1)
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            runtime.select(*inputs, embedding_weight=embedding)
    assert runtime.checked == 3
    assert runtime.graph.calls == [True, False] * 3
    assert "steady-state fast path enabled" not in caplog.text


@pytest.mark.parametrize("fault", ["production_tokens", "scored_tokens", "score_drift", "nonfinite"])
def test_graph_fault_is_fatal_and_never_silently_falls_back(case, monkeypatch, fault):
    head, inputs, embedding = case
    runtime = make_runtime(monkeypatch, head)
    runtime.graph.fault = fault
    with pytest.raises(RuntimeError, match="COMPILED_PREFIX_PARITY_FAIL"):
        runtime.select(*inputs, embedding_weight=embedding)
    runtime.graph.fault = None
    with pytest.raises(RuntimeError, match="previously failed"):
        runtime.select(*inputs, embedding_weight=embedding)
    assert runtime.checked == 0


@pytest.mark.parametrize("name,value", [
    ("DFLASH_PREFIX_TOKEN_CACHE", "1"),
    ("DFLASH_PREFIX_FUSION", "local_unpaired"),
    ("DFLASH_PREFIX_SCORE_CHECK", "audit"),
    ("DFLASH_PREFIX_VALIDATE_STEPS", "0"),
])
def test_rejects_incompatible_legacy_options(case, monkeypatch, name, value):
    head, _, _ = case
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        runtime_module.CompiledPrefixInferenceRuntime(head, logging.getLogger("compiled_test"))


def test_graph_requires_npu_instead_of_silent_cpu_fallback(case):
    head, inputs, embedding = case
    graph = runtime_module.CompiledPrefixGraph(head, logging.getLogger("compiled_test"))
    with pytest.raises(ValueError, match="NPU device"):
        graph.select(*inputs, embedding_weight=embedding)


def test_graph_rejects_postcapture_weight_replacement(case):
    head, inputs, embedding = case
    graph = runtime_module.CompiledPrefixGraph(head, logging.getLogger("compiled_test"))
    graph.parameter_identity = tuple(runtime_module._weight_identity(p) for p in head.parameters())
    with torch.no_grad():
        head.head_weights.add_(1)
    with pytest.raises(RuntimeError, match="weights changed"):
        graph.select(*inputs, embedding_weight=embedding)


def test_vllm_checkpoint_adapter_constructs_new_head_without_legacy_tensors():
    """Exercise the real converter/factory without importing the vLLM engine."""
    from speculators.models.dflash_prefix.selector_v2 import make_prefix_head

    root = Path(__file__).resolve().parents[3]
    algos = root / "vllm/vllm/transformers_utils/configs/speculators/algos.py"
    function = next(node for node in ast.parse(algos.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "update_dflash_prefix")
    function.decorator_list = []
    namespace = {"update_dflash": lambda source, dest: dest.update(dflash_config={})}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(algos), "exec"), namespace)
    converted = {}
    namespace["update_dflash_prefix"]({
        "block_size": 5, "prefix_selector_kind": "compiled_prefix_v3",
        "prefix_rank": 8, "prefix_top_k": 4, "prefix_attention_heads": 2,
        "prefix_attention_layers": 1, "prefix_compiled_logit_cap": 6.0,
    }, converted)
    assert converted["architectures"] == ["DFlashPrefixDraftModel"]

    class BaseModel:
        def __init__(self, *, vllm_config, **kwargs):
            self.config = vllm_config

    model_path = root / "vllm/vllm/model_executor/models/qwen3_dflash_prefix.py"
    model_class = next(node for node in ast.parse(model_path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == "DFlashPrefixQwen3Model")
    namespace = {"DFlashQwen3Model": BaseModel, "make_prefix_head": make_prefix_head}
    exec(compile(ast.Module(body=[model_class], type_ignores=[]), str(model_path), "exec"), namespace)
    model = namespace["DFlashPrefixQwen3Model"](vllm_config=SimpleNamespace(
        vocab_size=29, hidden_size=12, dflash_config=converted["dflash_config"]
    ))
    assert type(model.prefix_head) is CompiledPrefixSelector
    assert model.prefix_head.compiled_logit_cap == 6.0
    assert not hasattr(model.prefix_head, "prefix_layers")
