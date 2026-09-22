"""Execute the actual patched hook methods on CPU, without loading NPU kernels.

This checks metadata indexing, not hardware execution. RETRACE_ROOT points at
the patched repo parent (the local test harness uses its reviewed source copy).
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from speculators.models.retrace.memory import ReTraceRequestCache


def production_proposer_methods():
    file = (
        Path(os.environ["RETRACE_ROOT"])
        / "vllm-ascend/vllm_ascend/spec_decode/retrace_proposer.py"
    )
    tree = ast.parse(file.read_text())
    cls = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "AscendReTraceProposer"
    )
    method_names = [x.name for x in cls.body if isinstance(x, ast.FunctionDef)]
    assert len(method_names) == len(set(method_names)), (
        "A later method silently shadows a hook"
    )
    names = {
        "retrace_begin_step",
        "retrace_model_kwargs",
        "retrace_capture",
        "retrace_finish_step",
    }
    methods = [
        x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names
    ]
    assert {x.name for x in methods} == names
    namespace = {"torch": torch}
    source = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), str(file), "exec"), namespace)
    return type("ProductionHooks", (), {name: namespace[name] for name in names})


def test_raw_target_rows_and_bonus_count_are_mapped_correctly():
    proposer = production_proposer_methods()()
    proposer.retrace_cache = ReTraceRequestCache(4, 2)
    proposer.model = SimpleNamespace(retrace_profile_run=False)
    proposer.dtype = torch.float32
    proposer.num_query_per_req = 5
    proposer.num_speculative_tokens = 4
    proposer.sample_from_anchor = False
    hidden = torch.arange(40.0).reshape(20, 2)
    positions = torch.zeros(20, dtype=torch.long)
    positions[7:12] = torch.arange(100, 105)
    proposer.retrace_begin_step(["A"], hidden, positions, None, [[9]])
    proposer.positions = torch.arange(100, 105)
    proposal_hidden = torch.arange(10.0).reshape(5, 2)
    proposer.retrace_capture(proposal_hidden, torch.tensor([1, 2, 3, 4]))
    proposer.retrace_finish_step(torch.tensor([[10, 11, 12, 13]]))
    metadata = SimpleNamespace(
        num_draft_tokens=[4],
        draft_token_ids=torch.tensor([10, 11, 12, 13]),
        logits_indices=torch.tensor([7, 8, 9, 10, 11]),
        target_logits_indices=torch.tensor([0, 1, 2, 3]),
    )
    # One accepted proposal + one correction = two sampled tokens, so A=1.
    proposer.retrace_begin_step(
        ["A"], hidden, positions, metadata, torch.tensor([[10, 44, -1, -1, -1]])
    )
    kwargs = proposer.retrace_model_kwargs(torch.arange(102, 107))
    memory = kwargs["retrace_memory"]
    assert memory.valid.tolist() == [[True, True, False, False]]
    torch.testing.assert_close(memory.target[0, :2], hidden[9:11])
    torch.testing.assert_close(memory.draft[0, :2], proposal_hidden[3:5])
    assert kwargs["retrace_runtime"]


def test_profiling_does_not_consume_a_live_request_memory():
    proposer = production_proposer_methods()()
    proposer.retrace_cache = ReTraceRequestCache(4, 2)
    proposer.model = SimpleNamespace(retrace_profile_run=True)
    assert proposer.retrace_model_kwargs(torch.zeros(7)) == {"retrace_runtime": True}
    proposer.retrace_capture(torch.zeros(7, 2), torch.zeros(7, dtype=torch.long))


def test_common_implementation_is_identical_between_training_and_serving():
    from speculators.models.retrace import memory as retrace

    root = Path(os.environ["RETRACE_ROOT"])
    serving = root / "vllm/vllm/model_executor/models/retrace.py"
    assert Path(retrace.__file__).read_bytes() == serving.read_bytes()
