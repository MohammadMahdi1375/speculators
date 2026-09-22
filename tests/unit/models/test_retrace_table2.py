"""Guard benchmark comparability and the native FP16 memory lifecycle."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import speculators
from speculators.models.retrace.table2_benchmark import (
    acceptance,
    paired_summary,
    summarize_rows,
)
from speculators.models.retrace.table2_data import format_prompt


def metric(rounds, accepted, drafted):
    return [
        {"name": "vllm:spec_decode_" + name, "value": value}
        for name, value in zip(
            ("num_drafts", "num_accepted_tokens", "num_draft_tokens"),
            (rounds, accepted, drafted),
            strict=True,
        )
    ]


def row(tokens, seconds, speculation=None):
    return {
        "prompt_token_ids": [1, 2],
        "token_ids": tokens,
        "seconds": seconds,
        "speculation": speculation or acceptance([], []),
    }


def test_acceptance_excludes_warmup_and_exposes_bonus_convention():
    delta = acceptance(metric(8, 20, 120), metric(10, 26, 150))
    assert delta["num_drafts"] == 2
    assert delta["accepted_per_round"] == 3
    assert delta["acceptance_length_with_bonus"] == 4
    with pytest.raises(ValueError, match="reset"):
        acceptance(metric(8, 20, 120), metric(7, 19, 119))


def test_failed_greedy_comparison_cannot_report_a_speedup():
    a = [row([3, 4, 5], 3)]
    b = [row([3, 4, 6], 1, acceptance([], metric(1, 2, 15)))]
    left = {"results": a, "summary": summarize_rows(a, False)}
    right = {"results": b, "summary": summarize_rows(b, True)}
    result = paired_summary(left, right, 0)
    assert not result["comparison_valid"]
    assert result["speedup_tps"] is None
    assert result["mismatching_requests"] == [0]
    stochastic = paired_summary(left, right, 1)
    assert stochastic["greedy_parity"] is None
    assert stochastic["speedup_tps"] == 3


def test_evaluation_templates_exclude_reference_answers():
    assert "secret answer" not in format_prompt(
        "alpaca",
        {"instruction": "Question", "input": "Context", "output": "secret answer"},
    )
    assert format_prompt("mtbench", {"prompt": ["First", "Second"]}) == "First"
    prompt = format_prompt(
        "lcb",
        {
            "question_content": "Write it",
            "starter_code": "def f():",
            "private_test_cases": "secret",
        },
    )
    assert "secret" not in prompt
    assert "def f():" in prompt


def test_native_cache_stores_fp16_and_drops_first_rejection():
    root = Path(speculators.__file__).resolve().parents[3]
    path = root / "vllm/vllm/model_executor/models/retrace.py"
    if not path.is_file():
        pytest.skip("Requires the pinned three-repository workspace")
    spec = importlib.util.spec_from_file_location("retrace_native_fp16_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cache = module.ReTraceRequestCache(4, 2, storage_dtype=torch.float16)
    cache.begin_step(
        ["request"],
        torch.empty(0, 2),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        [0],
        torch.tensor([0]),
    )
    hidden = torch.arange(8).reshape(1, 4, 2).float().requires_grad_()
    positions = torch.tensor([[10, 11, 12, 13]])
    tokens = torch.tensor([[21, 22, 23, 24]])
    cache.capture(hidden, positions)
    cache.finish_step(tokens)
    assert cache._previous["request"].hidden.dtype == torch.float16
    assert not cache._previous["request"].hidden.requires_grad
    scores = torch.arange(8).reshape(4, 2).float() + 20
    cache.begin_step(
        ["request"], scores, positions[0], tokens[0], [4], torch.tensor([1])
    )
    memory = cache.inputs(torch.tensor([[12, 13, 14, 15]]), torch.bfloat16)
    assert memory.draft.dtype == memory.target.dtype == torch.float16
    assert memory.valid.tolist() == [[True, True, False, False]]
    torch.testing.assert_close(memory.draft[0, :2], hidden.detach()[0, 2:].half())
    torch.testing.assert_close(memory.target[0, :2], scores[2:].half())
    assert not cache.inputs(
        torch.tensor([[12, 13, 14, 15]]), torch.bfloat16
    ).valid.any()
