"""Behavioral checks for suffix boundaries, request isolation, and learned fusion."""

import pytest
import torch

from speculators.models.retrace.memory import (
    ReTraceConditioner,
    ReTraceMemory,
    ReTraceRequestCache,
    carry_suffix,
    condition_query_block,
)


@pytest.mark.parametrize(
    "accepted,expected", [(0, [1, 2, 3]), (1, [2, 3]), (3, []), (4, [])]
)
def test_suffix_excludes_first_rejection_and_detaches(accepted, expected):
    draft = torch.arange(4.0).reshape(1, 4, 1).requires_grad_()
    target = (draft + 100).detach().requires_grad_()
    memory = carry_suffix(draft, target, torch.tensor([accepted]))
    assert memory.draft[0, memory.valid[0], 0].tolist() == expected
    assert memory.target[0, memory.valid[0], 0].tolist() == [100 + i for i in expected]
    assert not memory.draft.requires_grad and not memory.target.requires_grad
    assert memory.draft[~memory.valid].count_nonzero() == 0


def test_variable_verified_length_never_reuses_unverified_padding():
    x = torch.ones(3, 5, 2)
    memory = carry_suffix(x, x, torch.tensor([0, 1, -1]), torch.tensor([2, 2, 0]))
    assert memory.valid.sum(1).tolist() == [1, 0, 0]


def test_zero_initialization_preserves_anchor():
    torch.manual_seed(42)
    module = ReTraceConditioner(4)
    e = torch.randn(1, 5, 4)
    k = 4
    memory = ReTraceMemory(
        torch.randn(1, k, 4), torch.randn(1, k, 4), torch.ones(1, k, dtype=torch.bool)
    )
    initial = condition_query_block(module, e, memory)
    assert torch.equal(initial, e)
    with torch.no_grad():
        module.value.weight.copy_(torch.eye(4))
    result = condition_query_block(module, e, memory)
    assert torch.equal(result[:, 0], e[:, 0])
    assert not torch.equal(result[:, 1], e[:, 1])


def test_gradients_learn_current_fusion_without_cross_round_backprop():
    torch.manual_seed(1)
    module = ReTraceConditioner(4)
    e = torch.randn(1, 3, 4, requires_grad=True)
    m = torch.randn(1, 3, 4, requires_grad=True)
    u = torch.randn(1, 3, 4, requires_grad=True)
    memory = ReTraceMemory(m, u, torch.ones(1, 3, dtype=torch.bool))
    module(e, memory).square().sum().backward()
    assert module.value.weight.grad.abs().sum() > 0
    assert m.grad is None and u.grad is None
    assert e.grad is not None
    # Wv=0 initially blocks Wc/Wg gradients. After Wv learns, both must learn.
    with torch.no_grad():
        module.value.weight.add_(-0.1 * module.value.weight.grad)
    module.zero_grad()
    module(e, memory).square().sum().backward()
    assert module.correction.weight.grad.abs().sum() > 0
    assert module.gate.weight.grad.abs().sum() > 0


def seed_cache(cache, ids, offsets):
    cache.begin_step(
        ids,
        torch.zeros(0, 2),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        [0] * len(ids),
        torch.zeros(len(ids), dtype=torch.long),
    )
    positions = torch.stack([torch.arange(x, x + 4) for x in offsets])
    h = torch.stack(
        [torch.arange(8).reshape(4, 2).float() + 100 * i for i in range(len(ids))]
    )
    tokens = torch.stack(
        [torch.arange(10 + i * 10, 14 + i * 10) for i in range(len(ids))]
    )
    cache.capture(h, positions)
    cache.finish_step(tokens)
    return h, positions, tokens


def test_reordered_requests_keep_their_own_trajectory_and_expire():
    cache = ReTraceRequestCache(4, 2)
    h, positions, tokens = seed_cache(cache, ["A", "B"], [20, 50])
    order = [1, 0]
    cache.begin_step(
        ["B", "A"],
        torch.ones(8, 2),
        positions[order].flatten(),
        tokens[order].flatten(),
        [4, 4],
        torch.tensor([0, 1]),
    )
    next_positions = torch.stack((torch.arange(51, 55), torch.arange(22, 26)))
    memory = cache.inputs(next_positions, torch.float32)
    assert memory.valid.sum(1).tolist() == [3, 2]
    torch.testing.assert_close(memory.draft[0, :3], h[1, 1:])
    torch.testing.assert_close(memory.draft[1, :2], h[0, 2:])
    assert not cache.inputs(next_positions, torch.float32).valid.any()


@pytest.mark.parametrize("bad", ["tokens", "positions", "identity"])
def test_stale_trajectory_is_never_injected(bad):
    cache = ReTraceRequestCache(4, 2)
    _, positions, tokens = seed_cache(cache, ["A"], [20])
    if bad == "tokens":
        tokens = tokens + 1
    if bad == "positions":
        positions = positions + 1
    ids = ["new-request"] if bad == "identity" else ["A"]
    cache.begin_step(
        ids,
        torch.ones(4, 2),
        positions.flatten(),
        tokens.flatten(),
        [4],
        torch.tensor([0]),
    )
    assert not cache.inputs(torch.arange(21, 25)[None], torch.float32).valid.any()


def test_scheduler_skipped_request_loses_old_memory():
    cache = ReTraceRequestCache(4, 2)
    _, positions, tokens = seed_cache(cache, ["A"], [20])
    cache.begin_step(
        [],
        torch.zeros(0, 2),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        [],
        torch.zeros(0, dtype=torch.long),
    )
    cache.begin_step(
        ["A"],
        torch.ones(4, 2),
        positions.flatten(),
        tokens.flatten(),
        [4],
        torch.tensor([0]),
    )
    assert not cache.inputs(torch.arange(21, 25)[None], torch.float32).valid.any()
