"""Training-only fixed-budget round-to-finish credit on contiguous anchors.

No serving scheduler or acceptance relaxation. Unit cost per verification round
is a latency proxy for fixed block size/concurrency, NOT measured A2 latency.
Known greedy reference segments terminate at every unverified label/context.
"""

import torch


@torch.no_grad()
def round_saving_credit(accepted, repair_span, known_next_token):
    """Evaluate one repaired block, then resume the current policy.

At anchor i, current transition is i + accepted[i] + 1, including the target
bonus/correction token. Correcting the first error moves it by repair_span[i].
    J[i] = 1 + J[next[i]], J[segment_end] = 0. The detached repair advantage is
    J[next[i]] - J[repaired_next[i]], but ONLY if both paths meet again at an
    observed valid anchor before the segment boundary. Their common future then
    cancels, so an arbitrary finite-window boundary cannot create repair credit.
    The advantage can be negative: later anchors can be harder.
We never train a wrong choice for that reason. Only its positive part supplements
the ordinary repair-span signal. The finite-window, unit-cost J is a proxy.
"""
    if accepted.ndim != 1 or accepted.shape != repair_span.shape or accepted.shape != known_next_token.shape:
        raise ValueError("Expected matching contiguous-anchor vectors")
    n = accepted.numel()
    if not n:
        raise ValueError("At least one contiguous anchor is required")
    known = known_next_token.bool()
    remaining = torch.zeros(n + 1, device=accepted.device, dtype=torch.long)
    for i in range(n - 1, -1, -1):
        remaining[i] = torch.where(known[i], 1 + remaining[i + 1], 0)
    index = torch.arange(n, device=accepted.device)
    end = index + remaining[:-1]
    step = accepted.long().clamp_min(0) + 1
    base_next = torch.minimum(index + step, end)
    repaired_next = torch.minimum(index + step + repair_span.long().clamp_min(0), end)
    rounds = torch.zeros(n + 1, device=accepted.device, dtype=torch.float32)
    reach = torch.eye(n + 1, device=accepted.device, dtype=torch.bool)
    for i in range(n - 1, -1, -1):
        continuation = rounds.gather(0, base_next[i:i + 1]).squeeze(0)
        rounds[i] = torch.where(known[i], 1 + continuation, 0.)
        reach[i] = reach[i] | (reach.index_select(0, base_next[i:i + 1]).squeeze(0) & known[i])
    difference = rounds.gather(0, base_next) - rounds.gather(0, repaired_next)
    common = reach.index_select(0, base_next)[:, :n] & reach.index_select(0, repaired_next)[:, :n]
    witnessed = (common & known[None]).any(-1)
    eligible = known & repair_span.gt(0) & repaired_next.gt(base_next) & witnessed
    difference = difference * eligible
    return difference, eligible, rounds[:-1]


def dense_anchors(loss_mask, num_anchors, block_size):
    """One contiguous window, sampled without consulting model predictions.

All objective arms use the same sampler and seed. Fixed width preserves the
base backbone's shape contract. Document/mask boundaries are applied by the
backbone and by the round-credit mask; no transition crosses those boundaries.
"""
    if loss_mask.ndim != 2 or loss_mask.shape[0] != 1:
        raise ValueError("Dense anchor training expects [1, sequence_length]")
    last_start = loss_mask.shape[1] - block_size - num_anchors
    if num_anchors < 1 or last_start < 0:
        raise ValueError("MAX_ANCHORS + block size must fit within the sequence length")
    valid_start = loss_mask[0, :last_start + 1].bool()
    priority = torch.rand(last_start + 1, device=loss_mask.device)
    start = priority.masked_fill(~valid_start, -1).argmax()
    anchors = start + torch.arange(num_anchors, device=loss_mask.device)
    return anchors, loss_mask[0, anchors].bool()
