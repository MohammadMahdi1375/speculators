"""Clean DFlash block targets with actual previous-round rejected memory.

Collect a complete target-verified continuation with fixed draft parameters.
Then replay its draft blocks for gradients, supervising every available mask
slot against that CLEAN continuation. Rejected-path logits are used only for
verification; they are never used as labels beyond the first rejection.

The paper does not release an exact anchor sampler. Here anchors are the live
decoding-round boundaries, so every nonempty memory has a real preceding round.
This design choice is recorded in the training provenance.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .memory import ReTraceMemory, carry_suffix
from .rollout import _target_logits


@dataclass
class TrainingTrajectory:
    tokens: list[int]
    prompt_length: int
    context: torch.Tensor
    positions: list[int]
    memories: list[ReTraceMemory]
    accepted: int
    proposed: int
    committed: int
    target_calls: int


@torch.no_grad()
def collect_trajectory(
    target,
    draft,
    prompt,
    response_length,
    eos,
    beta,
    *,
    performance_mode="reference",
    trace_storage="cpu",
):
    if not prompt or response_length < 1:
        raise ValueError("Need a nonempty prompt and a positive response budget")
    target.reset()
    if performance_mode not in ("reference", "cached") or trace_storage not in (
        "cpu",
        "device",
    ):
        raise ValueError("Unknown performance mode or trace storage")
    from .performance import DraftContextCache

    cache = DraftContextCache(draft) if performance_mode == "cached" else None
    storage = "cpu" if trace_storage == "cpu" else draft.device
    calls = target.calls
    prefix = list(prompt)
    pref = target.states(prefix)
    anchor = _target_logits(target, draft.lm_head, pref.scoring[:, -1]).argmax(-1)
    generated = [int(anchor[0])]
    memory = None
    positions, memories = [], []
    accepted_total = proposed_total = committed_total = 0
    clean_context = pref.auxiliary
    k = draft.block_size - 1
    while len(generated) < response_length and generated[-1] not in eos:
        position = len(prefix)
        if position != len(prompt) + len(generated) - 1:
            raise RuntimeError("Committed prefix and anchor position disagree")
        clean_context = pref.auxiliary[:, :position]
        if memory is None:
            empty = torch.zeros(
                1, k, draft.hidden_size, device=anchor.device, dtype=torch.float16
            )
            memory = ReTraceMemory(
                empty,
                torch.zeros_like(empty),
                torch.zeros(1, k, device=anchor.device, dtype=torch.bool),
            )
        positions.append(position)
        memories.append(memory.detached(device=storage))
        block = (
            draft(clean_context, anchor, memory, beta)
            if cache is None
            else cache.propose(clean_context, anchor, memory, beta)
        )
        proposal = block.token_ids[0].tolist()
        verified = target.states(prefix + [int(anchor[0])] + proposal)
        scores = verified.scoring[:, position : position + k]
        desired = _target_logits(target, draft.lm_head, scores).argmax(-1)
        accepted = int((desired == block.token_ids).long().cumprod(-1).sum())
        correction = (
            desired[:, accepted]
            if accepted < k
            else _target_logits(target, draft.lm_head, verified.scoring[:, -1]).argmax(
                -1
            )
        )
        memory = carry_suffix(
            block.hidden, scores, torch.tensor([accepted], device=anchor.device)
        ).detached()
        remaining = response_length - len(generated)
        emitted = []
        for token in proposal[:accepted] + [int(correction[0])]:
            if len(emitted) >= remaining:
                break
            emitted.append(token)
            if token in eos:
                break
        generated.extend(emitted)
        accepted_total += min(accepted, len(emitted))
        proposed_total += k
        committed_total += len(emitted)
        prefix += [int(anchor[0])] + proposal[:accepted]
        pref = type(verified)(
            verified.auxiliary[:, : len(prefix)], verified.scoring[:, : len(prefix)]
        )
        anchor = correction
    return TrainingTrajectory(
        list(prompt) + generated,
        len(prompt),
        clean_context.detach().to(device=storage),
        positions,
        memories,
        accepted_total,
        proposed_total,
        committed_total,
        target.calls - calls,
    )


def clean_labels(tokens, positions, proposals, device):
    sequence = torch.tensor(tokens, device=device, dtype=torch.long)
    index = positions[:, None] + torch.arange(1, proposals + 1, device=device)
    valid = index < sequence.numel()
    labels = sequence[index.clamp(max=sequence.numel() - 1)]
    return labels.masked_fill(~valid, -100), valid


def block_ce_sum(logits, labels, gamma=4.0):
    """DFlash exponential CE numerator; denominator is valid token COUNT.

    The frozen target's committed tokens are hard labels. There is no soft CE,
    TV regularizer, gate loss, or gradient through cached target/draft states.
    """
    if gamma <= 0 or labels.shape != logits.shape[:-1]:
        raise ValueError("Invalid block labels or gamma")
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    weights = torch.exp(
        -torch.arange(labels.shape[1], device=logits.device).float() / gamma
    )
    return (per_token * weights[None]).sum()


def backward_trajectory(
    draft,
    trace,
    beta,
    *,
    blocks_per_forward=4,
    gamma=4.0,
    optimized=False,
):
    """Accumulate one prompt's mean CE gradient using bounded block chunks."""
    if blocks_per_forward < 1:
        raise ValueError("blocks_per_forward must be positive")
    if not trace.positions:
        return {"loss": 0.0, "labels": 0, "rounds": 0, "conditioned": 0}
    device = draft.device
    positions = torch.tensor(trace.positions, device=device, dtype=torch.long)
    labels, valid = clean_labels(trace.tokens, positions, draft.block_size - 1, device)
    count = int(valid.sum())
    if count == 0:
        raise RuntimeError(
            "A recorded round must have at least one clean prediction label"
        )
    context = trace.context.to(device=device, dtype=draft.embed_tokens.weight.dtype)
    tokens = torch.tensor(trace.tokens, device=device, dtype=torch.long)
    loss_sum = torch.zeros((), device=device) if optimized else 0.0
    for start in range(0, len(trace.positions), blocks_per_forward):
        end = min(start + blocks_per_forward, len(trace.positions))
        chunk = trace.memories[start:end]
        forward_dtype = draft.embed_tokens.weight.dtype
        with torch.autocast(device_type=device.type, enabled=False):
            memory = ReTraceMemory(
                torch.cat([x.draft.to(device=device, dtype=forward_dtype) for x in chunk]),
                torch.cat([x.target.to(device=device, dtype=forward_dtype) for x in chunk]),
                torch.cat([x.valid.to(device=device) for x in chunk]),
            )
        # Later clean context is invisible to this chunk under the block mask.
        chunk_context = (
            context[:, : max(trace.positions[start:end])] if optimized else context
        )
        result = draft.training_blocks(
            chunk_context,
            tokens[positions[start:end]],
            positions[start:end],
            memory,
            beta,
        )
        loss = block_ce_sum(result.logits, labels[start:end], gamma) / (count + 1e-5)
        if not optimized and not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite clean block loss")
        loss.backward()
        loss_sum += loss.detach() if optimized else float(loss.detach())
    if optimized and not torch.isfinite(loss_sum):
        raise FloatingPointError("Nonfinite clean block loss")
    conditioned = (
        (
            int(torch.stack([x.valid.sum() for x in trace.memories]).sum())
            if optimized
            else sum(int(x.valid.sum()) for x in trace.memories)
        )
        if draft.config.retrace_enabled
        else 0
    )
    return {
        "loss": float(loss_sum),
        "labels": count,
        "rounds": len(trace.positions),
        "conditioned": conditioned,
    }
