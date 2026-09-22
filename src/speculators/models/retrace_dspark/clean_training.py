"""Clean DSpark block targets with actual previous-round rejected memory.

Collect a complete target-verified continuation with fixed draft parameters.
Then replay blocks with clean Markov previous-token inputs and clean target
distributions. Retain DSpark CE/TV and confidence supervision. Actual rejected
paths supply conditioning memory only, never labels after the first rejection.

Training anchors here are the live decoding-round boundaries, so every
nonempty memory has a real preceding round. This implementation choice is
recorded in the training provenance.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .memory import ReTraceMemory, carry_suffix


def _target_logits(target, head, states):
    # LocalTarget.scoring is already normalized by the frozen final target norm.
    return head(states)


@dataclass
class TrainingTrajectory:
    tokens: list[int]
    prompt_length: int
    context: torch.Tensor
    scoring: torch.Tensor
    positions: list[int]
    memories: list[ReTraceMemory]
    accepted: int
    proposed: int
    committed: int
    target_calls: int


@torch.no_grad()
def collect_trajectory(target, draft, prompt, response_length, eos, beta):
    if not prompt or response_length < 1:
        raise ValueError("Need a nonempty prompt and a positive response budget")
    target.reset()
    calls = target.calls
    prefix = list(prompt)
    pref = target.states(prefix)
    anchor = _target_logits(target, draft.lm_head, pref.scoring[:, -1]).argmax(-1)
    generated = [int(anchor[0])]
    memory = None
    positions, memories = [], []
    accepted_total = proposed_total = committed_total = 0
    clean_context = pref.auxiliary
    k = draft.block_size
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
        memories.append(memory.detached(device="cpu"))
        block = draft(clean_context, anchor, memory, beta)
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
    # Crop any remaining rejected suffix and obtain CLEAN target distributions.
    # This call has no gradient and no lookahead is exposed to draft attention.
    clean = target.states(list(prompt) + generated)
    return TrainingTrajectory(
        list(prompt) + generated,
        len(prompt),
        clean.auxiliary.detach().cpu(),
        clean.scoring.detach().cpu(),
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


def block_loss_sum(
    logits,
    target_logits,
    confidence_logits,
    labels,
    *,
    ce_weight=0.1,
    tv_weight=0.9,
    confidence_alpha=1.0,
    gamma=4.0,
):
    """DSpark loss on clean tokens: .1 hard CE + .9 TV + confidence BCE.

    All terms use exp(-j/gamma) and the valid-token count denominator. The
    confidence target is detached overlap sum(min(p,q)), as in stock DSpark.
    """
    if (
        gamma <= 0
        or labels.shape != logits.shape[:-1]
        or target_logits.shape != logits.shape
    ):
        raise ValueError("Invalid block loss shapes or gamma")
    scores = logits.float()
    ce = F.cross_entropy(
        scores.reshape(-1, scores.shape[-1]),
        labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    target_probs = target_logits.detach().float().softmax(-1)
    overlap = torch.minimum(scores.softmax(-1), target_probs).sum(-1)
    confidence = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), overlap.detach().clamp(0, 1), reduction="none"
    )
    weights = torch.exp(
        -torch.arange(labels.shape[1], device=logits.device).float() / gamma
    )
    loss = ce_weight * ce + tv_weight * (1 - overlap) + confidence_alpha * confidence
    return (loss * weights[None] * (labels != -100)).sum()


def backward_trajectory(draft, trace, beta, *, blocks_per_forward=4, gamma=4.0):
    """Accumulate one prompt's mean DSpark gradient using bounded block chunks."""
    if blocks_per_forward < 1:
        raise ValueError("blocks_per_forward must be positive")
    if not trace.positions:
        return {"loss": 0.0, "labels": 0, "rounds": 0, "conditioned": 0}
    device = draft.device
    positions = torch.tensor(trace.positions, device=device, dtype=torch.long)
    labels, valid = clean_labels(trace.tokens, positions, draft.block_size, device)
    count = int(valid.sum())
    if count == 0:
        raise RuntimeError(
            "A recorded round must have at least one clean prediction label"
        )
    context = trace.context.to(device=device, dtype=draft.embed_tokens.weight.dtype)
    tokens = torch.tensor(trace.tokens, device=device, dtype=torch.long)
    scoring = trace.scoring.to(device=device, dtype=draft.embed_tokens.weight.dtype)
    loss_sum = 0.0
    for start in range(0, len(trace.positions), blocks_per_forward):
        end = min(start + blocks_per_forward, len(trace.positions))
        chunk = trace.memories[start:end]
        forward_dtype = draft.embed_tokens.weight.dtype
        memory = ReTraceMemory(
            torch.cat([x.draft.to(device=device, dtype=forward_dtype) for x in chunk]),
            torch.cat([x.target.to(device=device, dtype=forward_dtype) for x in chunk]),
            torch.cat([x.valid for x in chunk]).to(device),
        )
        score_index = positions[start:end, None] + torch.arange(
            draft.block_size, device=device
        )
        score_index = score_index.clamp(max=tokens.numel() - 1)
        previous_tokens = tokens[score_index]
        result = draft.training_blocks(
            context,
            tokens[positions[start:end]],
            positions[start:end],
            memory,
            beta,
            previous_token_ids=previous_tokens,
        )
        with torch.no_grad():
            targets = draft.lm_head(scoring[0, score_index])
        loss = block_loss_sum(
            result.logits,
            targets,
            result.confidence_logits,
            labels[start:end],
            ce_weight=draft.config.loss_weights.get("ce", 0),
            tv_weight=draft.config.loss_weights.get("tv", 0),
            confidence_alpha=draft.config.confidence_head_alpha,
            gamma=gamma,
        ) / (count + 1e-8)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite clean block loss")
        loss.backward()
        loss_sum += float(loss.detach())
    return {
        "loss": loss_sum,
        "labels": count,
        "rounds": len(trace.positions),
        "conditioned": sum(int(x.valid[:, 1:].sum()) for x in trace.memories)
        if draft.config.retrace_enabled
        else 0,
    }
