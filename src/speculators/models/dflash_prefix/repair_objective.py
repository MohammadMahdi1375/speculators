"""Training-only repair-span credit for a causal, fixed-shortlist greedy head.

This is a research hypothesis, not a claim of novel cost-sensitive learning or
an unbiased estimator of online acceptance. All labels use the unchanged target
context. An uncovered target or an unknown context is an absorbing boundary.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


def repair_settings(settings=None):
    settings = dict(settings or {})
    out = dict(
        objective=settings.get("objective", "renewal"),
        ce_weight=float(settings.get("ce_weight", .25)),
        repair_weight=float(settings.get("repair_weight", .5)),
        protect_weight=float(settings.get("protect_weight", 1.0)),
        kd_weight=float(settings.get("kd_weight", .05)),
        renewal_weight=float(settings.get("renewal_weight", 1.0)),
        margin=float(settings.get("margin", .25)),
        span_cap=int(settings.get("span_cap", 8)),
        anchor_chunk_size=int(settings.get("anchor_chunk_size", 8)),
    )
    if out["objective"] not in {"renewal", "repair", "frontier", "ce"}:
        raise ValueError("objective must be renewal, repair, frontier, or ce")
    if not all(math.isfinite(v) and v >= 0 for k, v in out.items() if k != "objective"):
        raise ValueError("Repair settings must be finite and nonnegative")
    if out["ce_weight"] <= 0 or not 1 <= out["span_cap"] <= 15:
        raise ValueError("ce_weight must be positive; span_cap must be in [1,15]")
    if not 1 <= out["anchor_chunk_size"] <= 16:
        raise ValueError("anchor_chunk_size must be in [1,16]")
    return out


@torch.no_grad()
def prefix_credit(correct, usable):
    """Return first repairable error and its exact one-repair reference span.

correct[i] is evaluated conditional on the reference tokens before i. Until
the first error, that context equals the head's greedy context. If that error
is forced to the reference/target token, it still equals the reference until
the next error. Therefore the contiguous correct run after the first error
is exactly the recovered span in real arithmetic, without scoring wrong
contexts against stale labels. There is no claim of bitwise equality between
different BF16 kernel shapes; the caller uses the serving table/walk path.
"""
    if correct.shape != usable.shape or correct.ndim != 2:
        raise ValueError("correct and usable must have matching [blocks, proposals] shapes")
    usable = usable.bool().long().cumprod(-1).bool()
    hit = correct.bool() & usable
    accepted = hit.long().cumprod(-1).bool()
    reached = torch.cat((torch.ones_like(hit[:, :1]), accepted[:, :-1]), -1)
    frontier = reached & usable & ~hit
    continuation = torch.zeros(hit.shape[0], device=hit.device, dtype=torch.long)
    spans = []
    for i in range(hit.shape[1] - 1, -1, -1):
        spans.append(1 + continuation)
        continuation = torch.where(hit[:, i], 1 + continuation, 0)
    spans = torch.stack(spans[::-1], -1)
    return frontier, spans * frontier, accepted, reached


def candidate_margin(scores, labels):
    gold = scores.gather(-1, labels[..., None]).squeeze(-1)
    others = scores.masked_fill(F.one_hot(labels, scores.shape[-1]).bool(), -torch.inf)
    return gold - others.max(-1).values


def repair_loss_terms(scores, baseline, candidates, target_ids, valid, correct_context, settings):
    """Masked sums/counts; all discrete credit and baseline values are detached."""
    cfg = repair_settings(settings)
    scores, baseline = scores.float(), baseline.detach().float()
    matches = candidates.eq(target_ids[..., None])
    labels = matches.long().argmax(-1)
    context = valid.bool() & correct_context.bool()
    usable = (context & matches.any(-1)).long().cumprod(-1).bool()
    current = scores.detach().argmax(-1).eq(labels)
    frontier, spans, accepted, reached = prefix_credit(current, usable)
    _, _, baseline_accepted, _ = prefix_credit(baseline.argmax(-1).eq(labels), usable)
    block_valid = valid[:, 0].float()
    margin = candidate_margin(scores, labels)
    baseline_margin = candidate_margin(baseline, labels)
    # A baseline tie receives no fictitious positive margin guarantee.
    required_margin = baseline_margin.clamp(min=0, max=cfg["margin"])
    protection_cost = baseline_accepted.flip(-1).long().cumsum(-1).flip(-1)
    protection_cost = protection_cost.float() * baseline_accepted
    credit = (spans.clamp(max=cfg["span_cap"]).float()
              if cfg["objective"] in {"renewal", "repair"} else frontier.float())
    if cfg["objective"] == "ce":
        credit = credit * 0
    ce = -scores.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    kl = (baseline.softmax(-1) * (baseline.log_softmax(-1) - scores.log_softmax(-1))).sum(-1)
    terms = {
        "target_ce": ((ce * usable).sum(), usable.float().sum()),
        # Normalize by blocks, not repaired tokens: longer repair spans must
        # have larger gradient weight even for one eligible block on a rank.
        "repair_rank": ((F.softplus(cfg["margin"] - margin) * credit).sum(), block_valid.sum()),
        "protect_margin": ((F.relu(required_margin - margin) * protection_cost).sum(), protection_cost.sum()),
        "accepted_kl": ((kl * baseline_accepted).sum(), baseline_accepted.float().sum()),
    }
    stats = {
        "target_prefix_proxy": (((1 + accepted.sum(-1)) * block_valid).sum(), block_valid.sum()),
        "baseline_target_prefix_proxy": (((1 + baseline_accepted.sum(-1)) * block_valid).sum(), block_valid.sum()),
        "repair_span": (spans.float().sum(), frontier.float().sum()),
        "repairable_frontier_fraction": (frontier.float().sum(), block_valid.sum()),
        "shortlist_blocked_fraction": ((reached & context & ~matches.any(-1)).float().sum(), block_valid.sum()),
        "first_position_accuracy": ((current[:, 0] & usable[:, 0]).float().sum(), block_valid.sum()),
        "protected_choice_damage": ((~current & baseline_accepted).float().sum(), baseline_accepted.float().sum()),
        "candidate_target_recall": ((matches.any(-1) & context).float().sum(), context.float().sum()),
    }
    return terms, stats


def global_masked_loss(terms, settings):
    """DDP average of this loss equals the global masked mean of each term.

One packed denominator collective per training forward, after all chunks; even
empty ranks participate. No all-reduce of tensors carrying a gradient.
"""
    cfg = repair_settings(settings)
    names = ("target_ce", "repair_rank", "protect_margin", "accepted_kl")
    coefficients = (cfg["ce_weight"], cfg["repair_weight"], cfg["protect_weight"], cfg["kd_weight"])
    if "round_saving_rank" in terms:
        names += ("round_saving_rank",)
        coefficients += (cfg["renewal_weight"] if cfg["objective"] == "renewal" else 0.,)
    counts = torch.stack([terms[name][1].detach().float() for name in names])
    world = 1
    if torch.is_grad_enabled() and dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(counts)
    return sum(weight * terms[name][0] * world / counts[i].clamp_min(1)
               for i, (name, weight) in enumerate(zip(names, coefficients)))
