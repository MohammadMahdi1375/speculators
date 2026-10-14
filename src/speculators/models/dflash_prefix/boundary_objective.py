"""Training-only distillation across verification boundaries.

The teacher gets refreshed target context only during training. Student inputs
and serving stay unchanged. Generic distillation and joint training are prior
art; the research hypothesis is the target-checked, same-token/common-pool
transfer across boundaries, not an asserted acceptance or novelty guarantee.
"""
import math

import torch
import torch.distributed as dist


def boundary_settings(settings=None):
    s = dict(settings or {})
    layers = s.get("train_last_layers", 1)
    layers = "all" if layers == "all" else int(layers)
    out = dict(
        objective=s.get("objective", "boundary"),
        train_last_layers=layers,
        backbone_lr=float(s.get("backbone_lr", 3e-6)),
        head_lr=float(s.get("head_lr", 3e-5)),
        full_ce_weight=float(s.get("full_ce_weight", .25)),
        selector_ce_weight=float(s.get("selector_ce_weight", 1.0)),
        boundary_weight=float(s.get("boundary_weight", .5)),
        support_weight=float(s.get("support_weight", .25)),
        preserve_weight=float(s.get("preserve_weight", .1)),
        teacher_ce_improvement=float(s.get("teacher_ce_improvement", .05)),
        anchor_chunk_size=int(s.get("anchor_chunk_size", 4)),
    )
    if out["objective"] not in ("dual_boundary", "boundary", "control"):
        raise ValueError("OBJECTIVE must be dual_boundary, boundary or control")
    if not all(math.isfinite(v) and v >= 0 for k, v in out.items()
               if k not in {"objective", "train_last_layers"}):
        raise ValueError("Boundary settings must be finite and nonnegative")
    if layers != "all" and not 1 <= layers <= 2:
        raise ValueError("TRAIN_LAST_LAYERS must be 1, 2 or all")
    if min(out["backbone_lr"], out["head_lr"], out["full_ce_weight"], out["selector_ce_weight"]) <= 0:
        raise ValueError("Learning rates and both CE weights must be positive")
    if not 1 <= out["anchor_chunk_size"] <= 8:
        raise ValueError("ANCHOR_CHUNK_SIZE must be in [1,8]")
    return out


def dense_boundary_anchors(loss_mask, total_anchors, block_size, *, training):
    """An unwrapped contiguous window; tail anchors supply refreshed teachers.

Validation picks the middle eligible start without touching any RNG. Document
boundaries and response gaps are subsequently excluded by token-level masks.
"""
    if loss_mask.ndim != 2 or loss_mask.shape[0] != 1:
        raise ValueError("Expected a packed [1, sequence] loss mask")
    last_start = loss_mask.shape[1] - block_size - total_anchors + 1
    if total_anchors < 1 or last_start < 0:
        raise ValueError("Sequence must fit active anchors plus teacher tail and block")
    eligible = loss_mask[0, 1:last_start + 2].bool()
    if training:
        priority = torch.rand(eligible.shape, device=loss_mask.device)
        start = priority.masked_fill(~eligible, -1).argmax()
    else:
        order = eligible.long().cumsum(0)
        middle = (eligible.long().sum() + 1) // 2
        start = (eligible & order.eq(middle)).long().argmax()
    anchors = start + torch.arange(total_anchors, device=loss_mask.device)
    valid = loss_mask[0, anchors + 1].bool() & eligible.any()
    return anchors, valid


def refresh_rows(active_anchors, proposals, *, device):
    """Slot j (zero based) predicts a+j+1; its refreshed anchor is a+j."""
    return (torch.arange(active_anchors, device=device)[:, None]
            + torch.arange(proposals, device=device)[None, :])


def boundary_loss_terms(scores, old_scores, refreshed_scores, candidates,
                        targets, usable, refresh_valid, native_refresh_correct,
                        settings):
    """Loss sums/counts with detached teachers and no invented pool labels."""
    cfg = boundary_settings(settings)
    s = scores.float()
    old = old_scores.detach().float()
    fresh = refreshed_scores.detach().float()
    matched = candidates.eq(targets[..., None])
    labels = matched.long().argmax(-1)
    usable = usable.bool() & matched.any(-1)
    sl, ol, fl = s.log_softmax(-1), old.log_softmax(-1), fresh.log_softmax(-1)
    ce = -sl.gather(-1, labels[..., None]).squeeze(-1)
    old_ce = -ol.gather(-1, labels[..., None]).squeeze(-1)
    fresh_ce = -fl.gather(-1, labels[..., None]).squeeze(-1)
    later = torch.arange(s.shape[1], device=s.device)[None] > 0
    # The refreshed teacher must be correct with BOTH its natural candidate
    # pool and the student's pool. A common-pool top-1 alone is insufficient.
    eligible = (usable & refresh_valid & native_refresh_correct & later
                & fresh.argmax(-1).eq(labels)
                & (fresh_ce + cfg["teacher_ce_improvement"] < old_ce))
    preserve = usable & old.argmax(-1).eq(labels)
    boundary_kl = (fl.exp() * (fl - sl)).sum(-1)
    preserve_kl = (ol.exp() * (ol - sl)).sum(-1)
    old_hit = old.argmax(-1).eq(labels) & usable
    reached = torch.cat((torch.ones_like(old_hit[:, :1]), old_hit.long().cumprod(-1)[:, :-1].bool()), -1)
    frontier = reached & usable & ~old_hit
    terms = {
        "selector_ce": ((ce * usable).sum(), usable.float().sum()),
        "boundary_kl": ((boundary_kl * eligible).sum(), eligible.float().sum()),
        "preserve_kl": ((preserve_kl * preserve).sum(), preserve.float().sum()),
    }
    stats = {
        "refresh_eligible_fraction": (eligible.float().sum(), usable.float().sum()),
        "refresh_rescue_fraction": ((eligible & ~old.argmax(-1).eq(labels)).float().sum(), usable.float().sum()),
        "refresh_ce_advantage": (((old_ce - fresh_ce) * eligible).sum(), eligible.float().sum()),
        "protected_choice_damage": ((preserve & ~s.argmax(-1).eq(labels)).float().sum(), preserve.float().sum()),
        "refresh_teachable_frontier_fraction": ((frontier & eligible).float().sum(), usable[:, 0].float().sum()),
    }
    return terms, stats


def boundary_global_loss(terms, settings):
    """One denominator collective; correct global masked means under DDP."""
    s = boundary_settings(settings)
    names = ["full_ce", "selector_ce", "boundary_kl", "preserve_kl"]
    weights = [s["full_ce_weight"], s["selector_ce_weight"],
               s["boundary_weight"] if s["objective"] != "control" else 0., s["preserve_weight"]]
    if "support_kl" in terms:
        names.append("support_kl")
        weights.append(s["support_weight"] if s["objective"] == "dual_boundary" else 0.)
    denominators = torch.stack([terms[k][1].detach().float() for k in names])
    world = 1
    if torch.is_grad_enabled() and dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(denominators)
    return sum(w * terms[k][0] * world / denominators[i].clamp_min(1)
               for i, (k, w) in enumerate(zip(names, weights)))
