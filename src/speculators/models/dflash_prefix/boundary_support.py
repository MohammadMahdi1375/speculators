"""Full-vocabulary support transfer from a refreshed training-only boundary.

No candidate is inserted into a shortlist and the student never reads refreshed
features. Support transfer is independent of the current shortlist, allowing a
gradient to an omitted target. Selector transfer remains conditional on actual
coverage. Generic privileged distillation is prior art; application-specific
novelty and serving gains remain hypotheses to be evaluated.
"""

import torch


def support_transfer_terms(student, old_teacher, refreshed_teacher, targets,
                           valid_context, covered, *, minimum_advantage=.05):
    """Compute detached-teacher full-vocabulary KL, not pool-renormalized KL.

Inputs are [anchors, proposals, vocabulary]. Only later slots with a known
matching greedy reference context are eligible. The refreshed UNARY teacher
must have the correct full-vocabulary argmax and better target NLL than the
frozen old-boundary unary teacher. No threshold uses a live student gradient.
This does not label a wrong-prefix continuation or claim it was visited.
"""
    if (student.ndim != 3 or student.shape != old_teacher.shape
            or student.shape != refreshed_teacher.shape
            or student.shape[:2] != targets.shape
            or targets.shape != valid_context.shape or targets.shape != covered.shape):
        raise ValueError("Support transfer needs matching [N,P,V] logits and [N,P] masks")
    log_student = student.float().log_softmax(-1)
    with torch.no_grad():
        old = old_teacher.detach().float()
        fresh = refreshed_teacher.detach().float()
        log_fresh = fresh.log_softmax(-1)
        old_nll = old.logsumexp(-1) - old.gather(-1, targets[..., None]).squeeze(-1)
        fresh_nll = -log_fresh.gather(-1, targets[..., None]).squeeze(-1)
        later = torch.arange(student.shape[1], device=student.device)[None] > 0
        eligible = (valid_context.bool() & later & fresh.argmax(-1).eq(targets)
                    & (fresh_nll + minimum_advantage < old_nll))
    element = (log_fresh.exp() * (log_fresh - log_student)).sum(-1)
    terms = {"support_kl": ((element * eligible).sum(), eligible.float().sum())}
    valid_count = valid_context.float().sum()
    miss = valid_context.bool() & ~covered.bool()
    stats = {
        "support_eligible_fraction": (eligible.float().sum(), valid_count),
        "support_teachable_miss_fraction": ((eligible & miss).float().sum(), valid_count),
        "support_teachable_miss_rate": ((eligible & miss).float().sum(), miss.float().sum()),
        "support_teacher_ce_advantage": (((old_nll - fresh_nll) * eligible).sum(), eligible.float().sum()),
    }
    return terms, stats
