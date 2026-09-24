def _target_logits(target, head, hidden):
    project = getattr(target, "project", None)
    return head(hidden) if project is None else project(head, hidden)
