"""Training-only constrained AdamW updates for the compiled selector.

This is an experimental, optional selector update rule, not a serving gate.
Replay holds detached training features and natural candidate lists. Backbone
updates are NOT constrained, and a replay check is not a population guarantee.
Projection uses the *actual optimizer displacement*, including preconditioning,
momentum and weight decay. Greedy replay checks use BF16 exported head weights.
"""
import copy

import torch
import torch.distributed as dist


FIELDS = ("hidden", "candidates", "unary", "anchors", "reference", "target", "valid")


def bf16_replay_copy(module):
    """An isolated inference copy; never change live DDP parameter metadata.

    Temporarily changing ``Parameter.data`` from FP32 to BF16 invalidates its
    gradient accumulator even when the original storage is restored. DDP's
    hooks then stop reducing those gradients on the next backward. Supply the
    replacement Parameters to deepcopy's memo instead, avoiding an additional
    FP32 copy. Buffers keep their original precision.
    """
    memo = {
        id(p): torch.nn.Parameter(p.detach().to(dtype=torch.bfloat16, copy=True),
                                  requires_grad=False)
        for p in module.parameters()
    }
    return copy.deepcopy(module, memo).eval()


def distributed_sum(value):
    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def target_prefix_mask(tokens, reference, target, valid):
    """Count only target-greedy labels whose earlier cached context agrees.

At the first reference/target disagreement we can score that next token, but
not its continuation: cached target logits then have a different context.
"""
    earlier = torch.cat((torch.ones_like(valid[:, :1]), reference[:, :-1].eq(target[:, :-1])), 1)
    reachable = earlier.long().cumprod(1).bool()
    return (tokens.eq(target) & valid & reachable).long().cumprod(1).bool()


class FeatureReplay:
    """Bounded rank-local FIFO; no validation/test examples or target weights."""
    def __init__(self, capacity=32, add_anchors=4):
        if capacity < 1 or add_anchors < 1 or add_anchors > capacity:
            raise ValueError("Invalid replay capacity/add_anchors")
        self.capacity, self.add_anchors, self.data = capacity, add_anchors, None

    @torch.no_grad()
    def add(self, features):
        eligible = features["valid"][:, 0].nonzero().flatten()
        if eligible.numel() == 0:
            return
        count = min(self.add_anchors, eligible.numel())
        pick = eligible[torch.linspace(0, eligible.numel() - 1, count, device=eligible.device).long()]
        fresh = {k: features[k].detach().index_select(0, pick).clone() for k in FIELDS}
        if self.data is None:
            self.data = fresh
        else:
            self.data = {k: torch.cat((self.data[k], fresh[k]), 0)[-self.capacity:].contiguous() for k in FIELDS}


def flatten_tensors(values):
    return torch.cat([x.reshape(-1).float() for x in values])


@torch.no_grad()
def assign_vector(parameters, vector):
    offset = 0
    for p in parameters:
        p.copy_(vector[offset:offset + p.numel()].view_as(p))
        offset += p.numel()
    if offset != vector.numel():
        raise ValueError("Parameter/vector size mismatch")


def project_displacement(delta, gradient):
    """Euclidean projection onto g^T delta <= 0; g is a global mean gradient."""
    inner = torch.dot(delta, gradient)
    norm2 = torch.dot(gradient, gradient)
    coefficient = inner.clamp_min(0) / norm2.clamp_min(1e-30)
    return delta - coefficient * gradient, inner, norm2


class SelectorUpdateGuard:
    def __init__(self, *, capacity=32, add_anchors=4, max_backtracks=3, chunk_size=4):
        self.replay = FeatureReplay(capacity, add_anchors)
        if max_backtracks < 0 or chunk_size < 1:
            raise ValueError("Invalid backtracking/chunk size")
        self.max_backtracks, self.chunk_size = max_backtracks, chunk_size

    def _chunks(self, features):
        if features is not None:
            for start in range(0, features["hidden"].shape[0], self.chunk_size):
                yield {k: v[start:start + self.chunk_size] for k, v in features.items()}

    @torch.no_grad()
    def prefix_masks(self, head, embedding):
        parts = []
        replay_head = bf16_replay_copy(head)
        with torch.autocast(embedding.device.type, dtype=torch.bfloat16):
            for f in self._chunks(self.replay.data):
                chosen = replay_head.greedy(f["hidden"], f["candidates"], f["unary"], f["anchors"],
                                     embedding_weight=embedding)
                parts.append(target_prefix_mask(chosen, f["reference"], f["target"], f["valid"]))
        if parts:
            return torch.cat(parts)
        return torch.zeros((0, head.max_proposals), device=embedding.device, dtype=torch.bool)

    def protection_gradient(self, head, embedding, masks, parameters):
        """Use a temporary head copy, then explicitly reduce its gradients.

        The copy keeps auxiliary autograd completely outside DDP's reducer:
        no second backward hooks run on the live model's Parameters. The
        optimizer displacement and returned gradient use identical ordering.
        """
        count = distributed_sum(masks.float().sum())
        gradient = torch.zeros(sum(p.numel() for p in parameters), device=embedding.device, dtype=torch.float32)
        shadow = copy.deepcopy(head)
        names = {id(p): name for name, p in head.named_parameters()}
        shadow_named = dict(shadow.named_parameters())
        shadow_parameters = [shadow_named[names[id(p)]] for p in parameters]
        offset = 0
        for f in self._chunks(self.replay.data):
            local_mask = masks[offset:offset + f["hidden"].shape[0]]
            offset += f["hidden"].shape[0]
            with torch.enable_grad(), torch.autocast(embedding.device.type, dtype=torch.bfloat16):
                scores = shadow.score_teacher_prefix(f["hidden"], f["candidates"], f["unary"],
                                                  f["anchors"], f["reference"], embedding_weight=embedding)
                match = f["candidates"].eq(f["target"][..., None])
                if bool((local_mask & ~match.any(-1)).any()):
                    raise RuntimeError("Protected target missing from its natural shortlist")
                labels = match.long().argmax(-1)
                ce = -scores.float().log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
                loss = (ce * local_mask).sum() / count.clamp_min(1)
                grads = torch.autograd.grad(loss, shadow_parameters, allow_unused=True)
            gradient.add_(flatten_tensors([torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]))
        gradient = distributed_sum(gradient)
        return gradient, count

    def step(self, head, optimizer, embedding):
        """All ranks call together, including ranks with no eligible replay.

On acceptance, Adam moments retain the proposed direction (as with step-size
backtracking). On complete rejection, restore parameters AND Adam state.
Only the selector optimizer participates; backbone optimizers run normally.
"""
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if {id(p) for p in parameters} != {id(p) for p in head.parameters() if p.requires_grad}:
            raise ValueError("Guard optimizer must contain exactly the trainable selector")
        if any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("Guard needs FP32 selector master parameters")
        before_mask = self.prefix_masks(head, embedding)
        gradient, count = self.protection_gradient(head, embedding, before_mask, parameters)
        report = dict(active=1., protected_positions=float(count.cpu()), projected=0., rejected=0.,
                      accepted_scale=1., proposed_lost_tokens=0., accepted_lost_tokens=0., backtracks=0.)
        if count.item() == 0:
            optimizer.step()
            return report
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Nonfinite global protection gradient")
        before = flatten_tensors([p.detach() for p in parameters])
        state = copy.deepcopy(optimizer.state_dict())
        try:
            optimizer.step()
            delta = flatten_tensors([p.detach() for p in parameters]) - before
            if not bool(torch.isfinite(delta).all()):
                raise RuntimeError("Nonfinite selector optimizer displacement")
            delta, inner, _ = project_displacement(delta, gradient)
            report["projected"] = float((inner > 0).cpu())
            before_lengths = before_mask.sum(-1)
            for attempt in range(self.max_backtracks + 1):
                scale = 2. ** -attempt
                assign_vector(parameters, before + scale * delta)
                after_lengths = self.prefix_masks(head, embedding).sum(-1)
                lost = distributed_sum((before_lengths - after_lengths).clamp_min(0).float().sum())
                if attempt == 0:
                    report["proposed_lost_tokens"] = float(lost.cpu())
                if lost.item() == 0:
                    report.update(accepted_scale=scale, backtracks=float(attempt))
                    return report
            assign_vector(parameters, before)
            optimizer.load_state_dict(state)
            report.update(rejected=1., accepted_scale=0., backtracks=float(self.max_backtracks))
            return report
        except BaseException:
            assign_vector(parameters, before)
            optimizer.load_state_dict(state)
            raise
