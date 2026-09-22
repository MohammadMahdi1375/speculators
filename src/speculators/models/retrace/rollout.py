"""Real proposal/verification rounds shared by training and greedy evaluation."""

from dataclasses import dataclass

import torch

from .memory import carry_suffix


@dataclass
class Round:
    block: object
    target_logits: torch.Tensor
    valid: torch.Tensor
    accepted: int
    proposed: int
    committed: int
    conditioned: int


def _target_logits(target, head, hidden):
    # The optional projection belongs only to the explicit numerical reference
    # control. Ordinary local/remote targets retain their original batched head.
    project = getattr(target, "project", None)
    return head(hidden) if project is None else project(head, hidden)


def rollout(target, draft, prompt, response_length, eos, beta=lambda: 1.0, train=False):
    if response_length <= 0:
        return []
    if not prompt:
        raise ValueError("A nonempty prompt is required")
    target.reset()
    committed = list(prompt)
    with torch.no_grad():
        pref = target.states(committed)
        anchor = _target_logits(target, draft.lm_head, pref.scoring[:, -1]).argmax(-1)
    generated = [int(anchor[0])]
    memory = None
    while len(generated) < response_length and generated[-1] not in eos:
        prefix_length = len(committed)
        with torch.set_grad_enabled(train):
            block = draft(pref.auxiliary, anchor, memory, beta())
        k = block.token_ids.shape[1]
        path = committed + [int(anchor[0])] + block.token_ids[0].tolist()
        with torch.no_grad():
            verified = target.states(path)
            scores = verified.scoring[:, prefix_length : prefix_length + k]
            target_logits = _target_logits(target, draft.lm_head, scores)
            desired = target_logits.argmax(-1)
            accepted = int((desired == block.token_ids).long().cumprod(-1).sum())
            correction = (
                desired[:, accepted]
                if accepted < k
                else _target_logits(
                    target, draft.lm_head, verified.scoring[:, -1]
                ).argmax(-1)
            )
            next_memory = carry_suffix(
                block.hidden, scores, torch.tensor([accepted], device=anchor.device)
            )
            if getattr(draft.config, "pretrained_fingerprint", None):
                next_memory = next_memory.detached()
            remaining = response_length - len(generated)
            valid = torch.arange(k, device=anchor.device)[None] < remaining
            new_tokens = block.token_ids[0, :accepted].tolist() + [int(correction[0])]
            emitted = []
            for token in new_tokens:
                if len(emitted) >= remaining:
                    break
                emitted.append(token)
                if token in eos:
                    break
            if emitted and emitted[-1] in eos:
                valid &= torch.arange(k, device=anchor.device)[None] < len(emitted)
            actual_accepted = min(accepted, len(emitted))
            conditioned = (
                int(memory.valid.sum())
                if memory is not None and draft.config.retrace_enabled
                else 0
            )
        yield Round(
            block,
            target_logits,
            valid,
            actual_accepted,
            int(valid.sum()),
            len(emitted),
            conditioned,
        )
        generated.extend(emitted)
        committed += [int(anchor[0])] + block.token_ids[0, :accepted].tolist()
        # Auxiliary context ends BEFORE the correction/next anchor.
        pref = type(verified)(
            verified.auxiliary[:, : len(committed)],
            verified.scoring[:, : len(committed)],
        )
        anchor, memory = correction, next_memory
    return generated


def soft_ce_loss(round_, gamma=4.0):
    logq = round_.block.logits.float().log_softmax(-1)
    p = round_.target_logits.detach().float().softmax(-1)
    ce = -(p * logq).sum(-1)
    weights = (
        torch.exp(-torch.arange(ce.shape[1], device=ce.device).float() / gamma)[None]
        * round_.valid
    )
    loss = (ce * weights).sum() / weights.sum()
    tv = ((p - logq.exp()).abs().sum(-1) * 0.5 * weights).sum() / weights.sum()
    return loss, float(tv.detach())


@torch.no_grad()
def generate(target, draft, prompt, response_length, eos):
    iterator = rollout(target, draft, prompt, response_length, eos)
    rounds = []
    while True:
        try:
            round_ = next(iterator)
            rounds.append(
                {
                    "accepted": round_.accepted,
                    "proposed": round_.proposed,
                    "committed": round_.committed,
                    "conditioned": round_.conditioned,
                }
            )
        except StopIteration as end:
            return end.value, rounds


@torch.no_grad()
def target_greedy(target, draft, prompt, response_length, eos):
    target.reset()
    tokens, generated = list(prompt), []
    for _ in range(response_length):
        states = target.states(tokens)
        token = int(
            _target_logits(target, draft.lm_head, states.scoring[:, -1]).argmax(-1)
        )
        tokens.append(token)
        generated.append(token)
        if token in eos:
            break
    return generated
