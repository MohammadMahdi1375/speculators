"""Full-epoch scratch training of the existing compiled_prefix_v3 architecture."""
import torch
from torch.utils.checkpoint import checkpoint

from speculators.model import SpeculatorModel
from speculators.models.dflash.metrics import compute_metrics
from .core import DFlashPrefixDraftModel, reachable_greedy_prefix
from .selector_v2 import unary_topk
from .joint_guard import SelectorUpdateGuard, target_prefix_mask


def ramp(step, start, duration):
    return max(0., min(1., (step - start) / max(1, duration)))


def selector_weights(valid, covered, reachable, focus, floor):
    """Dense covered-position supervision remains at all training stages."""
    return (valid & covered).float() * (1 - focus * (1 - floor) * (~reachable).float())


@SpeculatorModel.register("dflash_prefix_joint_scratch")
class JointCompiledScratch(DFlashPrefixDraftModel):
    def configure_joint(self, settings):
        if self.config.prefix_selector_kind != "compiled_prefix_v3" or self.config.prefix_freeze_backbone:
            raise ValueError("Joint scratch requires compiled_prefix_v3 and an unfrozen backbone")
        self.joint_settings = dict(settings)
        self.joint_step = 0
        self.joint_metrics = None
        self.config.prefix_detach_backbone = False
        self.config.prefix_loss_scope = "all"
        self.config.prefix_validation_gate_scales = []
        self.selector_guard = (SelectorUpdateGuard(capacity=settings["guard_replay_anchors"],
            add_anchors=settings["guard_add_anchors"], max_backtracks=settings["guard_backtracks"],
            chunk_size=settings["guard_chunk_size"]) if settings["guard_mode"] == "project" else None)
        return self

    def save_pretrained(self, save_directory, *, state_dict=None, **kwargs):
        result = super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)
        self.config.architectures = ["DFlashPrefixDraftModel"]
        self.config.save_pretrained(save_directory)
        return result

    def _scores(self, hidden, candidates, unary, anchor, reference):
        return self.prefix_head.score_teacher_prefix(hidden, candidates, unary, anchor, reference,
                                                    embedding_weight=self.embed_tokens.weight)

    def forward(self, hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                document_ids, position_ids=None, loss_config=None, gamma=4., max_anchors=128,
                per_position_loss_weight="fixed-exp-decay", dpace_alpha=.5, **kwargs):
        if not hasattr(self, "joint_settings"):
            raise RuntimeError("configure_joint is required before training")
        s = self.joint_settings
        hidden, unary, targets, aligned, indices = self._backbone_forward(
            hidden_states, input_ids, loss_mask, verifier_last_hidden_states, document_ids,
            position_ids, max_anchors=max_anchors, **kwargs)
        backbone_loss, metrics = compute_metrics(unary, targets, aligned, self.block_size,
            gamma=gamma, loss_config=loss_config, per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha, sample_from_anchor=False)
        metrics = {"unary_" + k: v for k, v in metrics.items()}
        n, p = indices.numel() // self.block_size, self.block_size - 1
        ids = input_ids[0, indices].reshape(n, self.block_size)
        docs = document_ids[0, indices].reshape(n, self.block_size)
        valid = (aligned.reshape(n, self.block_size)[:, 1:].bool() & docs[:, 1:].eq(docs[:, :1])).long().cumprod(1).bool()
        hidden = hidden.reshape(n, self.block_size, -1)[:, 1:]
        full_unary = unary.reshape(n, self.block_size, -1)[:, 1:]
        target = targets.reshape(n, self.block_size, -1)[:, 1:].argmax(-1)
        unary_ids = full_unary.argmax(-1)
        candidate_logits, candidates = unary_topk(full_unary, self.config.prefix_top_k)
        reference, anchors = ids[:, 1:], ids[:, 0]
        match = candidates.eq(target[..., None])
        covered = match.any(-1)
        reference_covered = candidates.eq(reference[..., None]).any(-1)
        reachable = reachable_greedy_prefix(reference, target, reference_covered)
        focus = ramp(self.joint_step, s["focus_start_steps"], s["focus_warmup_steps"]) if self.training else 1.
        alpha = s["loss_alpha"] * ramp(self.joint_step, s["delay_steps"], s["head_warmup_steps"]) if self.training else s["loss_alpha"]
        weights = selector_weights(valid, covered, reachable, focus, s["dense_floor"])
        score_parts = []
        for start in range(0, n, s["anchor_chunk_size"]):
            end = start + s["anchor_chunk_size"]
            args = (hidden[start:end], candidates[start:end], candidate_logits[start:end], anchors[start:end], reference[start:end])
            # This is an execution option. Both paths retain the same anchors,
            # targets, loss normalization and gradient connections.
            score_parts.append(checkpoint(self._scores, *args, use_reentrant=False)
                               if (self.training and torch.is_grad_enabled()
                                   and s.get("selector_checkpointing", True)) else self._scores(*args))
        scores = torch.cat(score_parts)
        ce = -scores.float().log_softmax(-1).gather(-1, match.long().argmax(-1)[..., None]).squeeze(-1)
        # A fixed valid-position denominator keeps rare shortlist hits from
        # dominating early scratch training. Pool misses still receive dense CE.
        selector_loss = (ce * weights).sum() / valid.float().sum().clamp_min(1)
        loss = backbone_loss + alpha * selector_loss
        one = loss.detach().new_tensor(1.)
        for key, value in (("loss", loss), ("selector_ce", selector_loss),
                           ("selector_alpha", one * alpha), ("prefix_focus", one * focus)):
            metrics[key + "_sum"], metrics[key + "_total"] = value.detach(), one.clone()
        with torch.no_grad():
            teacher_choice = candidates.gather(-1, scores.argmax(-1)[..., None]).squeeze(-1)
            for key, values in (("candidate_target_recall", covered), ("candidate_reference_recall", reference_covered),
                                ("selector_supervised_fraction", covered), ("reachable_greedy_fraction", reachable),
                                ("teacher_argmax_acc", teacher_choice.eq(target)),
                                ("teacher_rescue", teacher_choice.eq(target) & ~unary_ids.eq(target)),
                                ("teacher_damage", ~teacher_choice.eq(target) & unary_ids.eq(target))):
                self._record(metrics, key, values, valid)
            features = dict(hidden=hidden, candidates=candidates, unary=candidate_logits,
                            anchors=anchors, reference=reference, target=target, valid=valid)
            if self.training and self.selector_guard is not None and self.joint_step >= s["guard_start_steps"]:
                self.selector_guard.replay.add(features)
            if not self.training:
                paths = [self.prefix_head.greedy(hidden[i:i+4], candidates[i:i+4], candidate_logits[i:i+4],
                         anchors[i:i+4], embedding_weight=self.embed_tokens.weight) for i in range(0, n, 4)]
                chosen = torch.cat(paths)
                correct = (chosen.eq(reference) & valid).long().cumprod(-1)
                unary_correct = (unary_ids.eq(reference) & valid).long().cumprod(-1)
                oracle = (reference_covered & valid).long().cumprod(-1)
                target_correct = target_prefix_mask(chosen, reference, target, valid)
                for key, values in (("rollout_reference_eal", 1 + correct.sum(-1)),
                                    ("unary_reference_eal", 1 + unary_correct.sum(-1)),
                                    ("oracle_reference_eal", 1 + oracle.sum(-1)),
                                    ("target_prefix_proxy", 1 + target_correct.sum(-1))):
                    self._record(metrics, key, values, valid[:, 0])
        self.joint_metrics = metrics if self.training else None
        return None, loss, metrics
