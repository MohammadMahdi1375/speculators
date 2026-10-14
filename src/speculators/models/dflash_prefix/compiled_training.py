"""Training-only pilot that replaces a trained nonlinear history head.

The draft backbone, vocabulary projection, and local transition stay frozen.
The new compiled history learns from true-context CE, matched-prefix teacher
KL, and centered teacher/student score changes after an older-token edit.
Edited prefixes never receive cached target labels. These are training proxies,
not an assurance that the new architecture retains serving acceptance length.

Only a complete local_prefix_v2 Speculators checkpoint and a local, compatible
target are accepted. The relocated target path is installed before loading any
verifier weights. Compatibility checks establish architecture/shape agreement;
they cannot establish identical target weights without original target hashes.
"""

import copy
import math
from pathlib import Path

import torch
from transformers import AutoConfig

from speculators.config import VerifierConfig
from speculators.model import SpeculatorModel
from speculators.utils.loading import list_checkpoint_keys

from .compiled_selector import CompiledPrefixSelector
from .config import DFlashPrefixSpeculatorConfig
from .core import DFlashPrefixDraftModel, reachable_greedy_prefix
from .selector_v2 import unary_topk


_TARGET_OWNED = {
    "embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight",
    "verifier_norm.weight", "t2d", "d2t",
}


def _settings(settings):
    settings = dict(settings or {})
    out = {
        "cap": float(settings.get("cap", 8.0)),
        "kd_weight": float(settings.get("kd_weight", 0.5)),
        "history_weight": float(settings.get("history_weight", 0.1)),
        "intervention_fraction": float(settings.get("intervention_fraction", 0.25)),
        "anchor_chunk_size": int(settings.get("anchor_chunk_size", 64)),
    }
    if not all(math.isfinite(value) for value in out.values()):
        raise ValueError("Pilot settings must be finite")
    if not 0 < out["cap"] <= 8 or not 0 <= out["intervention_fraction"] <= 1:
        raise ValueError("cap must be in (0,8]; intervention_fraction in [0,1]")
    if min(out["kd_weight"], out["history_weight"]) < 0:
        raise ValueError("Distillation coefficients must be nonnegative")
    if not 1 <= out["anchor_chunk_size"] <= 64:
        raise ValueError("anchor_chunk_size must be from 1 through 64")
    return out


def older_history_intervention(reference, candidates, valid, fraction, *,
                               training=True, row_offset=0):
    """Swap one natural candidate at j and supervise only query i >= j+2.

Indices refer to draft slots: anchor is excluded and is never changed. The
immediate predecessor at every supervised query is therefore unchanged. The
returned mask describes affected queries; it is never a CE-label validity mask.
Validation uses deterministic hashes rather than consuming training RNG state.
"""
    n, p = reference.shape
    edited = reference.clone()
    mask = torch.zeros_like(valid, dtype=torch.bool)
    if p < 3 or fraction == 0:
        return edited, mask
    different = candidates[:, :p-2].ne(reference[:, :p-2, None])
    eligible = valid[:, :p-2] & valid[:, 2:] & different.any(-1)
    rows = torch.arange(n, device=reference.device) + row_offset
    if training:
        source_priority = torch.rand((n, p-2), device=reference.device)
        selected = torch.rand(n, device=reference.device) < fraction
    else:
        columns = torch.arange(p-2, device=reference.device)
        source_priority = ((rows[:, None] * 1664525 + columns * 1013904223)
                           % 10007).float() / 10007
        selected = ((rows * 1664525 + 1013904223) % 10007).float() < fraction * 10007
    j = source_priority.masked_fill(~eligible, -1).argmax(-1)
    selected = selected & eligible.any(-1)
    alternatives = different.long().argmax(-1)
    alternate_ids = candidates[:, :p-2].gather(-1, alternatives[..., None]).squeeze(-1)
    replacement = alternate_ids.gather(-1, j[:, None]).squeeze(-1)
    original = reference.gather(-1, j[:, None]).squeeze(-1)
    edited.scatter_(1, j[:, None], torch.where(selected, replacement, original)[:, None])
    positions = torch.arange(p, device=reference.device)
    mask = selected[:, None] & valid & (positions[None] >= j[:, None] + 2)
    return edited, mask


def pilot_loss_terms(student, teacher, candidates, target_ids, valid,
                     correct_context, *, student_edited=None, teacher_edited=None,
                     older_intervention_mask=None):
    """Return masked sums/counts; teacher tensors are detached defensively.

Teacher/student share the same natural shortlist and unedited reference prefix.
Only CE uses cached target IDs, and only when their context is both correct and
representable. Centered score-difference matching is invariant to per-row logit
shifts. No loss target is manufactured for the edited context.
"""
    scores = student.float()
    teacher = teacher.detach().float()
    matches = candidates.eq(target_ids[..., None])
    ce_mask = valid & correct_context & matches.any(-1)
    labels = matches.long().argmax(-1)
    ce = -scores.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    log_teacher = teacher.log_softmax(-1)
    kl = (log_teacher.exp() * (log_teacher - scores.log_softmax(-1))).sum(-1)
    # Connected zero gives finite zero gradients on batches with no valid labels.
    zero = scores.sum() * 0.0
    out = {
        "target_ce": ((ce * ce_mask).sum(), ce_mask.float().sum()),
        "matched_prefix_kl": ((kl * valid).sum(), valid.float().sum()),
        "history_delta_loss": (zero, zero.detach()),
    }
    if student_edited is not None:
        if teacher_edited is None or older_intervention_mask is None:
            raise ValueError("Edited scores require teacher scores and an older-edit mask")
        student_delta = student_edited.float() - scores
        teacher_delta = teacher_edited.detach().float() - teacher
        student_delta = student_delta - student_delta.mean(-1, keepdim=True)
        teacher_delta = teacher_delta - teacher_delta.mean(-1, keepdim=True)
        error = (student_delta - teacher_delta).square().mean(-1)
        mask = older_intervention_mask & valid
        out["history_delta_loss"] = ((error * mask).sum(), mask.float().sum())
    return out


@SpeculatorModel.register("dflash_prefix_compiled_pilot")
class CompiledPrefixPilot(DFlashPrefixDraftModel):
    """Training-only subclass; serialized checkpoints use the serving class.

Registering a separate class satisfies Speculators' exact-type trainer check.
The ordinary dflash_prefix registry entry and serving state-dict names remain
unchanged. FSDP and resuming this temporary teacher-bearing model are unsupported.
"""

    def configure_pilot(self, settings):
        if self.config.prefix_selector_kind != "local_prefix_v2":
            raise ValueError("Pilot initialization requires a local_prefix_v2 checkpoint")
        self.pilot_settings = _settings(settings)
        self.teacher_head = copy.deepcopy(self.prefix_head).eval().requires_grad_(False)
        self.requires_grad_(False)
        self.prefix_head = CompiledPrefixSelector(
            self.config.transformer_layer_config.vocab_size,
            self.config.transformer_layer_config.hidden_size,
            rank=self.config.prefix_rank, top_k=self.config.prefix_top_k,
            max_proposals=self.block_size - 1, heads=self.config.prefix_attention_heads,
            correction_scale=1.0, history_scale=self.config.prefix_history_scale,
            compiled_logit_cap=self.pilot_settings["cap"],
        ).initialize_from_teacher(self.teacher_head)
        # Keep frozen local calculations in the teacher's parameter dtype. New
        # history master weights remain FP32; trainer autocast handles matmuls.
        for name in ("predecessor_projection", "successor_projection", "hidden_projection"):
            getattr(self.prefix_head, name).to(dtype=getattr(self.teacher_head, name).weight.dtype)
        self.prefix_head.set_history_trainable(True)
        self.config.prefix_selector_kind = "compiled_prefix_v3"
        self.config.prefix_compiled_logit_cap = self.pilot_settings["cap"]
        self.config.prefix_attention_layers = 1
        self.config.prefix_freeze_backbone = True
        self.config.prefix_detach_backbone = True
        self.config.prefix_loss_kind = "target_ce"
        self.config.prefix_loss_alpha = 1.0
        self.config.prefix_loss_scope = "reachable_greedy_prefix"
        self.config.prefix_walk_backend = "torch"
        self.config.prefix_validation_gate_scales = []
        self.config.architectures = ["DFlashPrefixDraftModel"]
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "teacher_head"):
            self.teacher_head.eval()
        return self

    def save_pretrained(self, save_directory, *, state_dict=None, **kwargs):
        if kwargs.get("push_to_hub"):
            raise ValueError("Save the pilot checkpoint locally before publishing it")
        if not hasattr(self, "teacher_head"):
            return super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)
        source = self.state_dict() if state_dict is None else state_dict
        serving_state = {name: value for name, value in source.items()
                         if not name.startswith("teacher_head.")}
        expected = {name for name in self.state_dict() if not name.startswith("teacher_head.")}
        if set(serving_state) != expected:
            raise ValueError("Pilot save needs a complete, unwrapped model state_dict")
        result = super().save_pretrained(save_directory, state_dict=serving_state, **kwargs)
        # HF writes type(self).__name__; the training subclass is not a serving
        # architecture. Re-save the same config with the standard exported name.
        self.config.architectures = ["DFlashPrefixDraftModel"]
        self.config.save_pretrained(save_directory)
        return result

    @staticmethod
    def _add_metric(metrics, name, numerator, denominator):
        metrics[name + "_sum"] = metrics.get(name + "_sum", 0) + numerator.detach()
        metrics[name + "_total"] = metrics.get(name + "_total", 0) + denominator.detach()

    def forward(self, hidden_states, input_ids, loss_mask,
                verifier_last_hidden_states, document_ids, position_ids=None,
                loss_config=None, gamma=4.0, max_anchors=128,
                per_position_loss_weight="fixed-exp-decay", dpace_alpha=0.5, **kwargs):
        if not hasattr(self, "teacher_head"):
            raise RuntimeError("Call configure_pilot before training")
        with torch.no_grad():
            hidden, unary, targets, aligned, indices = self._backbone_forward(
                hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                document_ids, position_ids, max_anchors=max_anchors, **kwargs,
            )
            n, p = indices.numel() // self.block_size, self.block_size - 1
            ids = input_ids[0, indices].reshape(n, self.block_size)
            docs = document_ids[0, indices].reshape(n, self.block_size)
            valid = aligned.reshape(n, self.block_size)[:, 1:].bool()
            valid = (valid & docs[:, 1:].eq(docs[:, :1])).long().cumprod(-1).bool()
            hidden = hidden.reshape(n, self.block_size, -1)[:, 1:].detach()
            unary = unary.reshape(n, self.block_size, -1)[:, 1:]
            targets = targets.reshape(n, self.block_size, -1)[:, 1:]
            target_ids = targets.argmax(-1)
            candidate_logits, candidates = unary_topk(unary, self.config.prefix_top_k)
            anchor, reference = ids[:, 0], ids[:, 1:]
            covered = candidates.eq(reference[..., None]).any(-1)
            correct_context = reachable_greedy_prefix(reference, target_ids, covered)
        # Large vocabulary tensors are no longer needed while retaining history
        # activations; target/backbone computation itself remains unchanged.
        del unary, targets
        sums, counts, metrics = {}, {}, {}
        chunk_size = self.pilot_settings["anchor_chunk_size"]
        embedding = {"embedding_weight": self.embed_tokens.weight}
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            sl = slice(start, stop)
            arguments = (hidden[sl], candidates[sl], candidate_logits[sl], anchor[sl])
            with torch.no_grad():
                teacher = self.teacher_head.score_teacher_prefix(*arguments, reference[sl], **embedding)
            student = self.prefix_head.score_teacher_prefix(*arguments, reference[sl], **embedding)
            edited_student = edited_teacher = edit_mask = None
            if self.pilot_settings["history_weight"] > 0 and self.pilot_settings["intervention_fraction"] > 0 and p >= 3:
                edited, edit_mask = older_history_intervention(
                    reference[sl], candidates[sl], valid[sl],
                    self.pilot_settings["intervention_fraction"],
                    training=self.training, row_offset=start,
                )
                with torch.no_grad():
                    edited_teacher = self.teacher_head.score_teacher_prefix(*arguments, edited, **embedding)
                edited_student = self.prefix_head.score_teacher_prefix(*arguments, edited, **embedding)
            terms = pilot_loss_terms(
                student, teacher, candidates[sl], target_ids[sl], valid[sl], correct_context[sl],
                student_edited=edited_student, teacher_edited=edited_teacher,
                older_intervention_mask=edit_mask,
            )
            for name, (total, count) in terms.items():
                sums[name] = sums.get(name, 0) + total
                counts[name] = counts.get(name, 0) + count
            with torch.no_grad():
                direct = candidates[sl].gather(-1, student.argmax(-1, keepdim=True)).squeeze(-1)
                direct_prefix = (direct.eq(reference[sl]) & valid[sl]).long().cumprod(-1)
                block_valid = valid[sl, 0].float()
                self._add_metric(metrics, "direct_prefix_reference_proxy",
                                 ((1 + direct_prefix.sum(-1)) * block_valid).sum(), block_valid.sum())
                self._add_metric(metrics, "candidate_reference_recall",
                                 (covered[sl] & valid[sl]).float().sum(), valid[sl].float().sum())
                if not self.training:
                    # The compiled walk allocates candidate-pair tables. Smaller
                    # subchunks keep validation memory below teacher-score chunks.
                    for inner_start in range(start, stop, 8):
                        inner = slice(inner_start, min(stop, inner_start + 8))
                        inner_args = (hidden[inner], candidates[inner], candidate_logits[inner], anchor[inner])
                        for label, head in (("student", self.prefix_head), ("teacher", self.teacher_head)):
                            chosen = head.greedy(*inner_args, **embedding)
                            prefix = (chosen.eq(reference[inner]) & valid[inner]).long().cumprod(-1)
                            alive = valid[inner, 0].float()
                            self._add_metric(metrics, label + "_reference_eal",
                                             ((1 + prefix.sum(-1)) * alive).sum(), alive.sum())
        values = {name: sums[name] / counts[name].clamp_min(1) for name in sums}
        loss = (values["target_ce"] + self.pilot_settings["kd_weight"] * values["matched_prefix_kl"]
                + self.pilot_settings["history_weight"] * values["history_delta_loss"])
        for name in sums:
            self._add_metric(metrics, name, sums[name], counts[name])
        self._add_metric(metrics, "loss", loss, loss.new_ones(()))
        return None, loss, metrics


def build_compiled_pilot(checkpoint, target_path, settings=None):
    """Load the trained draft locally, relocate its target, then replace history.

The verifier path embedded in the old checkpoint is never dereferenced. Full
draft state coverage is checked after HF's shape-checked load, preventing a
missing history or backbone weight from silently becoming random initialization.
"""
    checkpoint, target_path = Path(checkpoint).resolve(), Path(target_path).resolve()
    settings = dict(settings or {})
    attention_impl = settings.get("draft_attn_impl", "sdpa")
    if attention_impl not in {"sdpa", "eager"}:
        raise ValueError("The compiled pilot requires draft_attn_impl=sdpa or eager")
    if not checkpoint.is_dir() or not target_path.is_dir():
        raise ValueError("Pilot checkpoint and target must be existing local directories")
    config = DFlashPrefixSpeculatorConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if config.prefix_selector_kind != "local_prefix_v2":
        raise ValueError("Pilot requires a complete local_prefix_v2 teacher checkpoint")
    if config.prefix_history_scale <= 0 or config.prefix_inference_gate_scale != 1:
        raise ValueError("Pilot expects the teacher's active history and unscaled serving corrections")
    target = AutoConfig.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=False)
    target = getattr(target, "text_config", target)
    layer = config.transformer_layer_config
    # This private runtime field is not reliably serialized by HF. Never let a
    # moved Ascend checkpoint silently fall back to simple_flex_attention.
    layer._attn_implementation = attention_impl
    for key in ("hidden_size", "vocab_size", "rms_norm_eps"):
        if getattr(target, key, None) != getattr(layer, key, None):
            raise ValueError(f"Target {key} differs from the trained draft")
    verifier = VerifierConfig.from_pretrained(str(target_path), local_files_only=True)
    if verifier.architectures != config.speculators_config.verifier.architectures:
        raise ValueError("Target architecture differs from the checkpoint verifier")
    if any(index < 0 or index > target.num_hidden_layers for index in config.aux_hidden_state_layer_ids):
        raise ValueError("Target has too few layers for the stored hidden-state indices")
    config.speculators_config.verifier = verifier
    checkpoint_keys = set(list_checkpoint_keys(checkpoint))
    target_keys = set(list_checkpoint_keys(target_path))
    if not any(name.endswith("model.norm.weight") for name in target_keys):
        raise ValueError("Target checkpoint lacks final norm weights")
    if not getattr(target, "tie_word_embeddings", False) and "lm_head.weight" not in target_keys:
        raise ValueError("Untied target checkpoint lacks its lm_head weights")
    model = CompiledPrefixPilot.from_pretrained(str(checkpoint), config=config,
                                               local_files_only=True, dtype="auto")
    expected = set(model.state_dict())
    missing = expected - checkpoint_keys - _TARGET_OWNED
    unexpected = checkpoint_keys - expected
    if missing or unexpected:
        raise ValueError(f"Incomplete/incompatible draft state: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}")
    model.configure_pilot(settings)
    return model
