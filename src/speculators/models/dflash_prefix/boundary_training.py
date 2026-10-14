"""Opt-in boundary-distillation pilot exporting an unchanged serving model."""
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from speculators.config import VerifierConfig
from speculators.model import SpeculatorModel
from speculators.utils.loading import list_checkpoint_keys
from .boundary_objective import (boundary_global_loss, boundary_loss_terms,
                                 boundary_settings, dense_boundary_anchors, refresh_rows)
from .boundary_support import support_transfer_terms
from .compiled_training import _TARGET_OWNED
from .config import DFlashPrefixSpeculatorConfig
from .core import DFlashPrefixDraftModel, reachable_greedy_prefix
from .selector_v2 import unary_topk


@SpeculatorModel.register("dflash_prefix_boundary_training")
class BoundaryPrefixTraining(DFlashPrefixDraftModel):
    def configure_boundary(self, settings):
        self.boundary_config = boundary_settings(settings)
        last = self.boundary_config["train_last_layers"]
        if last != "all" and len(self.layers) < last:
            raise ValueError("Checkpoint has fewer layers than TRAIN_LAST_LAYERS")
        self.requires_grad_(False)
        # Full frozen copy deliberately isolates the baseline from all training
        # parameters and buffers. This costs training memory, never serving time.
        self.teacher_model = copy.deepcopy(self).eval().requires_grad_(False)
        modules = [self.prefix_head, self.norm]
        if last == "all":
            modules += [self.fc, self.hidden_norm, *self.layers]
        else:
            modules += list(self.layers[-last:])
        for module in modules:
            module.requires_grad_(True)
            module.float()
        self.config.prefix_freeze_backbone = False
        self.config.prefix_detach_backbone = False
        self.config.architectures = ["DFlashPrefixDraftModel"]
        self.config.prefix_training_experiment = dict(
            name="verification_boundary_distillation_v2", **self.boundary_config,
            inference_architecture_changed=False,
        )
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "teacher_model"):
            self.teacher_model.eval()
        return self

    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        anchors, valid = self._boundary_anchors
        if anchors.numel() != max_anchors:
            raise ValueError("Boundary anchor count mismatch")
        full = sliding = None
        if self.uses_full_attn:
            full = self._create_attention_mask(document_ids, loss_mask.shape[1], anchors, device, sliding_window=None)
        if self.uses_sliding_window_attn:
            sliding = self._create_attention_mask(
                document_ids, loss_mask.shape[1], anchors, device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal)
        return full, sliding, anchors, valid

    @staticmethod
    def _metric(metrics, key, total, count):
        metrics[key + "_sum"] = metrics.get(key + "_sum", 0) + total.detach()
        metrics[key + "_total"] = metrics.get(key + "_total", 0) + count.detach()

    def save_pretrained(self, save_directory, *, state_dict=None, **kwargs):
        if kwargs.get("push_to_hub"):
            raise ValueError("Save locally before publishing")
        state = self.state_dict() if state_dict is None else state_dict
        state = {k: v for k, v in state.items() if not k.startswith("teacher_model.")}
        expected = {k for k in self.state_dict() if not k.startswith("teacher_model.")}
        if set(state) != expected:
            raise ValueError("Export requires a complete unwrapped model state")
        result = super().save_pretrained(save_directory, state_dict=state, **kwargs)
        self.config.architectures = ["DFlashPrefixDraftModel"]
        self.config.save_pretrained(save_directory)
        return result

    def forward(self, hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                document_ids, position_ids=None, max_anchors=32, **kwargs):
        if not hasattr(self, "teacher_model"):
            raise RuntimeError("Call configure_boundary before training")
        # Trainer's base-loss options must not be forwarded into attention.
        for key in ("loss_config", "gamma", "per_position_loss_weight", "dpace_alpha"):
            kwargs.pop(key, None)
        p, active = self.block_size - 1, max_anchors
        total = active + p - 1
        chosen = dense_boundary_anchors(loss_mask, total, self.block_size, training=self.training)
        self._boundary_anchors = self.teacher_model._boundary_anchors = chosen
        args = (hidden_states, input_ids, loss_mask, verifier_last_hidden_states, document_ids, position_ids)
        with torch.no_grad():
            th, tu, target, aligned, indices = self.teacher_model._backbone_forward(*args, max_anchors=total, **kwargs)
        h, u, _, _, _ = self._backbone_forward(*args, max_anchors=total, **kwargs)
        block = self.block_size
        h, th = h.reshape(total, block, -1)[:, 1:], th.reshape(total, block, -1)[:, 1:]
        u, tu = u.reshape(total, block, -1)[:, 1:], tu.reshape(total, block, -1)[:, 1:]
        target_ids = target.reshape(total, block, -1)[:, 1:].argmax(-1)
        del target
        ids = input_ids[0, indices].reshape(total, block)
        docs = document_ids[0, indices].reshape(total, block)
        valid = aligned.reshape(total, block)[:, 1:].bool() & docs[:, 1:].eq(docs[:, :1])
        valid = valid.long().cumprod(-1).bool()
        reference, anchor = ids[:, 1:], ids[:, 0]
        candidate_logits, candidates = unary_topk(u[:active], self.config.prefix_top_k)
        with torch.no_grad():
            old_logits, old_candidates = unary_topk(tu, self.config.prefix_top_k)
            covered = candidates.eq(reference[:active, :, None]).any(-1)
            context = reachable_greedy_prefix(reference[:active], target_ids[:active], covered)
            usable = valid[:active] & context
            # Unary support can learn omitted candidates. It has no chosen
            # candidate-prefix input, so do not require shortlist coverage here.
            greedy_context = reachable_greedy_prefix(reference[:active], target_ids[:active],
                                                      torch.ones_like(covered))
            target_covered = candidates.eq(target_ids[:active, :, None]).any(-1)
            rows = refresh_rows(active, p, device=ids.device)
            # Same absolute token AND same document, with the exact preceding
            # token as refreshed anchor. No position after the query is read.
            refresh_valid = (valid[rows, 0] & valid[:active]
                             & indices.reshape(total, block)[rows, 1].eq(indices.reshape(total, block)[:active, 1:])
                             & docs[rows, 0].eq(docs[:active, :1])
                             & target_ids[rows, 0].eq(target_ids[:active]))
            forced = candidates.eq(reference[:active, :, None]).long().argmax(-1)
        full_ce = F.cross_entropy(u[:active].float().reshape(-1, u.shape[-1]),
                                  target_ids[:active].reshape(-1), reduction="none").reshape(active, p)
        terms = {"full_ce": ((full_ce * valid[:active]).sum(), valid[:active].float().sum())}
        metrics = {}
        emb = {"embedding_weight": self.embed_tokens.weight}
        teacher_emb = {"embedding_weight": self.teacher_model.embed_tokens.weight}
        chunk = self.boundary_config["anchor_chunk_size"]
        for start in range(0, active, chunk):
            end = min(active, start + chunk)
            sl = slice(start, end)
            support, support_stats = support_transfer_terms(
                u[sl], tu[sl], tu[rows[sl], 0], target_ids[sl],
                refresh_valid[sl] & greedy_context[sl], target_covered[sl],
                minimum_advantage=self.boundary_config["teacher_ce_improvement"])
            for key, (num, den) in support.items():
                a, b = terms.get(key, (0, 0))
                terms[key] = (a + num, b + den)
            for key, pair in support_stats.items():
                self._metric(metrics, key, *pair)
            c, logits = candidates[sl], candidate_logits[sl]
            tables = self.prefix_head.prepare_tables(h[sl], c, logits, anchor[sl], **emb)
            _, scores = self.prefix_head.walk(tables, forced_indices=forced[sl], return_scores=True)
            with torch.no_grad():
                # Baseline on the student's current candidate support for KL;
                # baseline EAL below uses its own natural pool, never this pool.
                old_on_pool = tu[sl].gather(-1, c)
                baseline_tables = self.teacher_model.prefix_head.prepare_tables(th[sl], c, old_on_pool, anchor[sl], **teacher_emb)
                _, old_scores = self.teacher_model.prefix_head.walk(baseline_tables, forced_indices=forced[sl], return_scores=True)
                rr = rows[sl]
                n = end - start
                cf = c.reshape(n * p, 1, -1)
                rh = th[rr, 0].reshape(n * p, 1, -1)
                ru = tu[rr, 0].gather(-1, c).reshape(n * p, 1, -1)
                ra = anchor[rr].reshape(-1)
                fresh_tables = self.teacher_model.prefix_head.prepare_tables(rh, cf, ru, ra, **teacher_emb)
                _, fresh_scores = self.teacher_model.prefix_head.walk(fresh_tables, return_scores=True)
                fresh_scores = fresh_scores.reshape(n, p, -1)
                natural_tables = self.teacher_model.prefix_head.prepare_tables(
                    rh, old_candidates[rr, 0].reshape(n*p, 1, -1),
                    old_logits[rr, 0].reshape(n*p, 1, -1), ra, **teacher_emb)
                natural_tokens = self.teacher_model.prefix_head.walk(natural_tables).reshape(n, p)
                natural_correct = natural_tokens.eq(target_ids[sl])
            local, stats = boundary_loss_terms(scores, old_scores, fresh_scores, c,
                target_ids[sl], usable[sl], refresh_valid[sl], natural_correct, self.boundary_config)
            for key, (num, den) in local.items():
                a, b = terms.get(key, (0, 0))
                terms[key] = (a + num, b + den)
            for key, pair in stats.items():
                self._metric(metrics, key, *pair)
            if not self.training:
                with torch.no_grad():
                    actual = self.prefix_head.walk(tables)
                    native = self.teacher_model.prefix_head.prepare_tables(
                        th[sl], old_candidates[sl], old_logits[sl], anchor[sl], **teacher_emb)
                    baseline = self.teacher_model.prefix_head.walk(native)
                    lengths = []
                    alive = valid[sl, 0].float()
                    for name, tokens in (("student", actual), ("baseline", baseline)):
                        length = (tokens.eq(reference[sl]) & valid[sl]).long().cumprod(-1).sum(-1).float() + 1
                        lengths.append(length)
                        self._metric(metrics, name + "_reference_eal", (length * alive).sum(), alive.sum())
                    delta = lengths[0] - lengths[1]
                    self._metric(metrics, "paired_reference_gain", (delta * alive).sum(), alive.sum())
                    self._metric(metrics, "reference_tokens_gained", (delta.clamp_min(0) * alive).sum(), alive.sum())
                    self._metric(metrics, "reference_tokens_lost", ((-delta).clamp_min(0) * alive).sum(), alive.sum())
                    oracle = (c.eq(reference[sl, :, None]).any(-1) & valid[sl]).long().cumprod(-1).sum(-1).float() + 1
                    unary_length = (u[sl].argmax(-1).eq(reference[sl]) & valid[sl]).long().cumprod(-1).sum(-1).float() + 1
                    self._metric(metrics, "oracle_reference_eal", (oracle * alive).sum(), alive.sum())
                    self._metric(metrics, "unary_reference_eal", (unary_length * alive).sum(), alive.sum())
        loss = boundary_global_loss(terms, self.boundary_config)
        for key, pair in terms.items():
            self._metric(metrics, key, *pair)
        self._metric(metrics, "candidate_target_recall", (candidates.eq(target_ids[:active, :, None]).any(-1) & valid[:active]).float().sum(), valid[:active].float().sum())
        self._metric(metrics, "loss", loss, loss.new_ones(()))
        return None, loss, metrics


def build_boundary_model(checkpoint, target_path, settings=None):
    checkpoint, target_path = Path(checkpoint).resolve(), Path(target_path).resolve()
    settings = dict(settings or {})
    config = DFlashPrefixSpeculatorConfig.from_pretrained(checkpoint, local_files_only=True)
    if (config.prefix_selector_kind != "compiled_prefix_v3" or config.sample_from_anchor
            or config.prefix_history_scale != 1 or config.prefix_inference_gate_scale != 1
            or getattr(config, "prefix_disable_selector", False)):
        raise ValueError("Use a complete enabled compiled_prefix_v3 checkpoint with scales=1")
    target = AutoConfig.from_pretrained(target_path, local_files_only=True, trust_remote_code=False)
    target = getattr(target, "text_config", target)
    layer = config.transformer_layer_config
    implementation = settings.get("draft_attn_impl", "sdpa")
    if implementation not in ("sdpa", "eager"):
        raise ValueError("Use sdpa or eager on A2")
    layer._attn_implementation = implementation
    for key in ("hidden_size", "vocab_size", "rms_norm_eps"):
        if getattr(target, key, None) != getattr(layer, key, None):
            raise ValueError(f"Target {key} differs from checkpoint")
    if config.draft_vocab_size != target.vocab_size:
        raise ValueError("Full target vocabulary required")
    verifier = VerifierConfig.from_pretrained(str(target_path), local_files_only=True)
    if verifier.architectures != config.speculators_config.verifier.architectures:
        raise ValueError("Target architecture differs")
    if any(i < 0 or i > target.num_hidden_layers for i in config.aux_hidden_state_layer_ids):
        raise ValueError("Invalid target layer IDs")
    target_keys = set(list_checkpoint_keys(target_path))
    if "model.norm.weight" not in target_keys or (not target.tie_word_embeddings and "lm_head.weight" not in target_keys):
        raise ValueError("Incomplete target checkpoint")
    config.speculators_config.verifier = verifier
    keys = set(list_checkpoint_keys(checkpoint))
    model = BoundaryPrefixTraining.from_pretrained(checkpoint, config=config, local_files_only=True, dtype="auto")
    expected = set(model.state_dict())
    if expected - keys - _TARGET_OWNED or keys - expected:
        raise ValueError("Incomplete/incompatible draft checkpoint state")
    return model.configure_boundary(settings)
