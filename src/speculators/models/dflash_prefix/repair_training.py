"""Opt-in continuation of a complete compiled-prefix checkpoint.

Train the existing local AND history parameters. Keep backbone, vocabulary,
shortlist, serving head class, block length, and verification unchanged.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from speculators.config import VerifierConfig
from speculators.model import SpeculatorModel
from speculators.utils.loading import list_checkpoint_keys

from .compiled_training import CompiledPrefixPilot, _TARGET_OWNED
from .config import DFlashPrefixSpeculatorConfig
from .core import reachable_greedy_prefix
from .repair_objective import candidate_margin, global_masked_loss, prefix_credit, repair_loss_terms, repair_settings
from .renewal_credit import dense_anchors, round_saving_credit
from .selector_v2 import unary_topk


@SpeculatorModel.register("dflash_prefix_repair_training")
class RepairPrefixTraining(CompiledPrefixPilot):
    """Training wrapper only; exports the existing DFlashPrefixDraftModel."""

    def configure_repair(self, settings):
        self.repair_config = repair_settings(settings)
        self.requires_grad_(False)
        self.teacher_head = copy.deepcopy(self.prefix_head).eval().requires_grad_(False)
        self.prefix_head.requires_grad_(True)
        # Loading BF16 to FP32 is exact; autocast still uses the existing BF16
        # projections. No new head, random reinitialization or parameter shapes.
        self.prefix_head.float()
        self.config.prefix_freeze_backbone = True
        self.config.prefix_detach_backbone = True
        self.config.architectures = ["DFlashPrefixDraftModel"]
        self.config.prefix_training_experiment = dict(
            name="round_saving_v1", anchor_sampler="contiguous", **self.repair_config,
            inference_architecture_changed=False,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        # Training subclass only; exported serving model has its original code.
        anchors, anchor_valid = dense_anchors(loss_mask, max_anchors, self.block_size)
        full = sliding = None
        if self.uses_full_attn:
            full = self._create_attention_mask(document_ids, loss_mask.shape[1], anchors,
                                               device, sliding_window=None)
        if self.uses_sliding_window_attn:
            sliding = self._create_attention_mask(
                document_ids, loss_mask.shape[1], anchors, device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal)
        return full, sliding, anchors, anchor_valid

    def forward(self, hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                document_ids, position_ids=None, loss_config=None, gamma=4.,
                max_anchors=64, per_position_loss_weight="dpace", dpace_alpha=.5, **kwargs):
        with torch.no_grad():
            hidden, unary, targets, aligned, indices = self._backbone_forward(
                hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                document_ids, position_ids, max_anchors=max_anchors, **kwargs)
            block = self.block_size
            n, p = indices.numel() // block, block - 1
            ids = input_ids[0, indices].reshape(n, block)
            docs = document_ids[0, indices].reshape(n, block)
            valid = aligned.reshape(n, block)[:, 1:].bool()
            valid = (valid & docs[:, 1:].eq(docs[:, :1])).long().cumprod(-1).bool()
            hidden = hidden.reshape(n, block, -1)[:, 1:].detach()
            unary = unary.reshape(n, block, -1)[:, 1:]
            targets = targets.reshape(n, block, -1)[:, 1:]
            target_ids = targets.argmax(-1)
            candidate_logits, candidates = unary_topk(unary, self.config.prefix_top_k)
            anchor, reference = ids[:, 0], ids[:, 1:]
            reference_matches = candidates.eq(reference[..., None])
            correct_context = reachable_greedy_prefix(reference, target_ids, reference_matches.any(-1))
            forced_indices = reference_matches.long().argmax(-1)
        del unary, targets
        if n == 0:
            raise ValueError("No draft anchors in this batch; inspect max_anchors/data masks")
        metrics, accumulated = {}, {}
        first_error_losses, accepted_counts, span_counts = [], [], []
        chunk = self.repair_config["anchor_chunk_size"]
        for start in range(0, n, chunk):
            sl = slice(start, min(n, start + chunk))
            args = (hidden[sl], candidates[sl], candidate_logits[sl], anchor[sl])
            embedding = {"embedding_weight": self.embed_tokens.weight}
            with torch.no_grad():
                baseline_tables = self.teacher_head.prepare_tables(*args, **embedding)
                _, baseline = self.teacher_head.walk(
                    baseline_tables, forced_indices=forced_indices[sl], return_scores=True)
            # Use the same all-candidate tables and sequential walk as serving,
            # rather than a different GEMM shape for teacher-forced scoring.
            tables = self.prefix_head.prepare_tables(*args, **embedding)
            _, scores = self.prefix_head.walk(tables, forced_indices=forced_indices[sl], return_scores=True)
            terms, stats = repair_loss_terms(scores, baseline, candidates[sl], target_ids[sl],
                                            valid[sl], correct_context[sl], self.repair_config)
            matches = candidates[sl].eq(target_ids[sl, :, None])
            labels = matches.long().argmax(-1)
            usable = valid[sl] & correct_context[sl] & matches.any(-1)
            frontier, spans, accepted, _ = prefix_credit(scores.detach().argmax(-1).eq(labels), usable)
            first_error_losses.append((F.softplus(self.repair_config["margin"] -
                                                 candidate_margin(scores.float(), labels)) * frontier).sum(-1))
            accepted_counts.append(accepted.sum(-1))
            span_counts.append(spans.sum(-1))
            for name, (total, count) in terms.items():
                old_total, old_count = accumulated.get(name, (0, 0))
                accumulated[name] = (old_total + total, old_count + count)
                self._add_metric(metrics, name, total, count)
            for name, (total, count) in stats.items():
                self._add_metric(metrics, name, total, count)
            if not self.training:
                with torch.no_grad():
                    for label, head, prepared in (("student", self.prefix_head, tables),
                                                  ("baseline", self.teacher_head, baseline_tables)):
                        chosen = head.walk(prepared)
                        accepted = (chosen.eq(reference[sl]) & valid[sl]).long().cumprod(-1)
                        alive = valid[sl, 0].float()
                        self._add_metric(metrics, label + "_reference_eal",
                                         ((1 + accepted.sum(-1)) * alive).sum(), alive.sum())
        # Dense anchors i and i+1 differ by exactly one absolute token. The
        # first-column target at i labels that next token, including the bonus
        # when a jump lands there. End segments at unknown/non-greedy contexts.
        known_next = valid[:, 0] & reference[:, 0].eq(target_ids[:, 0])
        advantage, eligible, _ = round_saving_credit(
            torch.cat(accepted_counts), torch.cat(span_counts), known_next)
        rank = torch.cat(first_error_losses)
        positive = advantage.clamp(min=0, max=self.repair_config["span_cap"])
        accumulated["round_saving_rank"] = ((rank * positive).sum(), eligible.float().sum())
        self._add_metric(metrics, "round_saving_rank", *accumulated["round_saving_rank"])
        self._add_metric(metrics, "repair_saved_rounds", advantage.sum(), eligible.float().sum())
        self._add_metric(metrics, "positive_round_credit_fraction", (advantage > 0).float().sum(), eligible.float().sum())
        self._add_metric(metrics, "coalesced_repair_fraction", eligible.float().sum(), torch.cat(span_counts).gt(0).float().sum())
        self._add_metric(metrics, "known_round_segment_fraction", known_next.float().sum(), rank.new_tensor(n))
        loss = global_masked_loss(accumulated, self.repair_config)
        self._add_metric(metrics, "loss", loss, loss.new_ones(()))
        return None, loss, metrics


def build_repair_model(checkpoint, target_path, settings=None):
    """Relocate the target before loading; refuse incomplete/incompatible state."""
    checkpoint, target_path = Path(checkpoint).resolve(), Path(target_path).resolve()
    settings = dict(settings or {})
    attention_impl = settings.get("draft_attn_impl", "sdpa")
    if attention_impl not in {"sdpa", "eager"}:
        raise ValueError("Use draft_attn_impl=sdpa or eager")
    if not checkpoint.is_dir() or not target_path.is_dir():
        raise ValueError("Checkpoint and target must be existing local directories")
    config = DFlashPrefixSpeculatorConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if config.prefix_selector_kind != "compiled_prefix_v3":
        raise ValueError("INIT_CHECKPOINT must be a complete, trained compiled_prefix_v3 checkpoint")
    if (config.prefix_history_scale != 1 or config.prefix_inference_gate_scale != 1
            or getattr(config, "prefix_disable_selector", False) or config.sample_from_anchor):
        raise ValueError("Use enabled, unscaled compiled prefix with sample_from_anchor=False")
    target = AutoConfig.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=False)
    target = getattr(target, "text_config", target)
    layer = config.transformer_layer_config
    layer._attn_implementation = attention_impl
    for key in ("hidden_size", "vocab_size", "rms_norm_eps"):
        if getattr(target, key, None) != getattr(layer, key, None):
            raise ValueError(f"Target {key} differs from the checkpoint")
    if config.draft_vocab_size != target.vocab_size:
        raise ValueError("This experiment requires the full target vocabulary")
    verifier = VerifierConfig.from_pretrained(str(target_path), local_files_only=True)
    if verifier.architectures != config.speculators_config.verifier.architectures:
        raise ValueError("Target architecture differs")
    if any(i < 0 or i > target.num_hidden_layers for i in config.aux_hidden_state_layer_ids):
        raise ValueError("Target hidden-state indices are out of range")
    config.speculators_config.verifier = verifier
    checkpoint_keys = set(list_checkpoint_keys(checkpoint))
    target_keys = set(list_checkpoint_keys(target_path))
    if "model.norm.weight" not in target_keys:
        raise ValueError("Target lacks final norm weights")
    if not getattr(target, "tie_word_embeddings", False) and "lm_head.weight" not in target_keys:
        raise ValueError("Untied target lacks lm_head weights")
    model = RepairPrefixTraining.from_pretrained(str(checkpoint), config=config,
                                               local_files_only=True, dtype="auto")
    expected = set(model.state_dict())
    missing, unexpected = expected - checkpoint_keys - _TARGET_OWNED, checkpoint_keys - expected
    if missing or unexpected:
        raise ValueError(f"Incomplete/incompatible state: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}")
    model.configure_repair(settings)
    return model
