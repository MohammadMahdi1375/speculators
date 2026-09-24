"""DFlash backbone + prefix attention, with a native-PyTorch training loss."""

import json
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar

import torch
from safetensors.torch import load_file
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics as base_metrics
from speculators.models.dflash_prefix.config import DFlashPrefixSpeculatorConfig
from speculators.models.dflash_prefix.selector_v2 import make_prefix_head, unary_topk


def reachable_greedy_prefix(reference, target_argmax, reference_covered):
    """Exclusive survival: this slot's label never determines its eligibility.

    Along greedy verification's surviving path, earlier reference tokens must
    both be representable by the shortlist and equal the target greedy tokens.
    This is a reachability mask, not on-policy training or target verification
    under an incorrectly selected prefix.
    """
    earlier_ok = reference.eq(target_argmax) & reference_covered
    return torch.cat(
        (torch.ones_like(earlier_ok[:, :1]), earlier_ok[:, :-1]), dim=-1
    ).long().cumprod(-1).bool()


@SpeculatorModel.register("dflash_prefix")
class DFlashPrefixDraftModel(DFlashDraftModel):
    config_class: ClassVar[type[DFlashPrefixSpeculatorConfig]] = (
        DFlashPrefixSpeculatorConfig
    )

    def __init__(self, config):
        super().__init__(config)
        # Adding a head must not consume the CPU RNG used by the scratch
        # backbone/data pipeline. Training constructs the master weights on CPU.
        with torch.random.fork_rng(devices=[]):
            self.prefix_head = make_prefix_head(
                config.transformer_layer_config.vocab_size,
                config.transformer_layer_config.hidden_size,
                config.prefix_rank,
                config.prefix_top_k,
                config.block_size - 1,
                config.prefix_gate_init,
                kind=config.prefix_selector_kind,
                heads=config.prefix_attention_heads,
                layers=config.prefix_attention_layers,
                history_scale=config.prefix_history_scale,
            )
        self._apply_backbone_freeze()

    def _apply_backbone_freeze(self):
        if self.config.prefix_freeze_backbone and hasattr(self, "prefix_head"):
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("prefix_head."))
        if (hasattr(self, "prefix_head")
                and self.config.prefix_selector_kind == "local_prefix_v2"):
            self.prefix_head.set_history_trainable(self.config.prefix_history_scale > 0)

    def _prefix_kwargs(self):
        if getattr(self.prefix_head, "requires_token_embeddings", False):
            return {"embedding_weight": self.embed_tokens.weight}
        return {}

    def load_verifier_weights(self):
        super().load_verifier_weights()
        self._apply_backbone_freeze()

    @classmethod
    def from_training_args(cls, verifier_config, t2d=None, d2t=None, **kwargs):
        base = cls._build_base_config_kwargs("dflash_prefix", verifier_config, **kwargs)
        if kwargs.get("sliding_window_non_causal") is None:
            base["sliding_window_non_causal"] = True
        config = DFlashPrefixSpeculatorConfig(
            **base,
            prefix_rank=kwargs.get("prefix_rank", 64),
            prefix_top_k=kwargs.get("prefix_top_k", 16),
            prefix_loss_alpha=kwargs.get("prefix_loss_alpha", 1.0),
            prefix_gate_init=kwargs.get("prefix_gate_init", 0.1),
            prefix_freeze_backbone=kwargs.get("prefix_freeze_backbone", False),
            prefix_detach_backbone=kwargs.get("prefix_detach_backbone", False),
            prefix_loss_kind=kwargs.get("prefix_loss_kind", "restricted_soft_ce"),
            prefix_walk_backend=kwargs.get("prefix_walk_backend", "torch"),
            prefix_selector_kind=kwargs.get("prefix_selector_kind", "legacy"),
            prefix_attention_heads=kwargs.get("prefix_attention_heads", 4),
            prefix_attention_layers=kwargs.get("prefix_attention_layers", 2),
            prefix_history_scale=kwargs.get("prefix_history_scale", 1.0),
        )
        model = cls(config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        init_path = kwargs.get("prefix_backbone_init")
        if config.prefix_freeze_backbone and not init_path:
            raise ValueError(
                "Freezing a new random backbone is invalid: set --prefix-backbone-init"
            )
        if init_path:
            model.load_dflash_backbone(init_path)
        return model

    def load_dflash_backbone(self, directory):
        """Warm-start from a speculators DFlash checkpoint; reject partial loads."""
        directory = Path(directory)
        cfg = json.loads((directory / "config.json").read_text())
        if cfg.get("speculators_model_type") != "dflash":
            raise ValueError(
                "Warm start requires a speculators-format plain DFlash checkpoint"
            )
        if cfg.get("sample_from_anchor", False):
            raise ValueError("Warm-start checkpoint must use sample_from_anchor=False")
        if cfg.get("aux_hidden_state_layer_ids") != self.target_layer_ids:
            raise ValueError(
                "Warm-start target layers differ; set --target-layer-ids to match"
            )
        if int(cfg.get("draft_vocab_size", 0)) != self.config.draft_vocab_size:
            raise ValueError("Warm start must use the full matching target vocabulary")
        layer_cfg = cfg["transformer_layer_config"]
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            if layer_cfg.get(key) != getattr(self.config.transformer_layer_config, key):
                raise ValueError(f"Warm-start {key} does not match requested backbone")
        index = directory / "model.safetensors.index.json"
        if index.exists():
            filenames = sorted(
                set(json.loads(index.read_text())["weight_map"].values())
            )
        elif (directory / "model.safetensors").exists():
            filenames = ["model.safetensors"]
        else:
            raise ValueError("Warm start needs model.safetensors or its shard index")
        state = self.state_dict()
        required = {
            name
            for name in state
            if name.startswith(("layers.", "fc.", "hidden_norm.", "norm."))
        }
        loaded = set()
        with torch.no_grad():
            for filename in filenames:
                for name, value in load_file(
                    str(directory / filename), device="cpu"
                ).items():
                    if name in required:
                        if state[name].shape != value.shape:
                            raise ValueError(
                                f"Warm-start tensor shape differs for {name}"
                            )
                        state[name].copy_(value)
                        loaded.add(name)
        if required - loaded:
            raise ValueError(
                f"Incomplete backbone checkpoint: {sorted(required - loaded)[:8]}"
            )

    @staticmethod
    def get_trainer_kwargs(**kwargs):
        return DFlashDraftModel.get_trainer_kwargs(**kwargs)

    @staticmethod
    def _record(metrics, name, values, valid):
        metrics[name + "_sum"] = (values.float() * valid.float()).sum().detach()
        metrics[name + "_total"] = valid.float().sum().detach()

    def forward(
        self,
        hidden_states,
        input_ids,
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
        position_ids=None,
        loss_config=None,
        gamma=4.0,
        max_anchors=128,
        per_position_loss_weight="dpace",
        dpace_alpha=0.5,
        **kwargs,
    ):
        frozen = self.config.prefix_freeze_backbone
        with torch.no_grad() if frozen else nullcontext():
            hidden, unary, targets, aligned_mask, indices = self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        block = self.block_size
        n = indices.numel() // block
        block_ids = input_ids[0, indices].reshape(n, block)
        block_docs = document_ids[0, indices].reshape(n, block)
        # Stop loss/metrics at the first document or assistant-mask boundary.
        valid = aligned_mask.reshape(n, block)[:, 1:].bool()
        valid = valid & block_docs[:, 1:].eq(block_docs[:, :1])
        valid = valid.long().cumprod(-1).bool()
        unary_loss, metrics = base_metrics(
            unary,
            targets,
            # Keep precisely the upstream DFlash objective. The selector and
            # reference-prefix diagnostics use the stricter boundary mask above.
            aligned_mask,
            block,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=False,
        )
        # Retain baseline metrics under explicit unary_ names. Head rollout
        # reference matching below is NOT an end-to-end serving acceptance claim.
        metrics = {"unary_" + key: value for key, value in metrics.items()}
        hidden = hidden.reshape(n, block, -1)[:, 1:]
        unary = unary.reshape(n, block, -1)[:, 1:]
        targets = targets.reshape(n, block, -1)[:, 1:]
        if self.config.prefix_detach_backbone:
            hidden = hidden.detach()
            unary = unary.detach()
        if self.config.prefix_selector_kind == "local_prefix_v2":
            candidate_logits, candidates = unary_topk(unary, self.config.prefix_top_k)
        else:
            candidate_logits, candidates = unary.topk(self.config.prefix_top_k, dim=-1)
        anchor_ids, reference = block_ids[:, 0], block_ids[:, 1:]
        logits = self.prefix_head.score_teacher_prefix(
            hidden,
            candidates,
            candidate_logits,
            anchor_ids,
            reference,
            **self._prefix_kwargs(),
        )
        target_argmax = targets.argmax(-1)
        target_matches = candidates.eq(target_argmax.unsqueeze(-1))
        target_covered = target_matches.any(-1)
        reference_covered = candidates.eq(reference.unsqueeze(-1)).any(-1)
        unary_ids = unary.argmax(-1)
        reachable = reachable_greedy_prefix(
            reference, target_argmax, reference_covered
        )
        if self.config.prefix_loss_kind == "target_ce":
            # Greedy verification can only match the target argmax when it is in
            # the natural shortlist. Never insert the label into that shortlist.
            label = target_matches.long().argmax(-1)
            element_loss = (
                -logits.log_softmax(-1).gather(-1, label.unsqueeze(-1)).squeeze(-1)
            )
            selector_valid = valid & target_covered
        else:
            teacher = targets.gather(-1, candidates).float().softmax(-1).detach()
            element_loss = -(teacher * logits.log_softmax(-1)).sum(-1)
            selector_valid = valid
        if self.config.prefix_loss_scope == "reachable_greedy_prefix":
            selector_valid = selector_valid & reachable
        # Fixed early-position weighting isolates the architecture; the unary
        # backbone keeps the configured baseline D-PACE/CE recipe.
        positions = torch.arange(block - 1, device=logits.device).float()
        weights = selector_valid.float() * torch.exp(-positions / gamma)
        weights = weights * torch.where(
            unary_ids.eq(target_argmax), self.config.prefix_damage_weight, 1.0
        )
        selector_loss = (element_loss * weights).sum() / weights.sum().clamp_min(1.0)
        loss = unary_loss + self.config.prefix_loss_alpha * selector_loss
        if frozen:
            loss = self.config.prefix_loss_alpha * selector_loss
        one = loss.detach().new_tensor(1.0)
        metrics.update(
            {
                "loss_sum": loss.detach(),
                "loss_total": one,
                "selector_ce_sum": selector_loss.detach(),
                "selector_ce_total": one,
                "selector_gate_sum": self.prefix_head.gate.detach(),
                "selector_gate_total": one,
            }
        )
        with torch.no_grad():
            teacher_prediction = candidates.gather(
                -1, logits.argmax(-1, keepdim=True)
            ).squeeze(-1)
            self._record(metrics, "selector_supervised_fraction", selector_valid, valid)
            # Match the selector-off server, including BF16 logit ties.
            unary_correct = unary_ids.eq(target_argmax)
            selector_correct = teacher_prediction.eq(target_argmax)
            self._record(
                metrics, "teacher_rescue", selector_correct & ~unary_correct, valid
            )
            self._record(
                metrics, "teacher_damage", ~selector_correct & unary_correct, valid
            )
            self._record(
                metrics,
                "teacher_argmax_acc",
                teacher_prediction.eq(target_argmax),
                valid,
            )
            self._record(metrics, "reachable_greedy_fraction", reachable, valid)
            self._record(metrics, "candidate_target_recall", target_covered, valid)
            self._record(
                metrics, "candidate_reference_recall", reference_covered, valid
            )
            if not self.training:
                # Chunk anchors to bound validation working memory for either head.
                paths = [
                    self.prefix_head.greedy(
                        hidden[start : start + 8],
                        candidates[start : start + 8],
                        candidate_logits[start : start + 8],
                        anchor_ids[start : start + 8],
                        **self._prefix_kwargs(),
                    )
                    for start in range(0, n, 8)
                ]
                chosen = torch.cat(paths, dim=0)
                for start in range(0, n, 8):
                    scales = self.config.prefix_validation_gate_scales
                    if not scales:
                        break
                    stop = start + 8
                    table = self.prefix_head.prepare_tables(
                        hidden[start:stop], candidates[start:stop],
                        candidate_logits[start:stop], anchor_ids[start:stop],
                        **self._prefix_kwargs(),
                    )
                    for scale in scales:
                        if scale == 0:
                            selected = unary_ids[start:stop]
                        elif scale == 1:
                            selected = chosen[start:stop]
                        else:
                            if self.config.prefix_selector_kind == "local_prefix_v2":
                                selected = self.prefix_head.greedy_walk_scaled(table, scale)
                            else:
                                selected = self.prefix_head.greedy_walk(
                                    table._replace(gate=table.gate * scale)
                                )
                        prefix = (
                            selected.eq(reference[start:stop]) & valid[start:stop]
                        ).long().cumprod(-1)
                        suffix = format(scale, ".6g").replace(".", "p")
                        key = "calibration_reference_eal_scale_" + suffix
                        value = (1 + prefix.sum(-1)).float()
                        block_valid = valid[start:stop, 0].float()
                        for part, term in (
                            ("_sum", (value * block_valid).sum()),
                            ("_total", block_valid.sum()),
                        ):
                            name = key + part
                            metrics[name] = metrics.get(name, 0) + term
                correct_prefix = (chosen.eq(reference) & valid).long().cumprod(-1)
                unary_prefix = (
                    (unary_ids.eq(reference) & valid).long().cumprod(-1)
                )
                oracle_prefix = (reference_covered & valid).long().cumprod(-1)
                valid_blocks = valid[:, 0]
                if (self.config.prefix_selector_kind == "local_prefix_v2"
                        and self.prefix_head.history_scale > 0):
                    saved_scale = self.prefix_head.history_scale
                    try:
                        self.prefix_head.history_scale = 0.0
                        local_chosen = torch.cat([
                            self.prefix_head.greedy(
                                hidden[start:start+8], candidates[start:start+8],
                                candidate_logits[start:start+8], anchor_ids[start:start+8],
                                **self._prefix_kwargs())
                            for start in range(0, n, 8)], dim=0)
                    finally:
                        self.prefix_head.history_scale = saved_scale
                    local_prefix = (local_chosen.eq(reference) & valid).long().cumprod(-1)
                    self._record(metrics, "local_reference_eal",
                                 1 + local_prefix.sum(-1), valid_blocks)
                    self._record(metrics, "history_reference_eal_delta",
                                 correct_prefix.sum(-1) - local_prefix.sum(-1), valid_blocks)
                for label, correct in (("selector", chosen.eq(reference)),
                                       ("unary", unary_ids.eq(reference)),
                                       ("oracle", reference_covered)):
                    earlier = torch.cat((torch.ones_like(correct[:, :1]),
                                         correct[:, :-1] & valid[:, :-1]), dim=-1)
                    alive = earlier.long().cumprod(-1).bool() & valid
                    for pos in range(block - 1):
                        self._record(metrics, f"{label}_conditional_pos_{pos+1}",
                                     correct[:, pos], alive[:, pos])
                self._record(
                    metrics,
                    "rollout_reference_eal",
                    1 + correct_prefix.sum(-1),
                    valid_blocks,
                )
                self._record(
                    metrics,
                    "unary_reference_eal",
                    1 + unary_prefix.sum(-1),
                    valid_blocks,
                )
                self._record(
                    metrics,
                    "oracle_reference_eal",
                    1 + oracle_prefix.sum(-1),
                    valid_blocks,
                )
                self._record(
                    metrics, "rollout_reference_acc", chosen.eq(reference), valid
                )
        return None, loss, metrics
