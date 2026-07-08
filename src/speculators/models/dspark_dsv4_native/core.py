"""Native DeepSeek-V4 DSpark SpeculatorModel.

This model is for the exact official HF goal: train/fine-tune the native
DeepSeek-V4 DSpark drafter stored as ``mtp.*`` weights.  It deliberately does
not inherit the repository's DFlash-backed ``dspark`` model.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import ClassVar, Iterable, Optional

import torch
from safetensors.torch import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.utils import resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers

from .config import DeepSeekV4NativeDSparkConfig
from .modeling import (
    GatheredAnchors,
    NativeDSparkBlock,
    NativeParallelHead,
    RMSNorm,
)

logger = logging.getLogger(__name__)

__all__ = ["DeepSeekV4NativeDSparkModel"]


def _getattr_any(obj, names: Iterable[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return default


def _to_dict_config(config: PretrainedConfig | dict) -> dict:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(getattr(config, "__dict__", {}))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum().clamp_min(1.0)
    return (x.float() * mask.float()).sum() / denom


@SpeculatorModel.register("dspark_dsv4_native")
class DeepSeekV4NativeDSparkModel(SpeculatorModel):
    """Official-style native DeepSeek-V4 DSpark drafter.

    Shape/namespace contract:
      * trainable DSpark blocks are under ``self.mtp`` and save as ``mtp.*``
      * frozen verifier embedding/head/norm are loaded only for training loss
      * ``dspark_block_size`` is official semantics: number of emitted draft
        tokens, not DFlash's anchor+draft convention
    """

    config_class: ClassVar[type[DeepSeekV4NativeDSparkConfig]] = DeepSeekV4NativeDSparkConfig
    _no_split_modules: ClassVar[list[str]] = ["NativeDSparkBlock"]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[assignment]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[assignment]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    def __init__(self, config: DeepSeekV4NativeDSparkConfig) -> None:
        super().__init__(config=config)
        self.config = config
        # Tell Trainer not to materialize/copy a normal full state_dict into
        # DTensor parameters after FSDP. Native DSpark is too large for that path.
        self.skip_fsdp_rank0_state_broadcast = True
        self.hidden_size = config.dim
        self.block_size = config.dspark_block_size
        self.verifier_vocab_size = config.vocab_size
        self.draft_vocab_size = config.vocab_size
        self.use_draft_vocab = False
        self.t2d: torch.Tensor | None = None
        self.d2t: torch.Tensor | None = None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.lm_head = NativeParallelHead(config.vocab_size, config.dim, bias=False)
        self.verifier_lm_head = NativeParallelHead(config.vocab_size, config.dim, bias=False)
        self.verifier_norm = RMSNorm(config.dim, config.norm_eps)
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        self.verifier_lm_head.weight.requires_grad_(False)
        self.verifier_norm.weight.requires_grad_(False)

        # ``layers`` is expected by the repository Trainer/FSDP compatibility
        # check.  For native DSpark these are the actual MTP stages.
        self.mtp = nn.ModuleList(
            [
                NativeDSparkBlock(config.n_layers + i, config)
                for i in range(config.n_mtp_layers)
            ]
        )
        self.layers = self.mtp

        self.post_init()

        # After post_init(), make frozen verifier weights obvious if loading was
        # skipped.  This matches DraftVocabMixin's NaN sentinel behavior.
        with torch.no_grad():
            self.embed_tokens.weight.fill_(float("nan"))
            self.lm_head.weight.fill_(float("nan"))
            self.verifier_lm_head.weight.fill_(float("nan"))

    def _init_weights(self, module: nn.Module) -> None:  # transformers hook
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif hasattr(module, "attn_sink") and isinstance(getattr(module, "attn_sink"), nn.Parameter):
            nn.init.zeros_(module.attn_sink)
        elif hasattr(module, "weight") and isinstance(getattr(module, "weight"), nn.Parameter):
            # Direct-parameter modules such as the DeepSeek MoE gate.  Linear and
            # Embedding modules were handled above.
            if not isinstance(module, (nn.Linear, nn.Embedding, RMSNorm)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                bias = getattr(module, "bias", None)
                if isinstance(bias, nn.Parameter):
                    nn.init.zeros_(bias)
        elif isinstance(module, NativeDSparkBlock):
            # Important HC parameters.  Small values keep early training stable.
            for name, param in module.named_parameters(recurse=False):
                if name.endswith("_fn"):
                    nn.init.normal_(param, mean=0.0, std=0.01)
                elif name.endswith("_base"):
                    nn.init.zeros_(param)
                elif name.endswith("_scale"):
                    nn.init.ones_(param)

    @property
    def target_layer_ids(self) -> list[int]:
        return list(self.config.dspark_target_layer_ids)

    def load_vocab_mappings(self, t2d: torch.Tensor | None, d2t: torch.Tensor | None) -> None:
        if t2d is not None or d2t is not None:
            logger.warning(
                "dspark_dsv4_native uses the full verifier vocabulary; ignoring t2d/d2t mappings."
            )
        self.t2d = None
        self.d2t = None

    def load_verifier_weights(self) -> None:
        speculators_config = getattr(self.config, "speculators_config", None)
        if speculators_config is None or speculators_config.verifier.name_or_path is None:
            logger.warning("No verifier path in config; verifier embed/head/norm remain uninitialized.")
            return
        verifier_path = speculators_config.verifier.name_or_path
        weights = load_model_layers(
            [
                "embed_tokens.weight",
                "embed.weight",
                "lm_head.weight",
                "head.weight",
                "model.norm.weight",
                "norm.weight",
            ],
            verifier_path,
        )
        embed_w = weights.get("embed_tokens.weight", weights.get("embed.weight"))
        if embed_w is None:
            raise RuntimeError(f"Could not find verifier embedding weights in {verifier_path}")
        head_w = weights.get("lm_head.weight", weights.get("head.weight", embed_w))
        norm_w = weights.get("model.norm.weight", weights.get("norm.weight"))

        self.embed_tokens.load_state_dict({"weight": embed_w.to(self.embed_tokens.weight.dtype)})
        self.lm_head.load_state_dict({"weight": head_w.to(self.lm_head.weight.dtype)})
        self.verifier_lm_head.load_state_dict({"weight": head_w.to(self.verifier_lm_head.weight.dtype)})
        if norm_w is not None:
            self.verifier_norm.load_state_dict({"weight": norm_w.to(self.verifier_norm.weight.dtype)})

        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        self.verifier_lm_head.weight.requires_grad_(False)
        self.verifier_norm.weight.requires_grad_(False)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DeepSeekV4NativeDSparkModel":
        # ``verifier_config`` may be a Transformers config or the raw DeepSeek-V4
        # config shim from scripts/train.py.
        vc = getattr(verifier_config, "text_config", verifier_config)
        cfg_dict = _to_dict_config(vc)
        verifier_name = kwargs["verifier_name_or_path"]
        target_layer_ids = resolve_target_layer_ids(kwargs.get("target_layer_ids"), verifier_name)

        def c(name: str, *aliases: str, default=None):
            return _getattr_any(vc, (name, *aliases), cfg_dict.get(name, default))

        dim = int(c("dim", "hidden_size", default=4096))
        n_layers = int(c("n_layers", "num_hidden_layers", default=43))
        n_heads = int(c("n_heads", "num_attention_heads", default=64))
        head_dim = int(c("head_dim", default=512))
        vocab_size = int(c("vocab_size", default=129280))

        block_size = int(kwargs.get("block_size") or c("dspark_block_size", default=5))
        noise_token_id = kwargs.get("mask_token_id")
        if noise_token_id is None:
            noise_token_id = int(c("dspark_noise_token_id", default=128799))

        cfg = DeepSeekV4NativeDSparkConfig(
            vocab_size=vocab_size,
            dim=dim,
            moe_inter_dim=int(c("moe_inter_dim", "moe_intermediate_size", default=2048)),
            n_layers=n_layers,
            n_hash_layers=int(c("n_hash_layers", default=3)),
            n_mtp_layers=int(kwargs.get("num_layers") or c("n_mtp_layers", default=3)),
            dspark_block_size=block_size,
            block_size=block_size,
            dspark_noise_token_id=int(noise_token_id),
            dspark_target_layer_ids=list(target_layer_ids),
            aux_hidden_state_layer_ids=list(target_layer_ids),
            dspark_markov_rank=int(kwargs.get("markov_rank") or c("dspark_markov_rank", default=256)),
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=int(c("rope_head_dim", default=64)),
            q_lora_rank=int(c("q_lora_rank", default=1024)),
            o_groups=int(c("o_groups", default=8)),
            o_lora_rank=int(c("o_lora_rank", default=1024)),
            window_size=int(kwargs.get("sliding_window") or c("window_size", default=128)),
            rope_theta=float(c("rope_theta", default=10000.0)),
            n_routed_experts=int(c("n_routed_experts", default=256)),
            n_shared_experts=int(c("n_shared_experts", default=1)),
            n_activated_experts=int(c("n_activated_experts", "num_experts_per_tok", default=6)),
            score_func=str(c("score_func", "scoring_func", default="sqrtsoftplus")),
            route_scale=float(c("route_scale", "routed_scaling_factor", default=1.5)),
            swiglu_limit=float(c("swiglu_limit", default=10.0)),
            hc_mult=int(c("hc_mult", default=4)),
            hc_sinkhorn_iters=int(c("hc_sinkhorn_iters", default=20)),
            norm_eps=float(c("norm_eps", "rms_norm_eps", default=1e-6)),
            max_anchors=int(kwargs.get("max_anchors", 128)),
            loss_fn=str(kwargs.get("loss_fn", "kl_div")),
            confidence_head_alpha=float(kwargs.get("confidence_head_alpha", 1.0)),
            draft_vocab_size=vocab_size,
            speculators_config=SpeculatorsConfig(
                algorithm="dspark_dsv4_native",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block_size)],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(verifier_name),
            ),
        )
        model = cls(config=cfg)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        shared = {
            "loss_fn": kwargs.get("loss_fn", "kl_div"),
            "confidence_head_alpha": kwargs.get("confidence_head_alpha", 1.0),
        }
        return dict(shared), dict(shared)

    def load_mtp_weights_from_hf(self, checkpoint_dir: str, strict: bool = False) -> None:
        """Load official HF/native DSpark ``mtp.*`` weights if present.

        The official checkpoint may contain FP4/FP8 scale metadata or tensor-
        parallel slices.  This loader copies exact-shape tensors and reports
        skipped keys, so it is safe for both fine-tuning and scratch training.
        """

        state: dict[str, torch.Tensor] = {}
        for path in glob.glob(os.path.join(checkpoint_dir, "*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("mtp."):
                        state[key] = f.get_tensor(key)
        if not state:
            raise RuntimeError(f"No mtp.* tensors found in {checkpoint_dir}")

        own = self.state_dict()
        loadable = {}
        skipped = []
        for key, tensor in state.items():
            if key not in own:
                skipped.append((key, tuple(tensor.shape), "not-in-model"))
                continue
            if tuple(own[key].shape) != tuple(tensor.shape):
                skipped.append((key, tuple(tensor.shape), f"expected {tuple(own[key].shape)}"))
                continue
            loadable[key] = tensor.to(dtype=own[key].dtype)
        missing, unexpected = self.load_state_dict(loadable, strict=False)
        logger.info("loaded %d native DSpark mtp tensors from %s", len(loadable), checkpoint_dir)
        if skipped:
            logger.warning("skipped %d mtp tensors due to names/shapes; first 20: %s", len(skipped), skipped[:20])
        if strict and (missing or unexpected or skipped):
            raise RuntimeError(
                f"strict load failed: missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
            )

    def _select_anchors(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        document_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select up to max_anchors per batch item for official K-token DSpark blocks."""

        bsz, seq_len = input_ids.shape
        k = self.block_size
        all_positions = []
        all_valid = []
        for b in range(bsz):
            valid_positions = []
            # p predicts p+1 ... p+k, so p+k must be in-range.
            for p in range(max(0, seq_len - k)):
                if loss_mask[b, p + 1 : p + k + 1].bool().all() and (
                    document_ids[b, p : p + k + 1] == document_ids[b, p]
                ).all():
                    valid_positions.append(p)
            if valid_positions:
                pos = torch.tensor(valid_positions, device=input_ids.device, dtype=torch.long)
                if pos.numel() > self.config.max_anchors:
                    take = torch.linspace(
                        0, pos.numel() - 1, self.config.max_anchors, device=input_ids.device
                    ).long()
                    pos = pos[take]
                valid = torch.ones(pos.numel(), device=input_ids.device, dtype=torch.bool)
                if pos.numel() < self.config.max_anchors:
                    pad = self.config.max_anchors - pos.numel()
                    pos = torch.cat([pos, pos.new_zeros(pad)])
                    valid = torch.cat([valid, valid.new_zeros(pad)])
            else:
                pos = torch.zeros(self.config.max_anchors, device=input_ids.device, dtype=torch.long)
                valid = torch.zeros(self.config.max_anchors, device=input_ids.device, dtype=torch.bool)
            all_positions.append(pos)
            all_valid.append(valid)
        return torch.stack(all_positions, dim=0), torch.stack(all_valid, dim=0)

    def _gather_anchor_tensors(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> GatheredAnchors:
        device = hidden_states.device
        bsz, seq_len, _ = hidden_states.shape
        k = self.block_size
        window = min(self.config.window_size, seq_len)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)

        anchor_pos, anchor_valid = self._select_anchors(input_ids, loss_mask, document_ids)
        # Flatten batch x anchors.  Existing trainer usually uses B=1, but this
        # supports packed batches too.
        flat_b = torch.arange(bsz, device=device).unsqueeze(1).expand_as(anchor_pos).reshape(-1)
        flat_p = anchor_pos.reshape(-1)
        flat_valid = anchor_valid.reshape(-1)

        anchor_ids = input_ids[flat_b, flat_p]
        offsets = torch.arange(1, k + 1, device=device)
        target_pos = flat_p.unsqueeze(1) + offsets.unsqueeze(0)
        prev_pos = torch.cat([flat_p.unsqueeze(1), target_pos[:, :-1]], dim=1)
        target_token_ids = input_ids[flat_b.unsqueeze(1), target_pos]
        prev_token_ids = input_ids[flat_b.unsqueeze(1), prev_pos]

        # Verifier logits at position i should predict token i; the target model
        # raw logits at t predict t+1, so roll right by one as in DFlash.
        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(self.verifier_norm(verifier_last_hidden_states))
            verifier_logits = torch.roll(verifier_logits, shifts=1, dims=1)
            target_logits = verifier_logits[flat_b.unsqueeze(1), target_pos]

        ctx_offsets = torch.arange(window, device=device)
        ctx_pos = flat_p.unsqueeze(1) - (window - 1 - ctx_offsets).unsqueeze(0)
        ctx_pos = ctx_pos.clamp(min=0)
        main_hidden_context = hidden_states[flat_b.unsqueeze(1), ctx_pos]
        main_position_ids = position_ids[flat_b.unsqueeze(1), ctx_pos]
        draft_position_ids = position_ids[flat_b.unsqueeze(1), target_pos]

        valid_mask = flat_valid.unsqueeze(1).expand(-1, k).to(loss_mask.dtype)
        return GatheredAnchors(
            anchor_ids=anchor_ids,
            prev_token_ids=prev_token_ids,
            target_token_ids=target_token_ids,
            target_logits=target_logits,
            main_hidden_context=main_hidden_context,
            main_position_ids=main_position_ids,
            draft_position_ids=draft_position_ids,
            valid_mask=valid_mask,
        )

    def _loss_and_metrics(
        self,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        target_token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence_logits: torch.Tensor,
        loss_fn: str,
        confidence_head_alpha: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if valid_mask.sum() <= 0:
            zero = logits.sum() * 0.0
            return zero, {"loss": zero.detach(), "num_valid": valid_mask.sum().detach()}

        draft_logp = F.log_softmax(logits.float(), dim=-1)
        target_p = F.softmax(target_logits.float(), dim=-1)
        target_logp = F.log_softmax(target_logits.float(), dim=-1)
        if loss_fn == "ce":
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target_token_ids.reshape(-1),
                reduction="none",
            ).view_as(valid_mask)
        else:
            token_loss = (target_p * (target_logp - draft_logp)).sum(dim=-1)
        draft_p = draft_logp.exp()
        acceptance_overlap = torch.minimum(draft_p, target_p).sum(dim=-1).detach()
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence_logits.float(), acceptance_overlap.float(), reduction="none"
        )
        draft_loss = _masked_mean(token_loss, valid_mask)
        conf_loss = _masked_mean(confidence_loss, valid_mask)
        loss = draft_loss + confidence_head_alpha * conf_loss
        greedy = logits.argmax(dim=-1)
        acc = _masked_mean((greedy == target_token_ids).float(), valid_mask)
        metrics = {
            "loss": loss.detach(),
            "draft_loss": draft_loss.detach(),
            "confidence_loss": conf_loss.detach(),
            "acceptance_overlap": _masked_mean(acceptance_overlap, valid_mask).detach(),
            "greedy_token_acc": acc.detach(),
            "num_valid": valid_mask.sum().detach(),
        }
        return loss, metrics

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_fn: str = "kl_div",
        confidence_head_alpha: float = 1.0,
        **kwargs,  # noqa: ARG002
    ):
        gathered = self._gather_anchor_tensors(
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last_hidden_states,
            document_ids=document_ids,
            position_ids=position_ids,
        )

        x, main_x, draft_input_ids = self.mtp[0].forward_embed(
            gathered.main_hidden_context, gathered.anchor_ids, self.embed_tokens
        )
        for layer in self.mtp:
            x = layer(
                x,
                draft_input_ids,
                main_x,
                gathered.main_position_ids,
                gathered.draft_position_ids,
            )
        logits, confidence_logits, _hidden = self.mtp[-1].forward_head_teacher(
            x,
            gathered.anchor_ids,
            gathered.prev_token_ids,
            self.lm_head,
        )
        loss, metrics = self._loss_and_metrics(
            logits=logits,
            target_logits=gathered.target_logits,
            target_token_ids=gathered.target_token_ids,
            valid_mask=gathered.valid_mask,
            confidence_logits=confidence_logits,
            loss_fn=loss_fn,
            confidence_head_alpha=confidence_head_alpha,
        )
        draft_tokens = logits.argmax(dim=-1)
        return draft_tokens, loss, metrics
