from typing import ClassVar, Literal

from pydantic import Field, model_validator
from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig


@SpeculatorModelConfig.register("dflash_prefix")
class DFlashPrefixSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash with causal attention over shortlisted, actually chosen tokens."""

    # A valid configuration needs an explicit matching target/draft vocabulary.
    # HF must not instantiate an invalid no-argument config while saving it.
    has_no_defaults_at_init: ClassVar[bool] = True

    speculators_model_type: Literal["dflash_prefix"] = "dflash_prefix"
    architectures: list[str] = Field(default_factory=lambda: ["DFlashPrefixDraftModel"])
    prefix_rank: int = Field(default=64, ge=8)
    prefix_top_k: int = Field(default=16, ge=2)
    prefix_loss_alpha: float = Field(default=1.0, ge=0)
    prefix_gate_init: float = Field(default=0.1, gt=0, lt=4)
    prefix_freeze_backbone: bool = False
    # Preserve legacy checkpoint behavior when these fields are absent. The
    # reviewed scratch launcher explicitly selects detached inputs + hard CE.
    prefix_detach_backbone: bool = False
    prefix_loss_kind: Literal["restricted_soft_ce", "target_ce"] = "restricted_soft_ce"
    prefix_walk_backend: Literal["torch", "triton", "auto"] = "torch"
    prefix_selector_kind: Literal["legacy", "local_prefix_v2"] = "legacy"
    prefix_attention_heads: int = Field(default=4, ge=1)
    prefix_attention_layers: int = Field(default=2, ge=1)
    prefix_history_scale: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    # Optional recovery experiment. Legacy checkpoints keep the old objective.
    prefix_loss_scope: Literal["all", "reachable_greedy_prefix"] = "all"
    prefix_damage_weight: float = Field(default=1.0, ge=1, allow_inf_nan=False)
    # Calibration applies only in serving. Training continues at scale one.
    prefix_inference_gate_scale: float = Field(
        default=1.0, ge=0, le=1, allow_inf_nan=False
    )
    prefix_validation_gate_scales: list[float] = Field(default_factory=list)
    sliding_window_non_causal: bool = True

    @model_validator(mode="after")
    def validate_prefix_settings(self):
        if self.sample_from_anchor:
            raise ValueError("dflash_prefix requires sample_from_anchor=False")
        if self.block_size < 2:
            raise ValueError("block_size must include an anchor and at least one mask")
        vocab = self.transformer_layer_config.vocab_size
        if self.draft_vocab_size != vocab:
            raise ValueError("dflash_prefix requires the full target vocabulary")
        if self.prefix_top_k > vocab:
            raise ValueError("prefix_top_k exceeds vocabulary size")
        if self.prefix_selector_kind == "local_prefix_v2":
            if self.prefix_rank % self.prefix_attention_heads:
                raise ValueError("prefix_rank must be divisible by attention heads")
            if self.prefix_walk_backend != "torch":
                raise ValueError("local_prefix_v2 currently requires the torch walk")
        if (self.prefix_loss_scope == "reachable_greedy_prefix"
                and self.prefix_loss_kind != "target_ce"):
            raise ValueError("reachable_greedy_prefix requires target_ce")
        if any(not 0 <= scale <= 1 for scale in self.prefix_validation_gate_scales):
            raise ValueError("Validation gate scales must be finite and in [0, 1]")
        return self
