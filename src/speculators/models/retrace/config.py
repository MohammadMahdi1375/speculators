"""Independent checkpoint identity for the DFlash-based ReTrace drafter."""

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from speculators.config import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig


@SpeculatorModelConfig.register("retrace")
class ReTraceSpeculatorConfig(DFlashSpeculatorConfig):
    # Full-vocabulary and target dimensions must be supplied together. HF must
    # not construct an invalid no-argument config when saving a checkpoint.
    has_no_defaults_at_init: ClassVar[bool] = True
    speculators_model_type: Literal["retrace"] = "retrace"
    architectures: list[str] = Field(default_factory=lambda: ["ReTraceDraftModel"])
    retrace_enabled: bool = True
    retrace_beta_max: float = Field(default=1.0, ge=0)
    retrace_warmup_steps: int = Field(default=50, ge=0)
    retrace_format_version: Literal[1] = 1
    training_objective: Literal[
        "verification_soft_ce", "clean_block_ce", "stored_pair_kl"
    ] = "verification_soft_ce"
    pretrained_source: str | None = None
    pretrained_fingerprint: str | None = None

    @model_validator(mode="after")
    def validate_retrace(self):
        if self.sample_from_anchor:
            raise ValueError("ReTrace uses a verified anchor followed by mask slots")
        if self.block_size < 2:
            raise ValueError("block_size must include an anchor and at least one mask")
        if self.draft_vocab_size != self.target_vocab_size:
            raise ValueError(
                "This ReTrace implementation requires full target vocabulary"
            )
        if self.transformer_layer_config.model_type != "qwen3":
            raise ValueError("This implementation supports dense Qwen3")
        if any(
            x != "full_attention" for x in self.transformer_layer_config.layer_types
        ):
            raise ValueError("ReTrace currently requires full attention")
        return self
