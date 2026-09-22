"""Independent checkpoint identity for the DSpARK-based ReTrace drafter."""

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from speculators.config import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig


@SpeculatorModelConfig.register("retrace_dspark")
class ReTraceDSparkSpeculatorConfig(DSparkSpeculatorConfig):
    # Full-vocabulary and target dimensions must be supplied together. HF must
    # not construct an invalid no-argument config when saving a checkpoint.
    has_no_defaults_at_init: ClassVar[bool] = True
    speculators_model_type: Literal["retrace_dspark"] = "retrace_dspark"
    architectures: list[str] = Field(
        default_factory=lambda: ["ReTraceDSparkDraftModel"]
    )
    retrace_enabled: bool = True
    retrace_beta_max: float = Field(default=1.0, ge=0)
    retrace_warmup_steps: int = Field(default=50, ge=0)
    retrace_format_version: Literal[1] = 1
    retrace_backbone: Literal["dspark"] = "dspark"
    training_objective: Literal[
        "dspark_on_policy", "dspark_clean_block", "dspark_stored_pair"
    ] = "dspark_on_policy"
    pretrained_source: str | None = None
    pretrained_fingerprint: str | None = None
    sample_from_anchor: Literal[True] = True
    markov_head_type: Literal["vanilla"] = "vanilla"
    markov_rank: int = Field(default=256, gt=0)
    enable_confidence_head: Literal[True] = True
    confidence_head_with_markov: Literal[True] = True
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {"ce": 0.1, "tv": 0.9}
    )
    confidence_head_alpha: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_retrace(self):
        if self.block_size < 2:
            raise ValueError("DSpARK block_size must be at least 2")
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
        if (
            not self.loss_weights
            or any(
                key not in {"ce", "tv"} or value < 0
                for key, value in self.loss_weights.items()
            )
            or sum(self.loss_weights.values()) <= 0
        ):
            raise ValueError(
                "loss_weights must contain nonnegative ce/tv weights with a positive sum"
            )
        return self
