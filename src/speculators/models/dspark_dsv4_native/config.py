"""Native DeepSeek-V4 DSpark drafter config.

This is intentionally separate from ``speculators.models.dspark``.  The current
``dspark`` implementation in this repository is DFlash-backed; this config is
for the official DeepSeek-V4 ``mtp.*`` DSpark module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.config import SpeculatorsConfig

__all__ = ["DeepSeekV4NativeDSparkConfig"]


@SpeculatorModelConfig.register("dspark_dsv4_native")
class DeepSeekV4NativeDSparkConfig(SpeculatorModelConfig):
    """Config for the official-style native DeepSeek-V4 DSpark drafter.

    The fields mirror the released DeepSeek-V4-Flash-DSpark inference config:
    3 native ``DSparkBlock`` stages under ``mtp.*``, MoE FFNs, multi-head
    Hyper-Connections, sliding-window DSpark attention, a rank-256 Markov head,
    and a confidence head.
    """

    speculators_model_type: Literal["dspark_dsv4_native"] = "dspark_dsv4_native"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DeepSeekV4NativeDSparkModel"],
        description="Model architectures that can load these native DSpark weights.",
    )

    # DeepSeek-V4 target dimensions.
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3

    # Official DSpark fields.
    n_mtp_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: list[int] = Field(default_factory=lambda: [40, 41, 42])
    dspark_markov_rank: int = 256

    # Attention.
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    rope_theta: float = 10000.0

    # MoE.
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # Hyper-Connections.
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # Runtime/training.
    norm_eps: float = 1e-6
    max_anchors: int = 128
    loss_fn: str = "kl_div"
    confidence_head_alpha: float = 1.0
    temperature: float = 0.0

    # Stored for compatibility with the speculators training/export plumbing.
    draft_vocab_size: int = 129280
    block_size: int = 5
    aux_hidden_state_layer_ids: list[int] = Field(default_factory=lambda: [40, 41, 42])
    speculators_config: SpeculatorsConfig | None = Field(default=None)
