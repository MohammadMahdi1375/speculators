from __future__ import annotations

import torch


def _make_mp_policy():
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy

        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
    except Exception:
        return None


def _fully_shard(module, mp_policy=None):
    from torch.distributed.fsdp import fully_shard

    if mp_policy is None:
        fully_shard(module)
    else:
        fully_shard(module, mp_policy=mp_policy)


def apply_native_dspark_expert_fsdp2(model):
    """Apply MoE-aware FSDP2 wrapping to native DeepSeek-V4 DSpark.

    Architecture is unchanged. Only sharding granularity changes:
      - each routed expert is wrapped separately
      - shared expert is wrapped separately
      - gate, attention, block, and root are wrapped after children
    """

    mp_policy = _make_mp_policy()

    mtp_layers = getattr(model, "mtp", None)
    if mtp_layers is None:
        raise TypeError("Expected native DSpark model with .mtp layers")

    num_experts_wrapped = 0

    for layer in mtp_layers:
        ffn = getattr(layer, "ffn", None)
        if ffn is not None:
            experts = getattr(ffn, "experts", None)
            if experts is not None:
                for expert in experts:
                    _fully_shard(expert, mp_policy=mp_policy)
                    num_experts_wrapped += 1

            shared = getattr(ffn, "shared_experts", None)
            if shared is not None:
                _fully_shard(shared, mp_policy=mp_policy)

            gate = getattr(ffn, "gate", None)
            if gate is not None:
                _fully_shard(gate, mp_policy=mp_policy)

            _fully_shard(ffn, mp_policy=mp_policy)

        attn = getattr(layer, "attn", None)
        if attn is not None:
            _fully_shard(attn, mp_policy=mp_policy)

        _fully_shard(layer, mp_policy=mp_policy)

    _fully_shard(model, mp_policy=mp_policy)

    return {
        "num_mtp_layers": len(mtp_layers),
        "num_experts_wrapped": num_experts_wrapped,
    }
