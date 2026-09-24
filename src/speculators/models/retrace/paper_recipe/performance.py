"""Optional execution changes for the pretrained DFlash trainer.

Cache lifetime is ONE no-grad rollout with fixed weights. No cache is used for
gradient replay, carried to another prompt, or serialized in a checkpoint.
The target, rejection test, clean labels and loss normalization are unchanged.
"""

import torch

from speculators.models.dflash.model_definitions import apply_rotary_pos_emb

from .core import DraftBlock
from .execution import draft_attention
from .memory import condition_query_block


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_gib(device):
    if device.type == "npu":
        return torch.npu.max_memory_allocated(device) / 2**30
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    return 0.0


def validate_resume_signature(
    saved, requested, allow_performance_change=False, migrate_to_vllm=False
):
    """An explicit migration may change execution only, never the recipe."""
    defaults = {
        "performance_mode": "reference",
        "trace_storage": "cpu",
        "rollout_batch_size": 1,
        "attention_backend": "eager",
        "target_norm": "reference",
        "work_distribution": "static",
    }
    old, new = dict(saved), dict(requested)
    for key, value in defaults.items():
        old.setdefault(key, value)
        new.setdefault(key, value)
    changes = {
        key: {"before": old.get(key), "after": new.get(key)}
        for key in old.keys() | new.keys()
        if old.get(key) != new.get(key)
    }
    allowed = {
        "performance_mode",
        "trace_storage",
        "blocks_per_forward",
        "rollout_batch_size",
        "attention_backend",
        "target_norm",
        "work_distribution",
    }
    permitted = allowed if allow_performance_change else set()
    if migrate_to_vllm:
        old_world, new_world = old.get("world_size"), new.get("world_size")
        if not (
            old.get("target_backend") == "local"
            and new.get("target_backend") == "vllm"
            and type(old_world) is int and type(new_world) is int
            and 1 <= new_world <= old_world
            and new.get("target_norm") == "reference"
        ):
            raise ValueError(
                "--migrate-to-vllm requires a local-target checkpoint, a vLLM target, "
                "reference target norm and no increase in trainer worker count"
            )
        permitted = permitted | {"target_backend", "world_size", "target_norm"}
    if changes and not changes.keys() <= permitted:
        raise ValueError(
            "Resume configuration/prompt pool/world size differs from the checkpoint: "
            f"{changes}. Execution-only changes require --allow-performance-change; "
            "local-to-vLLM migration requires --migrate-to-vllm. Recipe changes remain forbidden."
        )
    return changes


class DraftContextCache:
    """Incremental FC and layer context K/V projection for a single rollout.

    Only committed context enters this cache. Mask/query keys are recomputed every
    round. RoPE is applied afresh to the assembled keys, so cached tensors contain
    no stale positional transformation. Tokenwise projections can have small BF16
    rounding differences from full-prefix GEMMs; bitwise identity is not promised.
    """

    def __init__(self, draft):
        self.draft = draft
        self.length = 0
        self.keys = [None] * len(draft.layers)
        self.values = [None] * len(draft.layers)
        self.parameter_versions = tuple(p._version for p in draft.parameters())
        self.projected_context_tokens = 0

    def propose(self, context_hidden, anchor, memory, beta):
        draft = self.draft
        if torch.is_grad_enabled() or draft.training:
            raise RuntimeError("Draft cache is restricted to eval/no-grad rollouts")
        if self.parameter_versions != tuple(p._version for p in draft.parameters()):
            raise RuntimeError("Discard the draft cache after a parameter update")
        length = context_hidden.shape[1]
        if context_hidden.shape[0] != 1 or anchor.shape != (1,) or length < self.length:
            raise ValueError("Draft cache requires one monotonically growing prompt")
        if length == 0:
            raise ValueError("A nonempty committed prefix is required")
        added = length - self.length
        context = (
            draft.hidden_norm(draft.fc(context_hidden[:, self.length :].detach()))
            if added
            else None
        )
        ids = torch.full(
            (1, draft.block_size),
            draft.mask_token_id,
            device=anchor.device,
            dtype=torch.long,
        )
        ids[:, 0] = anchor
        hidden = draft.embed_tokens(ids)
        if draft.config.retrace_enabled and memory is not None:
            hidden = condition_query_block(draft.retrace, hidden, memory, beta)
        positions = torch.arange(length + draft.block_size, device=anchor.device)[None]
        rotary = draft.rotary_emb(context_hidden, positions)
        mask = hidden.new_zeros(1, 1, draft.block_size, length + draft.block_size)
        for i, layer in enumerate(draft.layers):
            attn = layer.self_attn
            residual = hidden
            query_hidden = layer.input_layernorm(hidden)
            size = query_hidden.shape[1]
            q = attn.q_norm(attn.q_proj(query_hidden).view(1, size, -1, attn.head_dim))
            q = q.transpose(1, 2)
            if added:
                k = attn.k_norm(attn.k_proj(context).view(1, added, -1, attn.head_dim))
                v = attn.v_proj(context).view(1, added, -1, attn.head_dim)
                self.keys[i] = (
                    k if self.keys[i] is None else torch.cat((self.keys[i], k), 1)
                )
                self.values[i] = (
                    v if self.values[i] is None else torch.cat((self.values[i], v), 1)
                )
            k_noise = attn.k_norm(
                attn.k_proj(query_hidden).view(1, size, -1, attn.head_dim)
            )
            v_noise = attn.v_proj(query_hidden).view(1, size, -1, attn.head_dim)
            k = torch.cat((self.keys[i], k_noise), 1).transpose(1, 2)
            v = torch.cat((self.values[i], v_noise), 1).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *rotary)
            attended, _ = draft_attention(
                attn,
                q,
                k,
                v,
                mask,
                dropout=0.0,
                scaling=attn.scaling,
                sliding_window=attn.sliding_window,
            )
            hidden = residual + attn.o_proj(attended.reshape(1, size, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        self.length = length
        self.projected_context_tokens += added
        hidden = draft.norm(hidden)[:, 1:]
        logits = draft.lm_head(hidden)
        return DraftBlock(hidden, logits, logits.detach().argmax(-1))
