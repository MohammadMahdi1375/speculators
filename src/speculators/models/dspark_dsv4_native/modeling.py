"""BF16/PyTorch training implementation of the native DeepSeek-V4 DSpark drafter.

The official DeepSeek inference code uses FP4/FP8 and custom sparse-attention /
MegaMoE kernels.  This module keeps the same module structure and parameter
names for the drafter-side ``mtp.*`` weights, but uses ordinary PyTorch ops so it
can be trained inside the existing Speculators trainer on Ascend/NPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import DeepSeekV4NativeDSparkConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.weight.float()).to(dtype)


class NativeLinear(nn.Linear):
    """Small alias so checkpoint names resemble DeepSeek's ``Linear`` modules."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)


class NativeParallelHead(nn.Linear):
    """Full-vocab training equivalent of DeepSeek's tensor-parallel head."""

    def __init__(self, vocab_size: int, dim: int, bias: bool = False, dtype=torch.float32) -> None:
        super().__init__(dim, vocab_size, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor, full_logits: bool = True) -> torch.Tensor:  # noqa: ARG002
        return F.linear(x.float(), self.weight.float(), self.bias)


def _apply_rotary_last(x: torch.Tensor, position_ids: torch.Tensor, rope_theta: float) -> torch.Tensor:
    """Apply RoPE to a tensor whose final dimension is the rotary dimension.

    ``x`` shape may be ``[B, S, D]`` or ``[B, S, H, D]``. ``position_ids`` is
    ``[B, S]``.  This implementation is intentionally simple and dtype-safe.
    """

    if x.numel() == 0:
        return x
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"rotary dimension must be even, got {d}")
    device = x.device
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    angles = position_ids.float().unsqueeze(-1) * inv_freq  # [B,S,D/2]
    cos = angles.cos()
    sin = angles.sin()
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    x_float = x.float()
    x1 = x_float[..., 0::2]
    x2 = x_float[..., 1::2]
    out = torch.empty_like(x_float)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.to(x.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training-friendly approximation of DeepSeek's hc_split_sinkhorn kernel."""

    m = hc_mult
    pre_logits = mixes[..., :m] * scale[0] + base[:m]
    post_logits = mixes[..., m : 2 * m] * scale[1] + base[m : 2 * m]
    comb_logits = mixes[..., 2 * m :].view(*mixes.shape[:-1], m, m)
    comb_logits = comb_logits * scale[2] + base[2 * m :].view(m, m)

    pre = torch.softmax(pre_logits, dim=-1)
    post = torch.sigmoid(post_logits) + eps
    comb = torch.exp(comb_logits - comb_logits.amax(dim=(-1, -2), keepdim=True))
    for _ in range(iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class NativeDeepSeekV4Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 10.0) -> None:
        super().__init__()
        self.w1 = NativeLinear(dim, inter_dim, bias=False)
        self.w2 = NativeLinear(inter_dim, dim, bias=False)
        self.w3 = NativeLinear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        out = F.silu(gate) * up
        if weights is not None:
            out = out * weights.float()
        return self.w2(out.to(dtype))


class NativeDeepSeekV4Gate(nn.Module):
    def __init__(self, layer_id: int, cfg: DeepSeekV4NativeDSparkConfig) -> None:
        super().__init__()
        self.dim = cfg.dim
        self.topk = cfg.n_activated_experts
        self.score_func = cfg.score_func
        self.route_scale = cfg.route_scale
        self.hash = layer_id < cfg.n_hash_layers
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(cfg.vocab_size, cfg.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(cfg.n_routed_experts, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            if input_ids is None:
                raise ValueError("hash-routed Gate requires input_ids")
            indices = self.tid2eid[input_ids].long()
        else:
            indices = scores.topk(self.topk, dim=-1).indices
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        weights = weights * self.route_scale
        return weights, indices


class NativeDeepSeekV4MoE(nn.Module):
    def __init__(self, layer_id: int, cfg: DeepSeekV4NativeDSparkConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.dim = cfg.dim
        self.n_routed_experts = cfg.n_routed_experts
        self.n_activated_experts = cfg.n_activated_experts
        self.gate = NativeDeepSeekV4Gate(layer_id, cfg)
        self.experts = nn.ModuleList(
            [
                NativeDeepSeekV4Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit)
                for _ in range(cfg.n_routed_experts)
            ]
        )
        if cfg.n_shared_experts != 1:
            raise ValueError("DeepSeek-V4 DSpark expects exactly one shared expert")
        self.shared_experts = NativeDeepSeekV4Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        ids_flat = input_ids.reshape(-1)
        weights, indices = self.gate(x_flat, ids_flat)
        y = torch.zeros_like(x_flat, dtype=torch.float32)
        counts = torch.bincount(indices.reshape(-1), minlength=self.n_routed_experts)
        for expert_id, expert in enumerate(self.experts):
            if int(counts[expert_id].item()) == 0:
                continue
            token_idx, which_top = torch.where(indices == expert_id)
            expert_out = expert(x_flat[token_idx], weights[token_idx, which_top, None])
            y.index_add_(0, token_idx, expert_out.float())
        y = y + self.shared_experts(x_flat).float()

        # FSDP/HCCL consistency for sparse MoE:
        #
        # Different ranks route tokens to different expert subsets. If an expert
        # receives zero tokens on one rank, its parameters may not participate in
        # backward on that rank, while they do participate on another rank. With
        # FSDP this causes inconsistent HCCL ReduceScatter counts.
        #
        # Add a zero-valued autograd edge to every routed expert parameter so all
        # ranks produce gradients for the same parameter set. This does not
        # change the forward output.
        if self.training and os.environ.get("DSPARK_NATIVE_TOUCH_ALL_EXPERTS", "1") != "0":
            touch = None
            for expert in self.experts:
                for param in expert.parameters():
                    term = param.reshape(-1)[:1].float().sum()
                    touch = term if touch is None else touch + term

            if touch is not None:
                y = y + touch.to(y.dtype) * 0.0

        return y.to(x.dtype).view(shape)


class NativeDSparkAttention(nn.Module):
    """Uncompressed sliding-window DSpark attention.

    This mirrors the official DSparkAttention module shape: q LoRA projection,
    shared MQA-style KV projection, per-head attention sink parameter, and
    grouped output LoRA projection.  For training we use dense attention over the
    gathered target context window plus the current draft block.
    """

    def __init__(self, layer_id: int, cfg: DeepSeekV4NativeDSparkConfig) -> None:  # noqa: ARG002
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.o_groups
        self.window_size = cfg.window_size
        self.eps = cfg.norm_eps
        self.rope_theta = cfg.rope_theta
        self.softmax_scale = self.head_dim**-0.5

        if self.n_heads % self.n_groups != 0:
            raise ValueError("n_heads must be divisible by o_groups")

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = NativeLinear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = NativeLinear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = NativeLinear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = NativeLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.wo_b = NativeLinear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        main_x: torch.Tensor,
        main_position_ids: Optional[torch.Tensor] = None,
        draft_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, block_size, _ = x.shape
        ctx_len = main_x.shape[1]
        device = x.device
        if main_position_ids is None:
            main_position_ids = torch.arange(ctx_len, device=device).unsqueeze(0).expand(bsz, ctx_len)
        if draft_position_ids is None:
            draft_position_ids = (
                torch.arange(ctx_len, ctx_len + block_size, device=device)
                .unsqueeze(0)
                .expand(bsz, block_size)
            )

        # Query.
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).view(bsz, block_size, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(q.dtype)
        q_rope = _apply_rotary_last(q[..., -self.rope_head_dim :], draft_position_ids, self.rope_theta)
        q = torch.cat([q[..., : -self.rope_head_dim], q_rope], dim=-1)

        # Shared KV for target context plus draft block.
        main_kv = self.kv_norm(self.wkv(main_x))
        main_rope = _apply_rotary_last(
            main_kv[..., -self.rope_head_dim :], main_position_ids, self.rope_theta
        )
        main_kv = torch.cat([main_kv[..., : -self.rope_head_dim], main_rope], dim=-1)

        draft_kv = self.kv_norm(self.wkv(x))
        draft_rope = _apply_rotary_last(
            draft_kv[..., -self.rope_head_dim :], draft_position_ids, self.rope_theta
        )
        draft_kv = torch.cat([draft_kv[..., : -self.rope_head_dim], draft_rope], dim=-1)
        kv = torch.cat([main_kv, draft_kv], dim=1)  # [B, W+K, D]

        attn = torch.einsum("bqhd,bkd->bhqk", q.float(), kv.float()) * self.softmax_scale
        # Keep attn_sink as a learnable parameter with official naming.  It is a
        # sink score; the associated value is zero, so it only regularizes mass.
        sink = self.attn_sink.view(1, self.n_heads, 1, 1).expand(bsz, -1, block_size, -1)
        attn_with_sink = torch.cat([sink, attn], dim=-1)
        probs = torch.softmax(attn_with_sink, dim=-1)[..., 1:]
        o = torch.einsum("bhqk,bkd->bqhd", probs.to(kv.dtype), kv)

        o = o.reshape(bsz, block_size, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a.to(o.dtype))
        return self.wo_b(o.flatten(2))


class NativeDSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = NativeParallelHead(vocab_size, rank, bias=False, dtype=torch.float32)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.markov_w1(token_ids)
        logits = self.markov_w2(embed, full_logits=True)
        return logits, embed


class NativeDSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = NativeLinear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed.to(hidden.dtype)], dim=-1)
        return self.proj(x.float()).squeeze(-1)


class NativeDSparkBlock(nn.Module):
    attention_cls = NativeDSparkAttention

    def __init__(self, layer_id: int, cfg: DeepSeekV4NativeDSparkConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.stage_id = layer_id - cfg.n_layers
        self.dim = cfg.dim
        self.block_size = cfg.dspark_block_size
        self.noise_token_id = cfg.dspark_noise_token_id
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.norm_eps

        self.attn = self.attention_cls(layer_id, cfg)
        self.ffn = NativeDeepSeekV4MoE(layer_id, cfg)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * cfg.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        if self.stage_id == 0:
            self.main_proj = NativeLinear(cfg.dim * len(cfg.dspark_target_layer_ids), cfg.dim, bias=False)
            self.main_norm = RMSNorm(cfg.dim, cfg.norm_eps)

        if self.stage_id == cfg.n_mtp_layers - 1:
            self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
            self.markov_head = NativeDSparkMarkovHead(cfg.vocab_size, cfg.dspark_markov_rank)
            self.confidence_head = NativeDSparkConfidenceHead(cfg.dim + cfg.dspark_markov_rank)
            self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
            self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def hc_pre(
        self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype)

    def forward_embed(
        self, main_hidden: torch.Tensor, input_ids: torch.Tensor, embed: nn.Embedding
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full(
            (input_ids.size(0), self.block_size), self.noise_token_id
        )
        draft_input_ids[:, 0] = input_ids
        x = embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x, draft_input_ids

    def forward(
        self,
        x: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        main_x: torch.Tensor | None = None,
        main_position_ids: torch.Tensor | None = None,
        draft_position_ids: torch.Tensor | None = None,
        *,
        main_hidden: torch.Tensor | None = None,
        anchor_ids: torch.Tensor | None = None,
        embed: nn.Embedding | None = None,
        return_embed: bool = False,
        prev_token_ids: torch.Tensor | None = None,
        head: nn.Linear | None = None,
        return_head: bool = False,
    ):
        """Run DSpark block through normal forward() so FSDP hooks fire.

        FSDP/DTensor hooks are attached to module.__call__/forward. Calling
        custom methods like forward_embed() or forward_head_teacher() directly
        can leave weights as DTensors and activations as normal torch.Tensor,
        causing mixed Tensor/DTensor matmul errors.
        """

        if return_embed:
            if main_hidden is None or anchor_ids is None or embed is None:
                raise ValueError(
                    "return_embed=True requires main_hidden, anchor_ids, and embed"
                )
            x, main_x, input_ids = self.forward_embed(main_hidden, anchor_ids, embed)

        if x is None or input_ids is None or main_x is None:
            raise ValueError("x, input_ids, and main_x must be provided")
        if main_position_ids is None or draft_position_ids is None:
            raise ValueError("main_position_ids and draft_position_ids must be provided")

        residual = x
        y, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        y = self.attn_norm(y)
        y = self.attn(y, main_x, main_position_ids, draft_position_ids)
        x = self.hc_post(y, residual, post, comb)

        residual = x
        y, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        x = self.hc_post(y, residual, post, comb)

        if return_head:
            if anchor_ids is None or prev_token_ids is None or head is None:
                raise ValueError(
                    "return_head=True requires anchor_ids, prev_token_ids, and head"
                )
            logits, confidence_logits, hidden = self.forward_head_teacher(
                x, anchor_ids, prev_token_ids, head
            )
            if return_embed:
                return x, main_x, input_ids, logits, confidence_logits, hidden
            return x, logits, confidence_logits, hidden

        if return_embed:
            return x, main_x, input_ids

        return x

    def forward_head_teacher(
        self,
        x: torch.Tensor,
        anchor_ids: torch.Tensor,
        prev_token_ids: torch.Tensor,
        head: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return teacher-forced DSpark logits and confidence.

        ``prev_token_ids[:, k]`` is the token before the token predicted at
        draft position ``k``.  This is the training counterpart of official
        ``forward_head()``, which samples sequentially from its own outputs.
        """

        del anchor_ids  # kept in the signature to mirror official forward_head
        h = self.hc_head(x)
        h = self.norm(h)
        base_logits = head(h, full_logits=True) if isinstance(head, NativeParallelHead) else head(h.float())
        markov_logits, markov_embed = self.markov_head(prev_token_ids)
        logits = base_logits.float() + markov_logits.float()
        confidence = self.confidence_head(h, markov_embed)
        return logits, confidence, h


@dataclass
class GatheredAnchors:
    anchor_ids: torch.Tensor
    prev_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    target_logits: torch.Tensor
    main_hidden_context: torch.Tensor
    main_position_ids: torch.Tensor
    draft_position_ids: torch.Tensor
    valid_mask: torch.Tensor
