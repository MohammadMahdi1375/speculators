"""Explicit execution backends for ReTrace; the training objective is unchanged.

Attention uses the same Transformers SDPA dispatcher as stock DFlash. Its NPU
adapter converts our zero/-infinity masks to the boolean masks required by
FlashAttentionScore. No target, draft, or packed-block causal mask is removed.
"""

import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3RMSNorm,
    eager_attention_forward,
)


def draft_attention(module, query, key, value, mask, **kwargs):
    implementation = module.config._attn_implementation or "eager"
    if implementation not in ("eager", "sdpa"):
        raise ValueError(f"Unsupported ReTrace attention backend: {implementation}")
    function = (
        eager_attention_forward
        if implementation == "eager"
        else ALL_ATTENTION_FUNCTIONS["sdpa"]
    )
    return function(module, query, key, value, mask, **kwargs)


def set_draft_attention(draft, implementation):
    if implementation not in ("eager", "sdpa"):
        raise ValueError("Use eager or sdpa attention")
    draft.config.transformer_layer_config._attn_implementation = implementation
    for layer in draft.layers:
        layer.self_attn.config._attn_implementation = implementation


class FrozenNPURMSNorm(nn.Module):
    """Fused target-only norm; the original frozen weight keeps its state key."""

    def __init__(self, original):
        super().__init__()
        if original.weight.requires_grad or original.weight.device.type != "npu":
            raise ValueError("Fused target RMSNorm requires a frozen NPU weight")
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon

    def forward(self, hidden_states):
        if hidden_states.requires_grad or self.weight.requires_grad:
            raise RuntimeError("Target-only fused norm cannot receive gradients")
        if hidden_states.dtype != self.weight.dtype:
            raise ValueError("Target RMSNorm input/weight dtype mismatch")
        import torch_npu

        return torch_npu.npu_rms_norm(
            hidden_states, self.weight, epsilon=self.variance_epsilon
        )[0]


def fuse_target_norms(model):
    """Opt in only; compare representative target norm shapes before replacing."""
    if model.device.type != "npu" or any(p.requires_grad for p in model.parameters()):
        raise ValueError("--target-norm npu requires a frozen target on an NPU")
    replacements = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, Qwen3RMSNorm):
                replacements.append((parent, name, child))
    if not replacements:
        raise ValueError("No inspected Qwen3 target RMSNorm modules found")
    generator = torch.Generator(device="cpu").manual_seed(42)
    checked = set()
    with torch.no_grad():
        for parent, name, original in replacements:
            fused = FrozenNPURMSNorm(original)
            width = original.weight.numel()
            if width not in checked:
                x = torch.randn((2, 3, width), generator=generator).to(
                    device=model.device, dtype=model.dtype
                )
                expected, actual = original(x), fused(x)
                tolerance = 0.035 if model.dtype == torch.bfloat16 else 1e-5
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                checked.add(width)
            setattr(parent, name, fused)
    return len(replacements)


class PromptQueue:
    """Atomic job claims within ONE optimizer update, with fixed model weights.

    Every rank already holds the same shuffled global prompt batch. The store
    only assigns its positions. It never chooses tokens, drops prompts, looks
    ahead to another update, or changes prompt-gradient normalization.
    """

    def __init__(self, store):
        self.store = store

    def groups(self, batch, step, microbatch):
        if microbatch < 1:
            raise ValueError("Microbatch size must be positive")
        while True:
            end = self.store.add(f"update_{step}", microbatch)
            start = end - microbatch
            if start >= len(batch):
                return
            positions = list(range(start, min(end, len(batch))))
            yield positions, [batch[i] for i in positions]


def make_prompt_queue(output, rank, world, device):
    if world == 1:
        return PromptQueue(dist.HashStore())
    # A fresh name avoids stale counters after resuming a saved checkpoint.
    # FileStore uses CPU file locking and does not consume an accelerator for
    # per-prompt scheduling. This launcher supports one shared-filesystem node.
    names = [f"retrace_queue_{uuid.uuid4().hex}.store" if rank == 0 else None]
    dist.broadcast_object_list(names, src=0, device=device)
    store = dist.FileStore(str(Path(output) / names[0]), world)
    store.set_timeout(timedelta(hours=2))
    return PromptQueue(store)


def static_prompt_groups(batch, rank, world, microbatch):
    positions = list(range(rank, len(batch), world))
    for start in range(0, len(positions), microbatch):
        group = positions[start : start + microbatch]
        yield group, [batch[i] for i in group]


def validate_prompt_coverage(positions, size, device, distributed):
    coverage = torch.zeros(size, device=device, dtype=torch.int32)
    for position in positions:
        coverage[position] += 1
    if distributed:
        dist.all_reduce(coverage)
    if not torch.all(coverage == 1):
        raise RuntimeError(
            "Prompt scheduler did not process every batch position exactly once"
        )
