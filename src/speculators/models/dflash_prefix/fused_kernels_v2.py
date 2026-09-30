"""Small inference-only vector kernels for the audited local_prefix_v2 head.

No GEMM, target verification, candidate extraction, or model weight changes.
Triton-Ascend is required on NPU. CPU execution is allowed ONLY under Triton's
explicit interpreter for tests; it is not an inference fallback.
"""
import math
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _choose_commit(
    Unary, Local, Hist, Context, Prev, Successors, Ids, Preds, Choices,
    Storage, NextPrev, Selected, Scores, Forced,
    # Inputs below are already slices at this proposal, batch=1. Explicit
    # strides preserve strided history scores, IDs, and projection views.
    US: tl.constexpr, LS: tl.constexpr, HS: tl.constexpr, CS: tl.constexpr,
    PS: tl.constexpr, SK: tl.constexpr, SR: tl.constexpr, IS: tl.constexpr,
    PK: tl.constexpr, PR: tl.constexpr, CK: tl.constexpr, CR: tl.constexpr,
    R: tl.constexpr, KV: tl.constexpr, K: tl.constexpr,
    BR: tl.constexpr, BKV: tl.constexpr, BK: tl.constexpr,
    SCALE: tl.constexpr, INV_SQRT_R: tl.constexpr,
    FUSE_LOCAL: tl.constexpr, HAS_HISTORY: tl.constexpr,
    WRITE_NEXT: tl.constexpr, WRITE_KV: tl.constexpr,
    WRITE_SCORES: tl.constexpr, USE_FORCED: tl.constexpr,
):
    c = tl.arange(0, BK)
    if FUSE_LOCAL:
        r = tl.arange(0, BR)
        ctx = tl.load(Context + r * CS, r < R, 0).to(tl.float32)
        previous = tl.load(Prev + r * PS, r < R, 0).to(tl.float32)
        succ = tl.load(Successors + c[:, None] * SK + r[None, :] * SR,
                       (c[:, None] < K) & (r[None, :] < R), 0).to(tl.float32)
        # Preserve the parenthesization of (context * previous) * successor.
        # A reduction tree can differ from torch.sum: this variant is gated
        # by exact-token checks and kept separate from the select-only variant.
        local = tl.sum((ctx * previous)[None, :] * succ, 1) * INV_SQRT_R
    else:
        local = tl.load(Local + c * LS, c < K, 0).to(tl.float32)
    history = tl.full((BK,), 0, tl.float32)
    if HAS_HISTORY:
        history = tl.load(Hist + c * HS, c < K, 0).to(tl.float32)
    unary = tl.load(Unary + c * US, c < K, 0).to(tl.float32)
    score = unary + SCALE * (local + history)
    score = tl.where(c < K, score, -float("inf"))
    best = tl.max(score, 0)
    # Match torch.argmax's first candidate on ties. Token ID order is NOT
    # candidate order. Do not use an unordered reduction of token IDs.
    index = tl.min(tl.where((c < K) & (score == best), c, 2147483647), 0)
    # Non-finite input is rejected by validation. Clamp protects indirect
    # loads even if an unexpected NaN occurs before that check can report it.
    index = tl.minimum(index, K - 1)
    if USE_FORCED:
        index = tl.load(Forced).to(tl.int32)
    chosen_id = tl.load(Ids + index * IS)
    tl.store(Selected, chosen_id)
    if WRITE_SCORES:
        tl.store(Scores + c, score, c < K)
    if WRITE_NEXT:
        r = tl.arange(0, BR)
        next_prev = tl.load(Preds + index * PK + r * PR, r < R, 0)
        tl.store(NextPrev + r, next_prev, r < R)
    if WRITE_KV:
        j = tl.arange(0, BKV)
        kv = tl.load(Choices + index * CK + j * CR, j < KV, 0)
        tl.store(Storage + j, kv, j < KV)


@triton.jit
def _attention_softmax(
    Logits, Bias, Mask, Output,
    Q: tl.constexpr, L: tl.constexpr,
    XH: tl.constexpr, XQ: tl.constexpr, XL: tl.constexpr,
    BH: tl.constexpr, BQ: tl.constexpr, BL: tl.constexpr,
    MQ: tl.constexpr, ML: tl.constexpr,
    INV_SQRT_D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = row // Q
    q = row % Q
    c = tl.arange(0, BLOCK)
    x = tl.load(Logits + h * XH + q * XQ + c * XL, c < L, 0).to(tl.float32)
    bias = tl.load(Bias + h * BH + q * BQ + c * BL, c < L, 0).to(tl.float32)
    masked = tl.load(Mask + q * MQ + c * ML, c < L, 1)
    z = tl.where((c < L) & ~masked, x * INV_SQRT_D + bias, -float("inf"))
    e = tl.exp(z - tl.max(z, 0))
    weight = e / tl.sum(e, 0)
    tl.store(Output + row * L + c, weight, c < L)


def _device(tensors):
    device = tensors[0].device
    if any(x.device != device for x in tensors):
        raise ValueError("Fused selector tensors must share one device")
    interpreting = os.environ.get("TRITON_INTERPRET") == "1"
    if interpreting:
        if device.type != "cpu":
            raise RuntimeError("TRITON_INTERPRET must be unset for NPU serving")
    elif device.type not in {"npu", "privateuseone"}:
        raise ValueError("Fused selector requires Ascend NPU; no CPU/CUDA fallback")


def choose_commit(*, unary, local, history, context, previous, successors,
                  ids, predecessors, choices, storage, next_previous,
                  selected, scores, correction_scale, fuse_local,
                  forced=None):
    """One batch=1 proposal. Outputs are unique, preallocated contiguous slices."""
    k, rank = successors.shape
    if k != 16 or rank not in (64, 256):
        raise ValueError("Fused selection currently supports K=16 and rank=64/256")
    inputs = [unary, context, previous, successors, ids, predecessors,
              next_previous, selected]
    inputs += [x for x in (local, history, choices, storage, scores, forced) if x is not None]
    _device(inputs)
    if unary.shape != (k,) or ids.shape != (k,) or predecessors.shape != (k, rank):
        raise ValueError("Invalid fused candidate shapes")
    if context.shape != (rank,) or previous.shape != (rank,):
        raise ValueError("Invalid fused local context shapes")
    for x in (context, previous, successors, predecessors):
        if x.dtype != torch.float32:
            raise ValueError("Fused local scoring/gathers require FP32 prepared projections")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Candidate IDs must be int32/int64")
    if not fuse_local and (local is None or local.shape != (k,) or local.dtype != torch.float32):
        raise ValueError("select mode needs original FP32 local scores")
    if history is not None and (history.shape != (k,) or history.dtype != torch.float32):
        raise ValueError("History scores must be FP32 [K]")
    write_next = next_previous.numel() != 0
    if write_next and (next_previous.shape != (rank,) or not next_previous.is_contiguous()
                       or next_previous.dtype != torch.float32):
        raise ValueError("Invalid next predecessor output")
    kv = 1
    if choices is not None or storage is not None:
        if choices is None or storage is None or choices.ndim != 2 or choices.shape[0] != k:
            raise ValueError("K/V choice table and destination must be supplied together")
        kv = choices.shape[1]
        if (storage.shape != (kv,) or not storage.is_contiguous()
                or storage.dtype != torch.float32 or choices.dtype != torch.float32):
            raise ValueError("Invalid K/V destination/precision")
    if scores is not None and (scores.shape != (k,) or not scores.is_contiguous()
                               or scores.dtype != torch.float32):
        raise ValueError("Invalid score output")
    if selected.numel() != 1 or selected.dtype != ids.dtype:
        raise ValueError("Invalid selected token output")
    if forced is not None and (forced.numel() != 1 or forced.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Invalid forced index; callers must supply an in-range value")
    if not math.isfinite(correction_scale) or not 0 <= correction_scale <= 1:
        raise ValueError("Invalid correction scale")
    # Unused pointer arguments still receive valid tensors, but their branches
    # are eliminated at compile time. No null/dummy pointer dereferences.
    dummy = unary
    _choose_commit[(1,)](
        unary, local if local is not None else dummy,
        history if history is not None else dummy,
        context, previous, successors, ids, predecessors,
        choices if choices is not None else dummy,
        storage if storage is not None else dummy,
        next_previous if write_next else dummy, selected, scores if scores is not None else dummy,
        forced if forced is not None else ids,
        unary.stride(0), local.stride(0) if local is not None else 1,
        history.stride(0) if history is not None else 1,
        context.stride(0), previous.stride(0), *successors.stride(), ids.stride(0),
        *predecessors.stride(), *(choices.stride() if choices is not None else (1, 1)),
        rank, kv, k, triton.next_power_of_2(rank), triton.next_power_of_2(kv),
        triton.next_power_of_2(k), correction_scale, 1 / math.sqrt(rank),
        fuse_local, history is not None, write_next, storage is not None,
        scores is not None, forced is not None,
        enable_fp_fusion=False,
    )


def attention_softmax(logits, bias, mask, head_dim):
    """Fuse only scaling/bias/mask/softmax; keep both original FP32 matmuls."""
    _device([logits, bias, mask])
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.dtype != torch.float32:
        raise ValueError("Fused attention epilogue requires FP32 batch=1 logits")
    _, heads, queries, length = logits.shape
    if (bias.shape != logits.shape or bias.dtype != torch.float32
            or mask.shape != (1, 1, queries, length) or mask.dtype != torch.bool
            or length > 16 or queries not in (16, 32) or head_dim not in (16, 64)):
        raise ValueError("Unsupported fused attention shape/precision")
    output = torch.empty_like(logits, memory_format=torch.contiguous_format)
    _attention_softmax[(heads * queries,)](
        logits, bias, mask, output, queries, length,
        *logits.stride()[1:], *bias.stride()[1:], *mask.stride()[2:],
        1 / math.sqrt(head_dim), triton.next_power_of_2(length),
        enable_fp_fusion=False,
    )
    return output
