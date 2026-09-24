#!/usr/bin/env python3
"""Measure post-logit selection overhead; this is not an end-to-end speed benchmark."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from speculators.models.dflash_prefix.model_definitions import PrefixAttentionSelector


def measure(fn, synchronize, warmup, repeats):
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--top-k", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument(
        "--backend", choices=("torch", "triton", "both"), default="torch"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--baseline-round-ms", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.steps,
            args.rank,
            args.vocab_size,
            args.hidden_size,
            args.repeats,
        )
        <= 0
    ):
        raise ValueError("Shapes and repeats must be positive")
    if any(k < 2 or k > args.vocab_size for k in args.top_k):
        raise ValueError("Each top-k must be in [2, vocab_size]")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.baseline_round_ms is not None and args.baseline_round_ms <= 0:
        raise ValueError("baseline-round-ms must be positive")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
        synchronize = torch.npu.synchronize
    elif device.type == "cuda":
        synchronize = torch.cuda.synchronize
    elif device.type == "cpu":
        synchronize = lambda: None
    else:
        raise ValueError("Unsupported device")
    fused = None
    if args.backend != "torch":
        from vllm_ascend.ops.triton.spec_decode.dflash_prefix import (
            greedy_select_prefix,
        )

        fused = greedy_select_prefix
    torch.manual_seed(42)
    shape = (args.batch_size, args.steps)
    hidden = torch.randn(*shape, args.hidden_size, dtype=torch.bfloat16, device=device)
    logits = torch.randn(*shape, args.vocab_size, dtype=torch.bfloat16, device=device)
    anchors = torch.randint(args.vocab_size, (args.batch_size,), device=device)
    unary_ms = measure(
        lambda: logits.argmax(-1), synchronize, args.warmup, args.repeats
    )
    results = []
    for k in args.top_k:
        head = (
            PrefixAttentionSelector(
                args.vocab_size, args.hidden_size, args.rank, k, args.steps
            )
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
        values, candidates = logits.topk(k, dim=-1)
        tables = head.prepare_tables(hidden, candidates, values, anchors)
        if fused is not None:
            torch.testing.assert_close(
                fused(tables), head.greedy_walk(tables), rtol=0, atol=0
            )
        names = ("torch", "triton") if args.backend == "both" else (args.backend,)
        for name in names:
            walk = head.greedy_walk if name == "torch" else fused

            def select(k=k, walk=walk, head=head):
                values, ids = logits.topk(k, dim=-1)
                return walk(head.prepare_tables(hidden, ids, values, anchors))

            total_ms = measure(select, synchronize, args.warmup, args.repeats)
            walk_ms = measure(
                lambda walk=walk, tables=tables: walk(tables),
                synchronize,
                args.warmup,
                args.repeats,
            )
            overhead = total_ms - unary_ms
            row = {
                "top_k": k,
                "backend": name,
                "argmax_ms": unary_ms,
                "topk_and_selector_ms": total_ms,
                "walk_ms": walk_ms,
                "extra_selection_ms": overhead,
                "pair_table_mib": 2
                * args.batch_size
                * args.steps**2
                * k**2
                * 4
                / 2**20,
            }
            if args.baseline_round_ms is not None:
                row["approx_required_acceptance_length_ratio"] = (
                    1 + overhead / args.baseline_round_ms
                )
            results.append(row)
            print(json.dumps(row))
    report = {
        "scope": "Synthetic BF16 post-logit selection, including top-k; excludes target verification and all transformer/LM-head computation",
        "device": str(device),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "rank": args.rank,
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
