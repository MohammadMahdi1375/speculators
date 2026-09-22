"""Greedy token parity and measured acceptance on held-out prompts."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from .data import read_prompts
from .reference import TokenwiseLocalTarget
from .rollout import generate, target_greedy
from .runtime import add_model_args, device_for, setup, sync_device


def first_difference(expected, actual):
    for index in range(max(len(expected), len(actual))):
        a = expected[index] if index < len(expected) else None
        b = actual[index] if index < len(actual) else None
        if a != b:
            return {"generated_index": index, "expected_token": a, "actual_token": b}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--response-length", type=int, default=256)
    parser.add_argument("--disable-conditioning", action="store_true")
    parser.add_argument(
        "--target-execution",
        choices=("batched", "tokenwise"),
        default="batched",
        help="tokenwise is a slow numerical correctness control, not a serving optimization",
    )
    args = parser.parse_args()
    if not args.draft:
        parser.error("Evaluation requires --draft CHECKPOINT")
    if args.vllm_endpoint:
        parser.error(
            "Reference timing uses the local HF target; use the native vLLM benchmark for serving performance"
        )
    if args.response_length <= 0 or args.max_prompts <= 0:
        parser.error("Evaluation lengths and counts must be positive")
    dest = Path(args.output)
    if dest.exists():
        parser.error("Use a new report file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    prompts, data = read_prompts(
        args.data, len(tokenizer), args.max_prompts, args.prompt_length, "validation"
    )
    draft, target, eos = setup(args, device)
    draft.config.retrace_enabled = not args.disable_conditioning
    draft.eval()
    verification_target = (
        TokenwiseLocalTarget(target.model, draft.target_layer_ids)
        if args.target_execution == "tokenwise"
        else target
    )
    generate(verification_target, draft, prompts[0], min(16, args.response_length), eos)
    versions = {}
    for name in ("torch", "torch-npu", "transformers", "speculators"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    provenance = {
        "argv": sys.argv,
        "versions": versions,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    print(
        f"Target execution: {args.target_execution}; native serving is evaluated separately",
        flush=True,
    )
    rows = []

    def write_report(parity):
        events = [event for row in rows for event in row["rounds"]]
        summary = {
            "parity": parity,
            "requests": len(rows),
            "requested_requests": len(prompts),
            "rounds": len(events),
            "accepted_per_round": sum(e["accepted"] for e in events)
            / max(1, len(events)),
            "target_execution": args.target_execution,
            "native_serving_validated": False,
            "runtime": (
                "HF tokenwise correctness control; no throughput claim"
                if args.target_execution == "tokenwise"
                else "HF eager reference; not native serving throughput"
            ),
        }
        if args.target_execution == "batched" and parity:
            summary["tokens_per_second"] = sum(len(r["tokens"]) for r in rows) / sum(
                r["seconds"] for r in rows
            )
        dest.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "data": data,
                    "summary": summary,
                    "provenance": provenance,
                    "results": rows,
                },
                indent=2,
            )
            + "\n"
        )
        return summary

    for index, prompt in enumerate(prompts):
        reference_calls = target.calls
        expected = target_greedy(target, draft, prompt, args.response_length, eos)
        reference_calls = target.calls - reference_calls
        target.reset()
        sync_device(device)
        start = time.perf_counter()
        verification_calls = verification_target.calls
        actual, rounds = generate(
            verification_target, draft, prompt, args.response_length, eos
        )
        verification_calls = verification_target.calls - verification_calls
        sync_device(device)
        elapsed = time.perf_counter() - start
        row = {
            "row": data["rows"][index],
            "parity": actual == expected,
            "tokens": actual,
            "expected_tokens": expected,
            "first_difference": first_difference(expected, actual),
            "reference_forward_calls": reference_calls,
            "verification_forward_calls": verification_calls,
            "seconds": elapsed,
            "rounds": rounds,
        }
        rows.append(row)
        print(f"{index + 1}/{len(prompts)} parity={row['parity']}", flush=True)
        if actual != expected:
            write_report(False)
            print(json.dumps(row["first_difference"]), flush=True)
            raise RuntimeError("Greedy parity failed; mismatch written to output")
    summary = write_report(True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
