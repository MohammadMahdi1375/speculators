#!/usr/bin/env python
"""Run target, original DSpARK, or isolated ReTrace in separate NPU processes."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

from speculators.models.retrace.data import read_prompts
from speculators.models.retrace.serving import speculative_config


def metrics(llm):
    return [
        dataclasses.asdict(x) if dataclasses.is_dataclass(x) else repr(x)
        for x in llm.get_metrics()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--compare", help="A report produced on the identical prompts and settings"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=2048)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        parser.error("Use a new output file")
    if min(args.concurrency, args.max_tokens, args.max_prompts, args.prompt_length) < 1:
        parser.error("Counts and lengths must be positive")
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    ids, data = read_prompts(
        args.data, len(tokenizer), args.max_prompts, args.prompt_length, "validation"
    )
    prompts = [{"prompt_token_ids": p} for p in ids]
    kwargs = dict(
        model=args.target,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        max_num_seqs=args.concurrency,
        max_model_len=args.prompt_length + args.max_tokens + 64,
        max_num_batched_tokens=args.prompt_length + args.max_tokens + 64,
        gpu_memory_utilization=0.85,
        disable_log_stats=False,
        additional_config={"enable_reduce_sample": False},
    )
    if args.draft:
        kwargs["speculative_config"] = speculative_config(args.draft)
    baseline = json.loads(Path(args.compare).read_text()) if args.compare else None
    if baseline:
        for key in ("target", "concurrency", "max_tokens", "prompt_length"):
            if baseline["args"][key] != vars(args)[key]:
                parser.error(f"Comparison setting differs: {key}")
        if [r["prompt_token_ids"] for r in baseline["results"]] != ids:
            parser.error("Comparison prompt sequences differ")
        old = baseline["engine"].get("speculative_config")
        new = kwargs.get("speculative_config")
        if (
            old
            and new
            and old["num_speculative_tokens"] != new["num_speculative_tokens"]
        ):
            parser.error("Draft comparison requires equal proposal counts K")
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torch-npu",
            "transformers",
            "speculators",
            "vllm",
            "vllm-ascend",
        )
    }
    configs = {
        name: hashlib.sha256((Path(path) / "config.json").read_bytes()).hexdigest()
        for name, path in (("target", args.target), ("draft", args.draft))
        if path
    }
    llm = LLM(**kwargs)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=42)
    llm.generate(prompts[: args.concurrency], sampling, use_tqdm=False)
    before = metrics(llm)
    start = time.perf_counter()
    outputs = []
    for offset in range(0, len(prompts), args.concurrency):
        outputs.extend(
            llm.generate(
                prompts[offset : offset + args.concurrency], sampling, use_tqdm=False
            )
        )
    elapsed = time.perf_counter() - start
    rows = [
        {
            "prompt_token_ids": x.prompt_token_ids,
            "token_ids": list(x.outputs[0].token_ids),
            "finish_reason": x.outputs[0].finish_reason,
        }
        for x in outputs
    ]
    result = {
        "args": vars(args),
        "engine": kwargs,
        "data": data,
        "versions": versions,
        "config_sha256": configs,
        "visible_npus": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "wall_seconds": elapsed,
        "tokens_per_second": sum(len(r["token_ids"]) for r in rows) / elapsed,
        "metrics_before": before,
        "metrics_after": metrics(llm),
        "results": rows,
        "npu_info": subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True
        ).stdout,
    }
    if baseline:
        result["mismatching_requests"] = [
            i
            for i, (a, b) in enumerate(zip(rows, baseline["results"]))
            if a["token_ids"] != b["token_ids"]
        ]
        result["greedy_parity"] = not result["mismatching_requests"]
        if result["greedy_parity"]:
            result["throughput_ratio_to_comparison"] = (
                result["tokens_per_second"] / baseline["tokens_per_second"]
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in {
                    "wall_seconds",
                    "tokens_per_second",
                    "greedy_parity",
                    "mismatching_requests",
                    "throughput_ratio_to_comparison",
                }
            },
            indent=2,
        )
    )
    if result.get("mismatching_requests"):
        raise SystemExit(
            "Greedy parity failed; report saved. Do not use the throughput ratio."
        )


if __name__ == "__main__":
    main()
