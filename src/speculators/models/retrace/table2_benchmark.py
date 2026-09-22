"""Paired, concurrency-one native benchmarks using a frozen prompt manifest.

Run each method in a fresh process. T=0 mismatches remain failures. T=1 uses
the target sampler's distribution; same-seed output identity is not required.
Ascend eager measurements are not NVIDIA A800 speedup reproductions.
"""

import argparse
import csv
import dataclasses
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .pretrained import checkpoint_files, sha256
from .prompt_pool import prompt_key
from .serving import speculative_config
from .table2_data import COUNTS

METRICS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


def snapshot(llm):
    return [
        dataclasses.asdict(value)
        for value in llm.get_metrics()
        if dataclasses.is_dataclass(value)
    ]


def counters(metrics):
    return {
        key: sum(
            value["value"]
            for value in metrics
            if value["name"] == "vllm:spec_decode_" + key and "value" in value
        )
        for key in METRICS
    }


def acceptance(before, after):
    a, b = counters(before), counters(after)
    delta = {key: b[key] - a[key] for key in METRICS}
    if any(value < 0 for value in delta.values()):
        raise ValueError("Speculative counters reset inside a measured request")
    rounds = delta["num_drafts"]
    delta["accepted_per_round"] = (
        delta["num_accepted_tokens"] / rounds if rounds else None
    )
    delta["acceptance_length_with_bonus"] = (
        1 + delta["accepted_per_round"] if rounds else None
    )
    return delta


def summarize_rows(rows, speculative):
    seconds = sum(row["seconds"] for row in rows)
    tokens = sum(len(row["token_ids"]) for row in rows)
    rounds = sum(row["speculation"]["num_drafts"] for row in rows)
    accepted = sum(row["speculation"]["num_accepted_tokens"] for row in rows)
    per_request = [
        row["speculation"]["accepted_per_round"]
        for row in rows
        if row["speculation"]["num_drafts"]
    ]
    return {
        "requests": len(rows),
        "generated_tokens": tokens,
        "wall_seconds": seconds,
        "tokens_per_second": tokens / seconds if seconds else None,
        "verification_rounds": rounds,
        "accepted_per_round_pooled": accepted / rounds if rounds else None,
        "acceptance_length_with_bonus_pooled": 1 + accepted / rounds
        if rounds
        else None,
        "accepted_per_round_request_mean": statistics.mean(per_request)
        if per_request
        else None,
        "acceptance_length_with_bonus_request_mean": 1 + statistics.mean(per_request)
        if per_request
        else None,
        "speculative_metrics_available": (not speculative) or rounds > 0,
    }


def paired_summary(reference, candidate, temperature):
    left, right = reference["results"], candidate["results"]
    if [x["prompt_token_ids"] for x in left] != [x["prompt_token_ids"] for x in right]:
        raise ValueError("Benchmark prompts differ; comparison is invalid")
    mismatch = [
        i
        for i, (a, b) in enumerate(zip(left, right, strict=True))
        if a["token_ids"] != b["token_ids"]
    ]
    parity = not mismatch if temperature == 0 else None
    valid = (
        parity is not False and candidate["summary"]["speculative_metrics_available"]
    )
    return {
        "greedy_parity": parity,
        "mismatching_requests": mismatch if temperature == 0 else None,
        "stochastic_path_identity_required": False,
        "comparison_valid": valid,
        "speedup_tps": candidate["summary"]["tokens_per_second"]
        / reference["summary"]["tokens_per_second"]
        if valid
        else None,
        "speedup_total_latency": reference["summary"]["wall_seconds"]
        / candidate["summary"]["wall_seconds"]
        if valid
        else None,
        "speedup_request_tps_mean": statistics.mean(
            (len(b["token_ids"]) / b["seconds"]) / (len(a["token_ids"]) / a["seconds"])
            for a, b in zip(left, right, strict=True)
            if a["token_ids"] and b["token_ids"]
        )
        if valid
        else None,
    }


def versions():
    result = {}
    for name in (
        "torch",
        "torch-npu",
        "transformers",
        "speculators",
        "vllm",
        "vllm-ascend",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def worker(args):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch_npu  # noqa: F401
    from transformers import AutoConfig, GenerationConfig
    from vllm import LLM, SamplingParams

    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new benchmark report filename")
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["target_config_sha256"] != sha256(Path(args.target) / "config.json"):
        raise ValueError("Evaluation target configuration differs from frozen manifest")
    for name, digest in manifest["tokenizer_sha256"].items():
        if sha256(Path(args.target) / name) != digest:
            raise ValueError("Tokenizer differs from frozen manifest")
    longest = 0
    for dataset in manifest["datasets"].values():
        for row in dataset["results"]:
            if prompt_key(row["prompt_token_ids"]) != row["sha256"]:
                raise ValueError("Modified evaluation prompt tokens")
            longest = max(longest, len(row["prompt_token_ids"]))
    config = AutoConfig.from_pretrained(args.target, local_files_only=True)
    length = longest + args.max_tokens + 16
    if length > config.max_position_embeddings:
        raise ValueError(
            "Evaluation prompt + response exceeds target context; do not silently truncate benchmarks"
        )
    generation = GenerationConfig.from_pretrained(args.target, local_files_only=True)
    eos = generation.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    eos = [token for token in eos if token is not None]
    options = {
        "model": args.target,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "max_num_seqs": 1,
        "max_model_len": length,
        "max_num_batched_tokens": length,
        "gpu_memory_utilization": 0.85,
        "disable_log_stats": False,
        "generation_config": "vllm",
        "seed": 42,
        "additional_config": {"enable_reduce_sample": False},
    }
    if args.draft:
        options["speculative_config"] = speculative_config(args.draft)
        if options["speculative_config"]["num_speculative_tokens"] != 15:
            raise ValueError(
                "Table 2 recipe requires DFlash block size 16 (15 proposals + anchor)"
            )
    sampling = {
        "temperature": args.temperature,
        "top_p": 1.0 if args.temperature == 0 else 0.95,
        "top_k": 1 if args.temperature == 0 else 20,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "stop_token_ids": eos,
        "ignore_eos": False,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    report = {
        "args": vars(args),
        "engine": options,
        "sampling": sampling,
        "versions": versions(),
        "manifest_sha256": sha256(args.manifest),
        "datasets": {},
        "runtime": "Ascend vLLM eager; tensor parallel 1; concurrency 1",
        "author_protocol_exact": False,
        "complete": False,
        "checkpoint_sha256": {
            "target": {
                p.name: sha256(p)
                for p in [
                    Path(args.target) / "config.json",
                    *checkpoint_files(args.target),
                ]
            }
        },
        "visible_npus": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "source_sha256": sha256(__file__),
    }
    if args.draft:
        report["checkpoint_sha256"]["draft"] = {
            p.name: sha256(p)
            for p in [Path(args.draft) / "config.json", *checkpoint_files(args.draft)]
        }
    try:
        report["npu_info"] = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        report["npu_info"] = "unavailable"
    output.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(**options)
    reference = json.loads(Path(args.compare).read_text()) if args.compare else None
    if reference and (
        reference["manifest_sha256"] != report["manifest_sha256"]
        or reference["sampling"] != sampling
        or reference["checkpoint_sha256"]["target"]
        != report["checkpoint_sha256"]["target"]
    ):
        raise ValueError(
            "Reference dataset, target weights or sampling settings differ"
        )
    try:
        for name, dataset in manifest["datasets"].items():
            prompts = dataset["results"]
            for row in prompts[: min(3, len(prompts))]:
                llm.generate(
                    [{"prompt_token_ids": row["prompt_token_ids"]}],
                    SamplingParams(
                        **(sampling | {"max_tokens": min(128, args.max_tokens)})
                    ),
                    use_tqdm=False,
                )
            llm.reset_prefix_cache()
            records = []
            for index, row in enumerate(prompts):
                before = snapshot(llm)
                started = time.perf_counter()
                generated = llm.generate(
                    [{"prompt_token_ids": row["prompt_token_ids"]}],
                    SamplingParams(**sampling),
                    use_tqdm=False,
                )[0].outputs[0]
                seconds = time.perf_counter() - started
                after = snapshot(llm)
                records.append(
                    {
                        "source_row": row["source_row"],
                        "prompt_token_ids": row["prompt_token_ids"],
                        "token_ids": list(generated.token_ids),
                        "finish_reason": generated.finish_reason,
                        "seconds": seconds,
                        "speculation": acceptance(before, after),
                    }
                )
                print(
                    json.dumps(
                        {
                            "method": args.method,
                            "temperature": args.temperature,
                            "benchmark": name,
                            "request": index + 1,
                            "total": len(prompts),
                            "seconds": seconds,
                            "tokens": len(generated.token_ids),
                        }
                    ),
                    flush=True,
                )
            entry = {
                "results": records,
                "summary": summarize_rows(records, bool(args.draft)),
            }
            if reference:
                entry["comparison_to_ar"] = paired_summary(
                    reference["datasets"][name], entry, args.temperature
                )
            report["datasets"][name] = entry
            output.write_text(json.dumps(report, indent=2) + "\n")
        report["complete"] = True
        output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        llm.llm_engine.engine_core.shutdown()
    print(f"Report: {output}", flush=True)


def suite(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new evaluation output directory")
    if not args.baseline or not args.retrace:
        raise ValueError(
            "Provide --baseline INITIAL/dflash and --retrace TRAINED_CHECKPOINT"
        )
    for path, expected in ((args.baseline, "dflash"), (args.retrace, "retrace")):
        raw = json.loads((Path(path) / "config.json").read_text())
        if raw.get("speculators_model_type") != expected or raw.get("block_size") != 16:
            raise ValueError(f"Expected a b16 {expected} checkpoint: {path}")
    output.mkdir(parents=True)
    report_paths = []
    for temperature in args.temperatures:
        for method, draft in (
            ("ar", None),
            ("dflash", args.baseline),
            ("retrace", args.retrace),
        ):
            path = output / f"{method}_t{temperature}.json"
            command = [
                sys.executable,
                "-m",
                "speculators.models.retrace.table2_benchmark",
                "worker",
                "--target",
                args.target,
                "--manifest",
                args.manifest,
                "--output",
                str(path),
                "--method",
                method,
                "--temperature",
                str(temperature),
                "--max-tokens",
                str(args.max_tokens),
            ]
            if draft:
                command += [
                    "--draft",
                    draft,
                    "--compare",
                    str(output / f"ar_t{temperature}.json"),
                ]
            with (output / f"{method}_t{temperature}.log").open(
                "w", buffering=1
            ) as log:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    if process.wait():
                        raise RuntimeError(f"Benchmark exited; read {log.name}")
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            report_paths.append(path)
    manifest = json.loads(Path(args.manifest).read_text())
    table = []
    for path in report_paths:
        report = json.loads(path.read_text())
        if report["args"]["method"] == "ar":
            continue
        group = []
        for name, entry in report["datasets"].items():
            row = {
                "method": report["args"]["method"],
                "temperature": report["args"]["temperature"],
                "benchmark": name,
                **{
                    key: entry["summary"][key]
                    for key in (
                        "tokens_per_second",
                        "accepted_per_round_pooled",
                        "acceptance_length_with_bonus_pooled",
                        "acceptance_length_with_bonus_request_mean",
                    )
                },
                **{
                    key: entry["comparison_to_ar"][key]
                    for key in (
                        "greedy_parity",
                        "comparison_valid",
                        "speedup_tps",
                        "speedup_total_latency",
                        "speedup_request_tps_mean",
                    )
                },
            }
            table.append(row)
            group.append(row)
        if (
            manifest["complete_table2_counts"]
            and args.max_tokens == 8192
            and len(group) == len(COUNTS)
            and all(row["comparison_valid"] for row in group)
        ):
            table.append(
                {
                    "method": report["args"]["method"],
                    "temperature": report["args"]["temperature"],
                    "benchmark": "MACRO_AVERAGE_7",
                    **{
                        key: statistics.mean(row[key] for row in group)
                        for key in (
                            "speedup_tps",
                            "speedup_total_latency",
                            "speedup_request_tps_mean",
                            "accepted_per_round_pooled",
                            "acceptance_length_with_bonus_pooled",
                            "acceptance_length_with_bonus_request_mean",
                        )
                    },
                }
            )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "rows": table,
                "paper_numbers_reproduced": None,
                "author_protocol_exact": False,
                "manifest_sha256": sha256(args.manifest),
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Paired results: {output / 'summary.csv'}", flush=True)
    if any(row.get("comparison_valid") is False for row in table):
        raise SystemExit(
            "A correctness/metrics check failed; invalid speedup cells are blank. Inspect individual reports."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("suite", "worker"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--retrace")
    parser.add_argument("--draft")
    parser.add_argument("--compare")
    parser.add_argument("--method", choices=("ar", "dflash", "retrace"), default="ar")
    parser.add_argument("--temperature", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--temperatures", type=int, nargs="+", choices=(0, 1), default=[0, 1]
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.max_tokens < 1 or len(set(args.temperatures)) != len(args.temperatures):
        parser.error("Invalid token limit or repeated temperature")
    (suite if args.action == "suite" else worker)(args)


if __name__ == "__main__":
    main()
