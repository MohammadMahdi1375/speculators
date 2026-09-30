#!/usr/bin/env python3
"""Replay selected frozen prompts using the installed evaluator's HTTP client.

Indices are zero-based within the selected dataset; with concurrency one,
69 completed requests means index 69 is the next request to investigate.
This is a diagnostic run, never a throughput or full-validation result.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile


DEFAULT_MAIN = "/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
DEFAULT_TARGET = "/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location("prefix_failure_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("Client", "sampling", "select_prompts", "run_selector_warmup", "run_warmups"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Installed evaluator lacks {name}; no request sent")
    return module


def selected_rows(ev, manifest, dataset, count, indices, target):
    rows = ev.select_prompts(manifest, [dataset], count, False)[dataset]
    if manifest.get("target_config_sha256") != file_hash(target / "config.json"):
        raise ValueError("Target config differs from the frozen manifest")
    for name, expected in manifest.get("tokenizer_sha256", {}).items():
        if Path(name).name != name or file_hash(target / name) != expected:
            raise ValueError("Target tokenizer differs from the frozen manifest")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError(f"Indices must be within 0..{len(rows)-1}")
    return rows


def write_report(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)


def run(args, ev):
    manifest = json.loads(args.manifest.read_text())
    indices = args.indices
    if args.replay_prefix:
        indices = list(range(max(indices) + 1))
    rows = selected_rows(ev, manifest, args.dataset, args.num_prompts, indices, args.target)
    eos = json.loads((args.target / "generation_config.json").read_text()).get("eos_token_id", 151645)
    eos = [eos] if isinstance(eos, int) else eos
    if not eos or any(type(x) is not int or x < 0 for x in eos):
        raise ValueError("Invalid EOS IDs in target generation_config.json")
    controls = SimpleNamespace(top_p=1.0, top_k=-1, min_p=0.0, max_new_tokens=args.max_new_tokens,
        seed=args.seed, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
        min_tokens=0, ignore_eos=False, stop_token_ids=None, skip_special_tokens=True)
    params = ev.sampling(controls, 0.0, eos)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "probe.json"
    report = {"format": "prefix_failure_probe_v1", "diagnostic_only": True,
        "full_validation_pass": False, "dataset": args.dataset, "indices_zero_based": indices,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": file_hash(args.manifest),
        "evaluator_sha256": file_hash(args.evaluator), "sampling": params,
        "requests": [], "complete": False,
        "note": "Same frozen token IDs, streaming client, greedy sampling, and warmup functions "
                "as the installed evaluator. Different preceding requests may affect reproduction. "
                "Use --replay-prefix only if the focused request does not reproduce the failure."}
    start_ns = time.time_ns()
    write_report(path, report)
    failed = False
    try:
        client = ev.Client(args.base_url, args.model, args.timeout_s, None)
        report["server_model_info"] = client.info
        report["startup_warmup"] = ev.run_selector_warmup(
            client, rows[0], params, args.warmup_tokens, required_steps=args.warmup_steps)
        ev.run_warmups(client, rows, params, 1, 1)
        for repetition in range(args.repeat):
            for index in indices:
                prompt = rows[index]
                item = {"dataset_index_zero_based": index, "request_number_one_based": index + 1,
                        "repetition": repetition + 1, "prompt_sha256": prompt["sha256"],
                        "source_row": prompt.get("source_row"), "source_id": prompt.get("source_id"),
                        "status": "running"}
                report["requests"].append(item)
                write_report(path, report)
                print(f"Diagnostic request: {args.dataset} index={index} (request {index+1}), "
                      f"repeat={repetition+1}; not a benchmark", flush=True)
                result = client.complete(prompt["prompt_token_ids"], params)
                item.update(status="completed", completion_tokens=result.get("completion_tokens"),
                            finish_reason=result.get("finish_reason"), speculation=result.get("speculation"))
                write_report(path, report)
        report["complete"] = True
    except Exception as exc:
        failed = True
        report["request_error"] = f"{type(exc).__name__}: {exc}"
        if report["requests"] and report["requests"][-1]["status"] == "running":
            report["requests"][-1]["status"] = "failed"
        print("Diagnostic request stopped: " + report["request_error"], flush=True)
    finally:
        # The server persists its snapshot before raising the HTTP error. Only
        # include new files from this probe interval, never previous run data.
        files = []
        for file in sorted(args.diagnostic_dir.glob("prefix_failure_*")):
            if file.is_file() and file.suffix in (".json", ".pt") and file.stat().st_mtime_ns >= start_ns:
                files.append(file)
        report["server_diagnostic_files"] = [str(p.resolve()) for p in files]
        write_report(path, report)
        archive = args.output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
            output.write(path, "probe.json")
            for file in files:
                output.write(file, "server/" + file.name)
        print(f"Upload this diagnostic ZIP: {archive.resolve()}", flush=True)
        if not files:
            print("No new server failure snapshot found. This does not certify full parity.", flush=True)
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec-main", type=Path, default=Path(DEFAULT_MAIN))
    p.add_argument("--evaluator", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--target", type=Path, default=Path(DEFAULT_TARGET))
    p.add_argument("--base-url", default="http://127.0.0.1:8209")
    p.add_argument("--model", default="qwen3-4b-dflash")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--indices", nargs="+", type=int, default=[69])
    p.add_argument("--replay-prefix", action="store_true")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--warmup-tokens", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=32)
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--diagnostic-dir", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.repeat, args.max_new_tokens, args.warmup_tokens, args.warmup_steps, args.timeout_s) <= 0:
        p.error("Repeat, token counts, warmup steps, and timeout must be positive")
    args.evaluator = args.evaluator or args.spec_main / "Evaluator/evaluator.py"
    args.manifest = args.manifest or args.spec_main / "Evaluator/retrace_eval_prompts_all_v2.json"
    args.diagnostic_dir = args.diagnostic_dir or args.spec_main / "output/prefix_fusion_diagnostics"
    args.output = args.output or args.spec_main / f"output/prefix_failure_probe_{time.time_ns()}"
    return run(args, load_evaluator(args.evaluator))


if __name__ == "__main__":
    raise SystemExit(main())
