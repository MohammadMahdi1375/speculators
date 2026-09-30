#!/usr/bin/env python3
"""Check unpaired fused walks on a saved selector failure, without vLLM/target.

This checks prepared-table walk arithmetic. It does not certify the serving
pipeline, unseen inputs, graph preparation, acceptance length, or throughput.
"""
import argparse
from functools import partial
import hashlib
import importlib
import json
import logging
from pathlib import Path
import sys
import time
import types


def modules(model_dir):
    name = "prefix_snapshot_replay_package"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(model_dir)]
    sys.modules[name] = pkg
    return {key: importlib.import_module(name + "." + key) for key in (
        "selector_v2", "inference_v2", "paired_history_v2", "fused_unpaired_v2", "failure_diagnostics_v2")}


def restore_snapshot(snapshot, device, mods):
    import torch
    # Only plain tensors/dictionaries written by the diagnostic snapshot.
    data = torch.load(snapshot, map_location="cpu", weights_only=True)
    if data.get("format") != "prefix_failure_snapshot_v1":
        raise ValueError("Unexpected snapshot format")
    cfg = data["head_config"]
    head = mods["selector_v2"].LocalPrefixSelector(**cfg)
    # Preserve each saved tensor's dtype, including any mixed-precision norms.
    head.load_state_dict(data["state_dict"], strict=True, assign=True)
    head = head.to(device=device).eval()
    prepared_cls = mods["selector_v2"].PreparedPrefix
    table = prepared_cls(*(data["prepared"][name].to(device) for name in prepared_cls._fields))
    return data, head, table


def check(args, mods):
    import torch
    compare = mods["failure_diagnostics_v2"].comparison
    atomic_json = mods["failure_diagnostics_v2"].atomic_json
    data, head, table = restore_snapshot(args.snapshot, args.device, mods)
    with torch.inference_mode():
        reference, scores = head.walk(table, return_scores=True)
        saved = compare(data["expected_tokens"], data["expected_scores"], reference, scores)
        report = {"format": "unpaired_snapshot_replay_v1", "complete": False,
            "snapshot": str(args.snapshot.resolve()), "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
            "device": args.device, "torch_version": str(torch.__version__),
            "head_config": data["head_config"], "saved_reference_vs_replay": saved,
            "saved_reference_bit_exact": torch.equal(data["expected_scores"], scores.cpu()) and torch.equal(data["expected_tokens"], reference.cpu()),
            "comparisons": {}, "candidate_pass": False,
            "note": "Prepared-table eager replay only. Strict token/score criteria retained. "
                    "A pass does not certify combined graph preparation, unseen prompts, tau, or throughput."}
        atomic_json(args.output, report)
        # Do not certify a fix if this process cannot reproduce the saved
        # reference on the same snapshot (device/autocast/layout may differ).
        if not report["saved_reference_bit_exact"]:
            report["error"] = "Saved reference did not replay bit-exactly; inspect the environment before comparing candidates"
            atomic_json(args.output, report)
            return report
        factories = {
            "cached_unpaired": mods["inference_v2"].CachedPrefixWalk,
            "paired": mods["paired_history_v2"].PairedHistoryWalk,
            "select_unpaired": partial(mods["fused_unpaired_v2"].FusedUnpairedHistoryWalk, variant="select"),
            "local_unpaired": partial(mods["fused_unpaired_v2"].FusedUnpairedHistoryWalk, variant="local"),
        }
        for name, factory in factories.items():
            # Kernel errors propagate; never continue using an unhealthy stream.
            walker = factory(head)
            tokens, candidate_scores = walker.walk(table, return_scores=True)
            production = walker.walk(table)
            result = compare(reference, scores, tokens, candidate_scores, production)
            result["bit_exact_scores"] = torch.equal(scores, candidate_scores)
            result["pass"] = (result["finite"] and result["close"] and result["scored_tokens_equal"]
                              and result["production_tokens_equal"])
            report["comparisons"][name] = result
            atomic_json(args.output, report)
        report["candidate"] = args.candidate
        report["candidate_pass"] = report["comparisons"][args.candidate]["pass"]
        report["complete"] = True
        atomic_json(args.output, report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec-main", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--candidate", choices=("local_unpaired", "select_unpaired"), default="local_unpaired")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.device != "npu:0":
        p.error("Use --device npu:0 with ASCEND_RT_VISIBLE_DEVICES selecting the physical NPU")
    args.output = args.output or args.spec_main / f"output/unpaired_snapshot_replay_{time.time_ns()}.json"
    if args.output.exists():
        p.error("Output already exists; choose a new report path")
    import torch
    import torch_npu
    torch_npu.npu.set_device(0)
    logging.basicConfig(level=logging.INFO)
    mods = modules(args.spec_main / "speculators/src/speculators/models/dflash_prefix")
    print(f"Replay report: {args.output.resolve()}", flush=True)
    report = check(args, mods)
    print("PREFIX_UNPAIRED_REPLAY " + json.dumps({
        "report": str(args.output.resolve()), "complete": report["complete"],
        "saved_reference_bit_exact": report["saved_reference_bit_exact"],
        "candidate_pass": report["candidate_pass"],
        "comparisons": {name: {key: result[key] for key in ("max_abs_error", "scored_tokens_equal", "production_tokens_equal", "pass")}
                        for name, result in report["comparisons"].items()},
    }, sort_keys=True))
    print("Upload the report if candidate_pass is false; do not start a benchmark on a failure.")
    return 0 if report["candidate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
