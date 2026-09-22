"""Compare cached eager/static training with SDPA/dynamic training at a saved step.

Both branches preserve the complete recipe and global prompt batches. This
does not certify author-code equivalence, greedy parity, or Table 2 results.
"""

import argparse
import json
import os
import shlex
import statistics
import sys
from pathlib import Path

from .batched_benchmark import summarize_branch
from .performance_benchmark import make_command, resume_script, run_branch
from .pretrained import sha256


def commands_for(
    signature, checkpoint, pool, output, stop, blocks, target_norm, profile=True
):
    commands = {}
    for name, attention, distribution, norm in (
        ("baseline", "eager", "static", "reference"),
        ("fast", "sdpa", "dynamic", target_norm),
    ):
        command = make_command(
            signature,
            checkpoint,
            checkpoint,
            pool,
            output / name / "checkpoints",
            "cached",
            blocks,
            stop,
            profile,
        )
        command += [
            "--rollout-batch-size",
            "1",
            "--attention-backend",
            attention,
            "--work-distribution",
            distribution,
            "--target-norm",
            norm,
        ]
        commands[name] = command
    return commands


def validate_inputs(args):
    if any(not value.strip() for value in (args.run, args.checkpoint, args.output)):
        raise ValueError("Run, checkpoint and output paths must not be empty")
    if (
        args.steps < args.warmup_updates + 2
        or args.warmup_updates < 0
        or args.blocks < 1
    ):
        raise ValueError(
            "Need at least two measured updates, nonnegative warmup and positive blocks"
        )
    run, checkpoint, output = [
        Path(p).resolve() for p in (args.run, args.checkpoint, args.output)
    ]
    if output.exists():
        raise ValueError("Choose a new benchmark output directory")
    state = json.loads((checkpoint / "training_state.json").read_text())
    signature, start = state["signature"], int(state["step"])
    if signature["target_backend"] != "local":
        raise ValueError(
            "This update requires the local-target pretrained DFlash trainer"
        )
    required = ["config.json", "master_parameters.pt", "optimizer.pt"] + [
        f"rng_rank_{r}.pt" for r in range(signature["world_size"])
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete checkpoint: {missing}")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("speculators_model_type") != "retrace" or not config.get(
        "pretrained_fingerprint"
    ):
        raise ValueError("Expected the pretrained DFlash ReTrace checkpoint")
    pool = run / "prompts.jsonl"
    if not pool.is_file() or sha256(pool) != signature["pool_sha256"]:
        raise ValueError(
            "Original prompt pool is missing or differs from the checkpoint"
        )
    ids = args.npus.split(",")
    if (
        not all(x.isdecimal() for x in ids)
        or len(set(ids)) != len(ids)
        or len(ids) != signature["world_size"]
    ):
        raise ValueError("Use distinct physical NPU IDs matching the saved world size")
    stop = start + args.steps
    if stop > signature["total_steps"]:
        raise ValueError("Not enough updates remain for this comparison")
    return run, checkpoint, output, pool, signature, start, stop


def summarize_workers(rows, warmup):
    report = summarize_branch(rows, warmup)
    measured = rows[warmup:]
    report["rank_prompt_counts"] = [
        [r["prompts"] for r in row["rank_performance"]] for row in measured
    ]
    balances = []
    for row in measured:
        workers = row["rank_performance"]
        if sum(r["prompts"] for r in workers) != row["global_prompts"]:
            raise ValueError("Worker prompt counts differ from the global batch")
        times = [r["compute_seconds"] for r in workers]
        balances.append(sum(times) / (len(times) * max(times)))
    report["mean_rank_work_balance"] = statistics.mean(balances)
    report["attention_backend"] = rows[0]["attention_backend"]
    report["target_norm"] = rows[0]["target_norm"]
    report["work_distribution"] = rows[0]["work_distribution"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="Original run containing prompts.jsonl"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Completed saved step with optimizer and rank RNG",
    )
    parser.add_argument(
        "--output", required=True, help="New directory for both training branches"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--target-norm", choices=("reference", "npu"), default="npu")
    parser.add_argument(
        "--npus", default=os.environ.get("RETRACE_EXEC_NPUS", "0,1,2,3,4,5,6,7")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run, checkpoint, output, pool, signature, start, stop = validate_inputs(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    commands = commands_for(
        signature, checkpoint, pool, output, stop, args.blocks, args.target_norm
    )
    check = [
        sys.executable,
        "-m",
        "speculators.models.retrace.execution_check",
        "--device",
        "npu:0",
        "--dtype",
        signature["dtype"],
        "--target-norm",
        args.target_norm,
        "--output",
        str(output / "operator_check.json"),
    ]
    print(
        f"Resume {start}; pause/save at {stop}; preserve {signature['total_steps']} total updates",
        flush=True,
    )
    print("Operator check: " + shlex.join(check), flush=True)
    for name, command in commands.items():
        print(f"{name}: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "original_run": str(run),
                "checkpoint": str(checkpoint),
                "signature": signature,
                "commands": commands,
                "comparison": "Same saved step, optimizer, prompt pool, batch32 and schedule; execution only",
                "limit": "BF16 execution and reduction order can change rounding and subsequent trajectories",
            },
            indent=2,
        )
        + "\n"
    )
    env = os.environ | {
        "ASCEND_RT_VISIBLE_DEVICES": args.npus,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    }
    run_branch(check, output / "operator_check", env)
    check_report = json.loads((output / "operator_check.json").read_text())
    if check_report.get("passed") is not True:
        raise RuntimeError("NPU operator check failed")
    root = Path(__file__).resolve().parents[5]
    reports = {}
    for name, command in commands.items():
        run_branch(command, output / name, env)
        rows = [
            json.loads(line)
            for line in (output / name / "checkpoints/metrics.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        if [row["step"] for row in rows] != list(range(start + 1, stop + 1)):
            raise RuntimeError("Branches did not complete the same planned updates")
        if any(row["planned_steps"] != signature["total_steps"] for row in rows):
            raise RuntimeError("Training schedule changed during comparison")
        reports[name] = summarize_workers(rows, args.warmup_updates)
        # Write each continuation as soon as its branch succeeds, even if the
        # other branch subsequently fails. Never point back to unsaved progress.
        saved = output / name / "checkpoints" / f"step_{stop:06d}"
        continuation = commands_for(
            signature,
            saved,
            pool,
            output,
            0,
            args.blocks,
            args.target_norm,
            profile=False,
        )[name]
        script = output / f"resume_{name}.sh"
        script.write_text(resume_script(continuation, args.npus, root))
        script.chmod(0o755)
    baseline, fast = [
        reports[name]["mean_step_seconds"] for name in ("baseline", "fast")
    ]
    remaining = signature["total_steps"] - stop
    report = {
        **reports,
        "measured_update_speedup": baseline / fast,
        "remaining_updates_after_comparison": remaining,
        "estimated_remaining_days": {
            name: remaining * r["mean_step_seconds"] / 86400
            for name, r in reports.items()
        },
        "fast_mode_faster": fast < baseline,
        "operator_check_passed": True,
        "native_serving_validated": False,
        "limits": "Short NPU sample; excludes startup/checkpoint I/O. Rank balance is not NPU utilization. Math backend may change greedy paths. No Table 2 reproduction or bitwise-equivalence claim.",
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Compare results at {output / 'summary.json'}", flush=True)
    print(
        f"Saved continuations: {output / 'resume_fast.sh'} and {output / 'resume_baseline.sh'}",
        flush=True,
    )
    if fast >= baseline:
        print(
            "Fast mode did not improve this sample. The baseline continuation is available.",
            flush=True,
        )


if __name__ == "__main__":
    main()
