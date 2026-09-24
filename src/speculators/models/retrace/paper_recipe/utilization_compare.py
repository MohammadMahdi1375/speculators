"""Compare execution on the same saved model, prompt order and training schedule."""

import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from .utilization_resume import check_checkpoint, resolve_run


def summarize(path, start_step, warmup):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in records if row["step"] > start_step + warmup]
    if not rows or any("step_seconds" not in row for row in rows):
        raise ValueError("No complete timed optimizer updates after warmup")
    def mean(name):
        return statistics.mean(float(row[name]) for row in rows)
    result = {
        "steps": [row["step"] for row in rows], "mean_step_seconds": mean("step_seconds"),
        "median_step_seconds": statistics.median(row["step_seconds"] for row in rows),
        "global_prompts_per_update": [row["global_prompts"] for row in rows],
        "mean_rollout_seconds_max": mean("rollout_seconds_max"),
        "mean_backward_seconds_max": mean("backward_seconds_max"),
        "mean_worker_wait_seconds_max": mean("worker_wait_seconds_max"),
        "peak_memory_gib": max(row["peak_memory_gib_max"] for row in rows),
        "mean_active_rollouts_per_draft_forward": mean("mean_active_rollouts_per_draft_forward"),
        "mean_target_forwards": mean("target_forwards"),
        "mean_draft_forwards": mean("draft_forwards"),
        "mean_logical_rounds": mean("rounds"),
        "mean_clean_labels": mean("clean_labels"),
        "rollout_scheduler": rows[0]["rollout_scheduler"],
        "rollout_batch_size": rows[0]["rollout_batch_size"],
    }
    seconds = sum(row["step_seconds"] for row in rows)
    result["prompts_per_second"] = sum(row["global_prompts"] for row in rows) / seconds
    balances = []
    for row in rows:
        times = [rank["compute_seconds"] for rank in row["rank_performance"]]
        balances.append(sum(times) / (len(times) * max(times)))
    result["approximate_rank_work_balance"] = statistics.mean(balances)
    return result


def compare(args):
    if args.resume or args.resume_run or args.smoke:
        raise ValueError("Use --compare-run separately from --resume, --resume-run and --smoke")
    if not 0 <= args.compare_warmup < args.compare_steps:
        raise ValueError("Comparison warmup must be smaller than comparison update count")
    checkpoint, pool = resolve_run(args.compare_run)
    state = check_checkpoint(checkpoint, len(args.trainer_npus))
    signature = state["signature"]
    if signature["target_backend"] != "local" or signature["world_size"] != len(args.trainer_npus):
        raise ValueError("Comparison requires the same worker count and a local-target checkpoint")
    if (signature.get("lr_warmup_ratio") != .05 or signature.get("weight_decay") != .01
            or signature.get("gamma") != 4. or signature.get("seed") != 42
            or signature.get("dtype") != "bfloat16"):
        raise ValueError("This launcher comparison supports the supplied paper recipe defaults only")
    stop = state["step"] + args.compare_steps
    if stop > signature["total_steps"]:
        raise ValueError("Checkpoint does not have enough remaining planned updates for this comparison")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Use a new comparison output directory")
    common = [sys.executable, "-m", "speculators.models.retrace.paper_recipe.utilization_launch",
        "--root", args.root, "--target", signature["target"], "--base-dflash", args.base_dflash,
        "--data", args.data, "--pool", str(pool), "--resume", str(checkpoint),
        "--trainer-npus", ",".join(args.trainer_npus), "--allow-performance-change",
        "--profile-performance", "--stop-after", str(stop), "--save-every", "0",
        "--max-prompts", str(sum(1 for line in pool.open() if line.strip())),
        "--max-steps", str(signature["total_steps"])]
    for name in ("prompt_length", "response_length", "global_prompt_batch", "epochs", "lr"):
        common += ["--" + name.replace("_", "-"), str(signature[name])]
    if signature.get("disable_conditioning"):
        common += ["--disable-conditioning"]
    execution = ("performance_mode", "attention_backend", "target_norm", "work_distribution",
                 "rollout_batch_size", "trace_storage", "blocks_per_forward", "rollout_scheduler")
    baseline = {key: signature.get(key, "cohort" if key == "rollout_scheduler" else getattr(args, key))
                for key in execution}
    fast = {key: getattr(args, key) for key in execution}
    commands = {}
    for name, settings in (("baseline", baseline), ("fast", fast)):
        command = common + ["--output", str(output / name)]
        for key, value in settings.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        commands[name] = command
        print(name + ": " + shlex.join(command), flush=True)
    if args.dry_run:
        print("Comparison dry run; no training processes started.")
        return
    output.mkdir(parents=True)
    results = {}
    for name, command in commands.items():
        subprocess.run(command, check=True, cwd=Path(args.root) / "speculators")
        run = output / name
        results[name] = summarize(run / "checkpoints/metrics.jsonl", state["step"], args.compare_warmup)
        saved = Path((run / "checkpoints/latest.txt").read_text().strip())
        resume = list(command)
        resume[resume.index("--resume") + 1] = str(saved)
        resume[resume.index("--stop-after") + 1] = "0"
        script = output / f"resume_{name}.sh"
        # Source the same NPU environment as the training bash, while preserving
        # all explicit paired settings and the already completed updates.
        launcher = Path(args.root) / "speculators/examples/train/train_retrace_dflash_local_fast.sh"
        options = resume[3:]
        script.write_text("#!/usr/bin/env bash\nset -eo pipefail\nexec bash " + shlex.quote(str(launcher))
                          + " " + shlex.join(options) + "\n")
        script.chmod(0o755)
    speedup = results["baseline"]["mean_step_seconds"] / results["fast"]["mean_step_seconds"]
    results.update(measured_step_speedup=speedup, fast_faster=speedup > 1.,
        remaining_updates=signature["total_steps"] - stop,
        estimated_remaining_days_at_fast_rate=(signature["total_steps"] - stop)
            * results["fast"]["mean_step_seconds"] / 86400,
        limits="Short same-checkpoint comparison; BF16 batching and scheduling can change trajectories. "
               "Excludes checkpoint/startup time. Rank work balance is not measured NPU compute utilization.")
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    best = "fast" if results["fast_faster"] else "baseline"
    best_script = output / "resume_best.sh"
    best_script.write_text("#!/usr/bin/env bash\nset -eo pipefail\nexec bash "
                           + shlex.quote(str(output / f"resume_{best}.sh")) + "\n")
    best_script.chmod(0o755)
    print("Continue the faster measured branch: bash " + str(best_script), flush=True)
