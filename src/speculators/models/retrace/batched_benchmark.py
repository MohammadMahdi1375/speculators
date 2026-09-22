"""Compare serial cached and batched rollouts from one complete checkpoint."""

import argparse
import json
import math
import os
import shlex
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .performance_benchmark import make_command, resume_script, run_branch, summarize
from .pretrained import sha256


def commands_for(signature, checkpoint, pool, output, stop, rollout_batch_size, blocks):
    commands = {}
    for name, size in (("serial", 1), ("batched", rollout_batch_size)):
        command = make_command(
            signature,
            checkpoint,
            checkpoint,
            pool,
            output / name / "checkpoints",
            "cached",
            blocks,
            stop,
            True,
        )
        command += ["--rollout-batch-size", str(size)]
        commands[name] = command
    return commands


def summarize_branch(rows, warmup):
    report = summarize(rows, warmup)
    selected = rows[warmup:]
    for name in (
        "worker_wait_seconds_max",
        "gradient_sync_seconds_max",
        "compute_seconds_max",
        "target_forwards",
        "draft_forwards",
    ):
        report["mean_" + name] = statistics.mean(row[name] for row in selected)
    report["rollout_batch_size"] = rows[0]["rollout_batch_size"]
    report["optimizer_timing"] = (
        "Gradient clipping and AdamW only; worker wait and gradient synchronization are separate"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="Original full run containing prompts.jsonl"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Complete saved step, with optimizer and every rank RNG",
    )
    parser.add_argument("--output", required=True, help="New benchmark directory")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument(
        "--npus", default=os.environ.get("RETRACE_BATCH_NPUS", "0,1,2,3,4,5,6,7")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.run.strip() or not args.checkpoint.strip() or not args.output.strip():
        parser.error("Run, checkpoint and output paths must not be empty")
    if (
        args.steps < args.warmup_updates + 2
        or args.warmup_updates < 0
        or args.blocks < 1
    ):
        parser.error(
            "Need at least two measured updates, nonnegative warmup and positive blocks"
        )
    run, checkpoint, output = (
        Path(value).resolve() for value in (args.run, args.checkpoint, args.output)
    )
    if output.exists():
        parser.error("Choose a new benchmark output directory")
    state_file = checkpoint / "training_state.json"
    if not state_file.is_file():
        parser.error(
            f"No completed training state at {checkpoint}; do not restart from initial"
        )
    state = json.loads(state_file.read_text())
    signature, start = state["signature"], int(state["step"])
    if signature["target_backend"] != "local":
        parser.error(
            "This benchmark requires the local-target pretrained DFlash trainer"
        )
    if (
        not 2
        <= args.rollout_batch_size
        <= math.ceil(signature["global_prompt_batch"] / signature["world_size"])
    ):
        parser.error(
            "Rollout batch size must be at least two and fit the saved local prompt batch"
        )
    required = ["config.json", "master_parameters.pt", "optimizer.pt"] + [
        f"rng_rank_{rank}.pt" for rank in range(signature["world_size"])
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        parser.error(f"Incomplete checkpoint: missing {missing}")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("speculators_model_type") != "retrace" or not config.get(
        "pretrained_fingerprint"
    ):
        parser.error("Expected an imported pretrained DFlash ReTrace checkpoint")
    pool = run / "prompts.jsonl"
    if not pool.is_file() or sha256(pool) != signature["pool_sha256"]:
        parser.error(
            "The original prompt pool is missing or differs from the checkpoint"
        )
    ids = args.npus.split(",")
    if (
        not all(x.isdecimal() for x in ids)
        or len(set(ids)) != len(ids)
        or len(ids) != signature["world_size"]
    ):
        parser.error("Use distinct physical NPU IDs matching the saved world size")
    stop = start + args.steps
    if stop > signature["total_steps"]:
        parser.error("Not enough updates remain for this benchmark")
    commands = commands_for(
        signature, checkpoint, pool, output, stop, args.rollout_batch_size, args.blocks
    )
    print(
        f"Resume step {start}; pause at {stop}; preserve original total {signature['total_steps']} updates",
        flush=True,
    )
    for name, command in commands.items():
        print(f"{name}: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "args": vars(args),
                "checkpoint": str(checkpoint),
                "signature": signature,
                "commands": commands,
                "comparison": "Same checkpoint/pool/updates; serial cached versus batched cached rollouts",
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
    branches = {}
    for name, command in commands.items():
        print(f"Starting {name} branch", flush=True)
        try:
            run_branch(command, output / name, env)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}\nFor batched allocation failure, retry in a NEW output directory with --rollout-batch-size 2. The input checkpoint is unchanged."
            ) from exc
        metrics = output / name / "checkpoints/metrics.jsonl"
        rows = [
            json.loads(line)
            for line in metrics.read_text().splitlines()
            if line.strip()
        ]
        if [row["step"] for row in rows] != list(range(start + 1, stop + 1)):
            raise RuntimeError("A branch did not complete the same planned updates")
        branches[name] = summarize_branch(rows, args.warmup_updates)
    serial = branches["serial"]["mean_step_seconds"]
    batched = branches["batched"]["mean_step_seconds"]
    report = {
        **branches,
        "measured_update_speedup": serial / batched,
        "remaining_updates_after_benchmark": signature["total_steps"] - stop,
        "estimated_remaining_days_at_measured_rate": (signature["total_steps"] - stop)
        * batched
        / 86400,
        "batched_mode_faster": batched < serial,
        "limits": "Four measured updates by default; excludes startup/save I/O. Batching may change BF16 rounding and subsequent trajectories. Target calls count logical row evaluations; target_forwards counts actual batched model invocations. Per-phase rank maxima must not be added.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    root = Path(__file__).resolve().parents[5]
    for name, size in (("serial", 1), ("batched", args.rollout_batch_size)):
        saved = output / name / "checkpoints" / f"step_{stop:06d}"
        command = make_command(
            signature,
            saved,
            saved,
            pool,
            output / name / "checkpoints",
            "cached",
            args.blocks,
            0,
            False,
        )
        command += ["--rollout-batch-size", str(size)]
        script = output / f"resume_{name}.sh"
        script.write_text(resume_script(command, args.npus, root))
        script.chmod(0o755)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Measured results: {output / 'summary.json'}", flush=True)
    print(
        f"Continue the chosen branch with resume_batched.sh or resume_serial.sh in {output}",
        flush=True,
    )
    if batched >= serial:
        print(
            "Batched execution was not faster in this sample; do not assume a speedup.",
            flush=True,
        )


if __name__ == "__main__":
    main()
