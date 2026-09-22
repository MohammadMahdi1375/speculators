"""Compare complete updates from the same saved DFlash-ReTrace checkpoint.

Both branches restore the same optimizer/RNG state, prompt order, world size,
and learning-rate schedule. Only execution settings differ. Each branch saves
at a planned pause; the original run is read-only. Hardware timing is measured,
never inferred from CPU unit tests or from the number of optimized operations.
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .launch import stop_group
from .pretrained import sha256


def make_command(signature, draft, resume, pool, output, mode, blocks, stop, profile):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(signature["world_size"]),
        "-m",
        "speculators.models.retrace.train_pretrained",
        "--draft",
        str(draft),
        "--pool",
        str(pool),
        "--output",
        str(output),
        "--device",
        "npu",
        "--save-every",
        "0",
        "--allow-performance-change",
        "--max-steps",
        str(signature["total_steps"]),
        "--blocks-per-forward",
        str(blocks),
        "--performance-mode",
        mode,
        "--trace-storage",
        "device" if mode == "cached" else "cpu",
    ]
    for name in (
        "target",
        "prompt_length",
        "response_length",
        "global_prompt_batch",
        "epochs",
        "lr",
        "lr_warmup_ratio",
        "weight_decay",
        "gamma",
        "seed",
        "dtype",
    ):
        command += ["--" + name.replace("_", "-"), str(signature[name])]
    if signature["disable_conditioning"]:
        command += ["--disable-conditioning"]
    if resume is not None:
        command += ["--resume", str(resume)]
    if stop:
        command += ["--stop-after", str(stop)]
    if profile:
        command += ["--profile-performance"]
    return command


def run_branch(command, output, env):
    process = None
    output.mkdir(parents=True, exist_ok=False)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with (output / "trainer.log").open("w", buffering=1) as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        if process.wait():
            raise RuntimeError(
                f"Training branch failed; see {output / 'trainer.log'}. "
                "For allocation failure, rerun into a new benchmark output with --blocks 8. "
                "The original checkpoint is unchanged."
            )
    finally:
        if process is not None and process.poll() is None:
            stop_group(process)


def summarize(rows, warmup):
    measured = rows[warmup:]
    if len(measured) < 2 or any(r.get("step_seconds", 0) <= 0 for r in measured):
        raise ValueError("Need at least two complete profiled updates after warmup")
    times = [r["step_seconds"] for r in measured]
    return {
        "steps": [r["step"] for r in measured],
        "mean_step_seconds": statistics.mean(times),
        "median_step_seconds": statistics.median(times),
        "min_step_seconds": min(times),
        "max_step_seconds": max(times),
        "mean_rollout_seconds_max": statistics.mean(
            r["rollout_seconds_max"] for r in measured
        ),
        "mean_backward_seconds_max": statistics.mean(
            r["backward_seconds_max"] for r in measured
        ),
        "mean_optimizer_seconds_max": statistics.mean(
            r["optimizer_seconds_max"] for r in measured
        ),
        "peak_memory_gib": max(r["peak_memory_gib_max"] for r in rows),
        "mean_rounds": statistics.mean(r["rounds"] for r in measured),
        "mean_target_calls": statistics.mean(r["target_calls"] for r in measured),
        "global_prompts_per_step": [r["global_prompts"] for r in measured],
    }


def resume_script(command, npus, root):
    cann = os.environ.get("RETRACE_CANN_ROOT", "/home/n84449292/m84379596/CANN/9.1.0")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            "source " + shlex.quote(cann + "/ascend-toolkit/set_env.sh"),
            "source " + shlex.quote(cann + "/nnal/atb/set_env.sh"),
            "export ASCEND_RT_VISIBLE_DEVICES=" + shlex.quote(npus),
            "export PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1",
            "export PYTHONPATH="
            + shlex.quote(str(root / "speculators/src"))
            + "${PYTHONPATH:+:$PYTHONPATH}",
            "exec " + shlex.join(command),
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="Existing full DFlash run directory"
    )
    parser.add_argument("--checkpoint", help="Defaults to RUN/checkpoints/latest.txt")
    parser.add_argument("--output", help="New directory for the paired benchmark")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument(
        "--npus", default=os.environ.get("RETRACE_PERF_NPUS", "0,1,2,3,4,5,6,7")
    )
    parser.add_argument(
        "--from-initial",
        action="store_true",
        help="Explicitly restart from the original imported drafter if no checkpoint is available; no old updates are recovered",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.steps < args.warmup_updates + 2
        or args.warmup_updates < 0
        or args.blocks < 1
    ):
        parser.error(
            "Need >=2 measured updates, nonnegative warmup, and positive blocks"
        )
    run = Path(args.run).resolve()
    pool = run / "prompts.jsonl"
    metadata = (
        json.loads((run / "checkpoints/train_command.json").read_text())
        if (run / "checkpoints/train_command.json").is_file()
        else None
    )
    if args.from_initial:
        if args.checkpoint:
            parser.error("--from-initial and --checkpoint are mutually exclusive")
        if metadata is None:
            parser.error("Restart requires the original train_command.json")
        signature, start = metadata["signature"], 0
        draft = run / "initial/retrace"
        checkpoint = None
    else:
        pointer = run / "checkpoints/latest.txt"
        if not args.checkpoint and not pointer.is_file():
            parser.error(
                "No complete checkpoint found. Wait for an epoch checkpoint, or explicitly use --from-initial to discard unsaved progress."
            )
        checkpoint = Path(args.checkpoint or pointer.read_text().strip()).resolve()
        state = json.loads((checkpoint / "training_state.json").read_text())
        signature, start = state["signature"], int(state["step"])
        draft = checkpoint
        required = ["master_parameters.pt", "optimizer.pt"] + [
            f"rng_rank_{rank}.pt" for rank in range(signature["world_size"])
        ]
        if any(not (checkpoint / name).is_file() for name in required):
            parser.error(
                "Checkpoint is missing an optimizer, FP32 master or rank RNG file"
            )
    if not draft.is_dir() or signature["target_backend"] != "local":
        parser.error("This benchmark requires the existing local-target DFlash trainer")
    ids = args.npus.split(",")
    if (
        not all(x.isdecimal() for x in ids)
        or len(set(ids)) != len(ids)
        or len(ids) != signature["world_size"]
    ):
        parser.error(
            "NPU count must match the saved world size; use distinct physical IDs"
        )
    if sha256(pool) != signature["pool_sha256"]:
        parser.error("The original prompt pool hash differs from the checkpoint")
    stop = start + args.steps
    if stop > signature["total_steps"]:
        parser.error("Not enough updates remain for the requested benchmark")
    root = Path(__file__).resolve().parents[5]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (
        Path(args.output).resolve()
        if args.output
        else root / "output" / f"retrace_dflash_performance_{stamp}"
    )
    if output.exists():
        parser.error("Choose a new benchmark output directory")
    commands = {
        mode: make_command(
            signature,
            draft,
            checkpoint,
            pool,
            output / mode / "checkpoints",
            mode,
            blocks,
            stop,
            True,
        )
        for mode, blocks in (
            ("reference", signature["blocks_per_forward"]),
            ("cached", args.blocks),
        )
    }
    print(
        f"Start step {start}; stop at {stop}; original run stays at {run}", flush=True
    )
    for mode, command in commands.items():
        print(f"{mode}: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ | {
        "ASCEND_RT_VISIBLE_DEVICES": args.npus,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    }
    (output / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "run": str(run),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "initialization": "resume" if checkpoint else "restart_from_initial",
                "signature": signature,
                "args": vars(args),
                "commands": commands,
            },
            indent=2,
        )
        + "\n"
    )
    summaries = {}
    for mode, command in commands.items():
        run_branch(command, output / mode, env)
        rows = [
            json.loads(line)
            for line in (output / mode / "checkpoints/metrics.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        if [row["step"] for row in rows] != list(range(start + 1, stop + 1)):
            raise RuntimeError("Benchmark did not complete the same planned updates")
        summaries[mode] = summarize(rows, args.warmup_updates)
    baseline, cached = (
        summaries["reference"]["mean_step_seconds"],
        summaries["cached"]["mean_step_seconds"],
    )
    report = {
        "reference": summaries["reference"],
        "cached": summaries["cached"],
        "measured_update_speedup": baseline / cached,
        "remaining_updates_after_benchmark": signature["total_steps"] - stop,
        "estimated_remaining_days_at_measured_rate": (signature["total_steps"] - stop)
        * cached
        / 86400,
        "cached_mode_faster": cached < baseline,
        "limits": "Short paired NPU training benchmark; excludes startup/checkpoint I/O. Prompt lengths and acceptance may change later. Floating-point execution changes are not bitwise equivalence. Max-rank phase times need not add to total.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    saved = output / "cached/checkpoints" / f"step_{stop:06d}"
    resume = make_command(
        signature,
        saved,
        saved,
        pool,
        output / "cached/checkpoints",
        "cached",
        args.blocks,
        0,
        False,
    )
    script = output / "resume_fast.sh"
    script.write_text(resume_script(resume, args.npus, root))
    script.chmod(0o755)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Measured results: {output / 'summary.json'}", flush=True)
    print(
        f"Continue cached branch, keeping benchmark progress: bash {shlex.quote(str(script))}",
        flush=True,
    )
    if cached >= baseline:
        print(
            "Cached mode was not faster in this sample. Review phase times before selecting it.",
            flush=True,
        )


if __name__ == "__main__":
    main()
