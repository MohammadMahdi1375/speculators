"""Prepare and supervise pretrained ReTrace training in the pinned NPU environment."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .utils import stop_group
from .pretrained import inspect_source, prepare, sha256
from .prompt_pool import prepare_pool
from .utilization_resume import preflight_resume, resolve_run

BASE = Path("/home/n84449292/m84379596")


def npu_ids(text):
    ids = text.split(",")
    if not ids or not all(x.isdecimal() for x in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(
            "Use a comma-separated list of distinct NPU IDs"
        )
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(BASE / "DFlash/vLLM_NPU_spec_main"))
    parser.add_argument(
        "--target",
        default=str(
            BASE
            / "Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
        ),
    )
    parser.add_argument(
        "--base-dflash", default=str(BASE / "Huggingface/Qwen3-4B-DFlash-b16")
    )
    parser.add_argument(
        "--data",
        default=str(
            BASE / "Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pool", help="Reuse a pool previously made by retrace.prompt_pool"
    )
    parser.add_argument(
        "--resume", help="Exact step checkpoint, with optimizer and RNG files"
    )
    parser.add_argument("--resume-run", help="Continue the latest completed checkpoint from this run in a new output directory")
    parser.add_argument("--target-backend", choices=("local",), default="local")
    parser.add_argument(
        "--trainer-npus", type=npu_ids, default=npu_ids("8,9,10,11,12,13,14,15")
    )
    parser.add_argument("--max-prompts", type=int, default=40000)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--global-prompt-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--blocks-per-forward", type=int, default=16)
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--rollout-scheduler", choices=("cohort", "continuous"), default="continuous")
    parser.add_argument(
        "--performance-mode", choices=("reference", "cached"), default="cached"
    )
    parser.add_argument("--trace-storage", choices=("cpu", "device"), default="device")
    parser.add_argument("--profile-performance", action="store_true")
    parser.add_argument(
        "--attention-backend", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument(
        "--target-norm", choices=("reference", "npu"), default="npu"
    )
    parser.add_argument(
        "--work-distribution", choices=("static", "dynamic"), default="dynamic"
    )
    parser.add_argument("--allow-performance-change", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Optimizer-step save interval; 0 saves at the end of each epoch",
    )
    parser.add_argument("--disable-conditioning", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare-run", help="Compare this mode with the saved execution settings of a local run")
    parser.add_argument("--compare-steps", type=int, default=6)
    parser.add_argument("--compare-warmup", type=int, default=2)
    args = parser.parse_args()
    if args.compare_run:
        from .utilization_compare import compare
        try:
            compare(args)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        return
    if args.resume_run:
        if args.resume or args.smoke:
            parser.error("Use --resume-run separately from --resume and --smoke")
        try:
            checkpoint, existing_pool = resolve_run(args.resume_run)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        args.resume = str(checkpoint)
        args.pool = args.pool or str(existing_pool)
        if Path(args.output).resolve() == Path(args.resume_run).resolve():
            parser.error("--resume-run requires a new output directory; the source run is preserved")
    if args.rollout_scheduler == "continuous" and args.performance_mode != "cached":
        parser.error("Continuous rollouts require cached execution")
    if args.smoke:
        if args.resume:
            parser.error("A smoke run cannot resume a full training run")
        args.max_prompts, args.global_prompt_batch = 12, 4
        args.prompt_length, args.response_length = 256, 64
        args.max_steps, args.save_every = 3, 3
    if (
        min(
            args.max_prompts,
            args.prompt_length,
            args.response_length,
            args.global_prompt_batch,
            args.epochs,
            args.blocks_per_forward,
            args.rollout_batch_size,
            args.lr,
        )
        <= 0
        or min(args.max_steps, args.save_every) < 0
    ):
        parser.error("Invalid training counts or rate")
    config, info = inspect_source(args.base_dflash, args.target)
    if (
        config.block_size != 16
        or config.transformer_layer_config.num_hidden_layers != 5
    ):
        parser.error(
            "This Qwen3-4B recipe requires the five-layer DFlash-b16 checkpoint"
        )
    print(
        json.dumps({k: v for k, v in info.items() if k != "tensor_shapes"}, indent=2),
        flush=True,
    )
    print(f"Local target + Speculators on logical NPUs {','.join(args.trainer_npus)}; "
          f"scheduler={args.rollout_scheduler}, active slots/worker={args.rollout_batch_size}, "
          f"trace storage={args.trace_storage}, blocks/forward={args.blocks_per_forward}", flush=True)
    output = Path(args.output).absolute()
    root = Path(args.root).absolute()
    if output.exists() and not args.resume:
        parser.error("Use a new output directory; existing runs are preserved")
    pool = Path(args.pool).absolute() if args.pool else output / "prompts.jsonl"
    if args.resume and not pool.is_file():
        parser.error(
            "Resume requires the existing prompt pool (--pool if it is elsewhere)"
        )
    if args.pool or args.resume:
        manifest = json.loads(pool.with_suffix(".manifest.json").read_text())
        if (
            manifest["counts"]["exported"] != args.max_prompts
            or manifest["prompt_length"] != args.prompt_length
        ):
            parser.error("The reused prompt pool has different count/length settings")
        if manifest["output_sha256"] != sha256(pool):
            parser.error("The reused prompt pool differs from its manifest")
    if args.resume:
        try:
            state, changes, total = preflight_resume(args, pool)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        print(f"Checkpoint step: {state['step']}/{total}; restored state includes FP32 masters and optimizer", flush=True)
        if changes:
            print("Resume execution changes (not bitwise-equivalent): " + json.dumps(changes), flush=True)
        if state["step"] >= total:
            print("This checkpoint already completed all configured updates; no processes started.")
            return
    initialization = (
        Path(args.resume).absolute() if args.resume else output / "initial/retrace"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(args.trainer_npus)),
        "-m",
        "speculators.models.retrace.paper_recipe.utilization_train",
        "--target",
        args.target,
        "--draft",
        str(initialization),
        "--pool",
        str(pool),
        "--output",
        str(output / "checkpoints"),
        "--block-size",
        "16",
        "--target-layer-ids",
        *map(str, config.aux_hidden_state_layer_ids),
    ]
    for field in (
        "prompt_length",
        "response_length",
        "global_prompt_batch",
        "epochs",
        "max_steps",
        "stop_after",
        "blocks_per_forward",
        "rollout_batch_size",
        "rollout_scheduler",
        "lr",
        "save_every",
        "performance_mode",
        "trace_storage",
        "attention_backend",
        "target_norm",
        "work_distribution",
    ):
        command += ["--" + field.replace("_", "-"), str(getattr(args, field))]
    if args.resume:
        command += ["--resume", args.resume]
    if args.disable_conditioning:
        command += ["--disable-conditioning"]
    for name in ("profile_performance", "allow_performance_change"):
        if getattr(args, name):
            command += ["--" + name.replace("_", "-")]
    server_command = None
    print("Trainer: " + shlex.join(command), flush=True)
    if args.dry_run:
        print("Dry run: no training process or target server was started.")
        return
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    (output / ("resume_launch.json" if args.resume else "launch.json")).write_text(
        json.dumps(
            {"args": vars(args), "trainer": command, "server": server_command}, indent=2
        )
    )
    if not args.resume:
        print(
            "Importing all pretrained backbone tensors and creating the unchanged DFlash baseline",
            flush=True,
        )
        prepare(args.base_dflash, args.target, output / "initial")
        if not args.pool:
            print("Preparing the random prompt-only pool on CPU", flush=True)
            prepare_pool(
                args.data, args.target, pool, args.max_prompts, args.prompt_length
            )
    trainer = None
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(map(str, (
            root / "vllm", root / "vllm-ascend", root / "speculators/src",
            root / "speculators/hs_connectors/src")))
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        if args.attention_backend == "sdpa" or args.target_norm == "npu":
            # Run a device guard before the long job, without loading a target.

            stamp = str(time.time_ns())
            check = [
                sys.executable,
                "-m",
                "speculators.models.retrace.paper_recipe.execution_check",
                "--device",
                "npu:0",
                "--dtype",
                "bfloat16",
                "--target-norm",
                args.target_norm,
                "--output",
                str(output / f"operator_check_{stamp}.json"),
            ]
            subprocess.run(
                check,
                env=env | {"ASCEND_RT_VISIBLE_DEVICES": ",".join(args.trainer_npus)},
                cwd=root / "speculators", check=True,
            )
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(args.trainer_npus)
        trainer = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=root / "speculators",
            start_new_session=True,
        )
        with (output / "trainer.log").open("a", buffering=1) as log:
            for line in trainer.stdout:
                print(line, end="", flush=True)
                log.write(line)
        code = trainer.wait()
        if code:
            raise SystemExit(code)
        print(
            f"Training finished. Checkpoint pointer: {output / 'checkpoints/latest.txt'}",
            flush=True,
        )
    finally:
        stop_group(trainer)


if __name__ == "__main__":
    main()
