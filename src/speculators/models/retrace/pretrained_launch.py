"""Prepare and supervise pretrained ReTrace training in the pinned NPU environment."""

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .launch import stop_group
from .pretrained import inspect_source, prepare, sha256
from .prompt_pool import prepare_pool

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
    parser.add_argument("--target-backend", choices=("local", "vllm"), default="local")
    parser.add_argument(
        "--trainer-npus", type=npu_ids, default=npu_ids("0,1,2,3,4,5,6,7")
    )
    parser.add_argument("--server-npus", type=npu_ids, default=npu_ids("0,1"))
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--boot-timeout", type=int, default=1800)
    parser.add_argument("--max-prompts", type=int, default=40000)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--global-prompt-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--blocks-per-forward", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument(
        "--performance-mode", choices=("reference", "cached"), default="reference"
    )
    parser.add_argument("--trace-storage", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--profile-performance", action="store_true")
    parser.add_argument(
        "--attention-backend", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument(
        "--target-norm", choices=("reference", "npu"), default="reference"
    )
    parser.add_argument(
        "--work-distribution", choices=("static", "dynamic"), default="static"
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
    args = parser.parse_args()
    if args.target_backend == "vllm" and set(args.trainer_npus) & set(args.server_npus):
        parser.error(
            "For vLLM use disjoint NPUs, for example --trainer-npus 2,3,4,5,6,7 --server-npus 0,1"
        )
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
        "speculators.models.retrace.train_pretrained",
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
    if args.target_backend == "vllm":
        length = args.prompt_length + args.response_length + 48
        server_command = [
            sys.executable,
            str(root / "speculators/scripts/launch_vllm.py"),
            "train",
            args.target,
            "--provenance-dir",
            str(output / "provenance"),
            "--hidden-states-backend",
            "file",
            "--hidden-states-path",
            str(output / "hidden_states"),
            "--target-layer-ids",
            *map(str, config.aux_hidden_state_layer_ids),
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--served-model-name",
            "retrace-target",
            "--data-parallel-size",
            str(len(args.server_npus)),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            str(length),
            "--max-num-batched-tokens",
            str(length),
            "--max-num-seqs",
            str(max(8, len(args.trainer_npus))),
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
        ]
        command += [
            "--vllm-endpoint",
            f"http://127.0.0.1:{args.port}/v1",
            "--served-model-name",
            "retrace-target",
        ]
    print("Trainer: " + shlex.join(command), flush=True)
    if server_command:
        print("Server: " + shlex.join(server_command), flush=True)
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
    server = trainer = None
    server_log = None
    try:
        env = os.environ.copy()
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        if args.attention_backend == "sdpa" or args.target_norm == "npu":
            # Run a device guard before the long job, without loading a target.
            from .performance_benchmark import run_branch

            stamp = str(time.time_ns())
            check = [
                sys.executable,
                "-m",
                "speculators.models.retrace.execution_check",
                "--device",
                "npu:0",
                "--dtype",
                "bfloat16",
                "--target-norm",
                args.target_norm,
                "--output",
                str(output / f"operator_check_{stamp}.json"),
            ]
            run_branch(
                check,
                output / f"operator_check_{stamp}",
                env | {"ASCEND_RT_VISIBLE_DEVICES": ",".join(args.trainer_npus)},
            )
        if server_command:
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", args.port))
                except OSError as exc:
                    raise RuntimeError(
                        f"Port {args.port} is already in use; choose --port"
                    ) from exc
            server_log = (output / "vllm_server.log").open("a")
            server_env = env | {"ASCEND_RT_VISIBLE_DEVICES": ",".join(args.server_npus)}
            server = subprocess.Popen(
                server_command,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Target server exited; read {output / 'vllm_server.log'}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() - started > args.boot_timeout:
                    raise TimeoutError("Target server startup timeout")
                print(
                    f"Waiting for target server ({time.monotonic() - started:.0f}s)",
                    flush=True,
                )
                time.sleep(5)
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(args.trainer_npus)
        trainer = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
        stop_group(server)
        if server_log:
            server_log.close()


if __name__ == "__main__":
    main()
