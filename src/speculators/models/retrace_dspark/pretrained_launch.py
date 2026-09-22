"""Launch isolated pretrained DSpark + ReTrace on a dedicated NPU group."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from .pretrained import inspect_source, prepare, sha256
from .prompt_pool import prepare_pool

BASE = Path("/home/n84449292/m84379596")


def npu_ids(text):
    ids = text.split(",")
    if not ids or not all(x.isdecimal() for x in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("Use distinct comma-separated NPU IDs")
    return ids


def stop_group(process):
    # Only this launcher's new session, never a pre-existing DFlash job.
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


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
        "--base-dspark", default=str(BASE / "Huggingface/Qwen3-4B-DSpark-block7")
    )
    parser.add_argument(
        "--data",
        default=str(
            BASE / "Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pool", help="Reuse an existing prepared prompt pool and its manifest"
    )
    parser.add_argument(
        "--resume",
        help="Step directory containing optimizer, FP32 masters and rank RNG states",
    )
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
    parser.add_argument("--blocks-per-forward", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="0 saves at every epoch end; positive values select a step interval",
    )
    parser.add_argument("--disable-conditioning", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        if args.resume:
            parser.error("Smoke cannot resume a full run")
        args.max_prompts = 3 * len(args.trainer_npus)
        args.global_prompt_batch = len(args.trainer_npus)
        args.prompt_length, args.response_length = 256, 64
        args.epochs, args.max_steps, args.save_every = 1, 3, 0
    if (
        min(
            args.max_prompts,
            args.prompt_length,
            args.response_length,
            args.global_prompt_batch,
            args.epochs,
            args.blocks_per_forward,
            args.lr,
        )
        <= 0
    ):
        parser.error("Training counts and LR must be positive")
    if min(args.max_steps, args.stop_after, args.save_every) < 0:
        parser.error("Step limits and save interval must be nonnegative")
    # Set visibility before any accelerator setup. Source import and data prep run on CPU.
    visible = ",".join(args.trainer_npus)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible
    config, info = inspect_source(args.base_dspark, args.target)
    tl = config.transformer_layer_config
    if config.block_size != 7 or tl.num_hidden_layers != 5 or config.markov_rank != 256:
        parser.error(
            "This recipe requires the five-layer, rank-256 DSpark-block7 checkpoint"
        )
    print(
        json.dumps({k: v for k, v in info.items() if k != "tensor_shapes"}, indent=2),
        flush=True,
    )
    output = Path(args.output).absolute()
    if output.exists() and not args.resume:
        parser.error("Use a new output directory; existing runs are preserved")
    pool = Path(args.pool).absolute() if args.pool else output / "prompts.jsonl"
    if args.pool or args.resume:
        manifest = json.loads(pool.with_suffix(".manifest.json").read_text())
        if (
            manifest["counts"]["exported"] != args.max_prompts
            or manifest["prompt_length"] != args.prompt_length
            or manifest["output_sha256"] != sha256(pool)
        ):
            parser.error(
                "Prompt pool count, length or hash differs from its manifest/settings"
            )
    initial = (
        Path(args.resume).absolute()
        if args.resume
        else output / "initial/retrace_dspark"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(args.trainer_npus)),
        "-m",
        "speculators.models.retrace_dspark.train_pretrained",
        "--target",
        args.target,
        "--draft",
        str(initial),
        "--pool",
        str(pool),
        "--output",
        str(output / "checkpoints"),
        "--block-size",
        "7",
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
        "lr",
        "save_every",
    ):
        command += ["--" + field.replace("_", "-"), str(getattr(args, field))]
    if args.resume:
        command += ["--resume", args.resume]
    if args.disable_conditioning:
        command += ["--disable-conditioning"]
    print(
        f"Physical NPU visibility: {visible}; worker-local IDs 0..{len(args.trainer_npus) - 1}",
        flush=True,
    )
    print("Trainer: " + shlex.join(command), flush=True)
    if args.dry_run:
        print("Dry run: no files written and no training process started.")
        return
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    (output / ("resume_launch.json" if args.resume else "launch.json")).write_text(
        json.dumps(
            {
                "args": vars(args),
                "trainer": command,
                "visible_npus": visible,
                "target_backend": "local HF; frozen target per worker",
                "rendezvous": "torchrun --standalone; independent random local port",
            },
            indent=2,
        )
        + "\n"
    )
    if not args.resume:
        print(
            "Importing pretrained DSpark backbone, Markov and confidence tensors on CPU",
            flush=True,
        )
        prepare(args.base_dspark, args.target, output / "initial")
        if not args.pool:
            print("Preparing a deterministic prompt-only pool on CPU", flush=True)
            prepare_pool(
                args.data, args.target, pool, args.max_prompts, args.prompt_length
            )
    env = os.environ | {
        "ASCEND_RT_VISIBLE_DEVICES": visible,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
    }
    trainer = None
    try:
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


if __name__ == "__main__":
    main()
