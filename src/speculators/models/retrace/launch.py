"""Supervise a file-connector vLLM server and online ReTrace torchrun workers."""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("/home/n84449292/m84379596")
    parser.add_argument(
        "--root",
        default=os.environ.get("RETRACE_ROOT", str(base / "DFlash/vLLM_NPU_spec_main")),
    )
    parser.add_argument(
        "--target",
        default=os.environ.get(
            "RETRACE_TARGET",
            str(
                base
                / "Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
            ),
        ),
    )
    parser.add_argument(
        "--data",
        default=str(
            base / "Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-npus", default="0,1")
    parser.add_argument("--trainer-npus", default="2,3,4,5,6,7")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--boot-timeout", type=int, default=1800)
    parser.add_argument(
        "--target-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33]
    )
    # DSpARK's sample_from_anchor=True, block_size=8 baseline has K=8.
    parser.add_argument("--block-size", type=int, default=9)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=40000)
    parser.add_argument("--grad-accum-rounds", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--draft", help="Independent ReTrace weights only; new optimizer"
    )
    parser.add_argument("--disable-conditioning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    server_ids = args.server_npus.split(",")
    trainer_ids = args.trainer_npus.split(",")
    for ids in (server_ids, trainer_ids):
        if not all(x.isdecimal() for x in ids) or len(ids) != len(set(ids)):
            parser.error("NPU lists must contain distinct integer IDs")
    if set(server_ids) & set(trainer_ids):
        parser.error("Target server and trainer NPUs must be disjoint")
    if not 0 < args.port < 65536 or args.boot_timeout <= 0:
        parser.error("Invalid port or boot timeout")
    for key in (
        "prompt_length",
        "response_length",
        "max_steps",
        "epochs",
        "max_prompts",
        "grad_accum_rounds",
        "save_every",
        "lr",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.block_size < 2:
        parser.error("block-size must include one anchor and at least one proposal")
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.exists():
        parser.error("Use a new output directory; existing runs are preserved")
    max_length = args.prompt_length + args.response_length + args.block_size + 32
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
        *map(str, args.target_layer_ids),
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "retrace-target",
        "--data-parallel-size",
        str(len(server_ids)),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_length),
        "--max-num-batched-tokens",
        str(max_length),
        "--max-num-seqs",
        str(max(8, len(trainer_ids))),
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",
    ]
    trainer_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(trainer_ids)),
        "-m",
        "speculators.models.retrace.train",
        "--target",
        args.target,
        "--data",
        args.data,
        "--output",
        str(output / "checkpoints"),
        "--vllm-endpoint",
        f"http://127.0.0.1:{args.port}/v1",
        "--served-model-name",
        "retrace-target",
        "--target-layer-ids",
        *map(str, args.target_layer_ids),
        "--block-size",
        str(args.block_size),
        "--prompt-length",
        str(args.prompt_length),
        "--response-length",
        str(args.response_length),
        "--max-steps",
        str(args.max_steps),
        "--epochs",
        str(args.epochs),
        "--max-prompts",
        str(args.max_prompts),
        "--grad-accum-rounds",
        str(args.grad_accum_rounds),
        "--save-every",
        str(args.save_every),
        "--lr",
        str(args.lr),
        "--check-greedy",
    ]
    if args.draft:
        trainer_command += ["--draft", args.draft]
        config = json.loads((Path(args.draft) / "config.json").read_text())
        if (
            config.get("speculators_model_type") != "retrace"
            or config["aux_hidden_state_layer_ids"] != args.target_layer_ids
            or config["block_size"] != args.block_size
        ):
            parser.error(
                "Checkpoint identity, target layers and block size must match the launcher"
            )
    if args.disable_conditioning:
        trainer_command += ["--disable-conditioning"]
    for name, command in (("Server", server_command), ("Trainer", trainer_command)):
        print(f"{name}: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    if (
        not Path(args.data).exists()
        or not (Path(args.target) / "config.json").is_file()
    ):
        parser.error("Target config or dataset path is missing")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True)
    (output / "launch.json").write_text(
        json.dumps(
            {"args": vars(args), "server": server_command, "trainer": trainer_command},
            indent=2,
        )
    )
    env = dict(os.environ, PYTHONUNBUFFERED="1", VLLM_USE_V2_MODEL_RUNNER="0")
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost,::1"
    server = trainer = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with (output / "vllm_server.log").open("w") as log:
            server = subprocess.Popen(
                server_command,
                cwd=root / "speculators",
                env=dict(env, ASCEND_RT_VISIBLE_DEVICES=args.server_npus),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            start = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Target server exited; inspect {log.name}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/v1/models", timeout=5
                    ) as response:
                        models = json.load(response)
                    if "retrace-target" in [item["id"] for item in models["data"]]:
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() - start > args.boot_timeout:
                    raise TimeoutError(
                        f"Target server startup timed out; inspect {log.name}"
                    )
                print(
                    f"Waiting for target server ({int(time.monotonic() - start)}s); log: {log.name}",
                    flush=True,
                )
                time.sleep(5)
            trainer = subprocess.Popen(
                trainer_command,
                cwd=root / "speculators",
                env=dict(env, ASCEND_RT_VISIBLE_DEVICES=args.trainer_npus),
                start_new_session=True,
            )
            code = trainer.wait()
            if code:
                raise SystemExit(code)
    finally:
        stop_group(trainer)
        stop_group(server)


if __name__ == "__main__":
    main()
