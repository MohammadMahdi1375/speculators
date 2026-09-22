"""Supervise vLLM target NPUs and standard DFlash trainers for stored ReTrace."""

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ReTraceSpeculatorConfig
from .launch import stop_group
from .pretrained import prepare as prepare_weights
from .stored_data import prepare as prepare_data


def ids(value):
    result = value.split(",")
    if not all(x.isdecimal() for x in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Use distinct comma-separated integer NPU IDs")
    return result


def source_environment(root, inherited, visible_devices):
    """Resolve real source packages before editable-install namespace portions."""
    env = inherited.copy()
    paths = [root / "vllm", root / "vllm-ascend", root / "speculators/src"]
    env["PYTHONPATH"] = os.pathsep.join(
        [*map(str, paths), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    )
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(visible_devices)
    for name in ("NO_PROXY", "no_proxy"):
        env[name] = ",".join(filter(None, [env.get(name), "127.0.0.1", "localhost"]))
    return env


IMPORT_CHECK = r"""
import importlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
origins = {}
for name, relative in (
    ("vllm", "vllm/vllm/__init__.py"),
    ("vllm_ascend", "vllm-ascend/vllm_ascend/__init__.py"),
):
    module = importlib.import_module(name)
    filename = getattr(module, "__file__", None)
    expected = root / relative
    if filename is None or Path(filename).resolve() != expected.resolve():
        raise ImportError(f"{name} resolved to {filename!r}; expected {expected}. "
                          "Check the source checkout and Python import paths.")
    origins[name] = filename
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
origins["SamplingParams"] = SamplingParams.__module__
origins["EngineArgs"] = EngineArgs.__module__
print("Target import check passed: " + json.dumps(origins), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "target", "base-dflash", "data", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--server-npus", type=ids, default=ids("0,1"))
    parser.add_argument("--trainer-npus", type=ids, default=ids("2,3,4,5,6,7"))
    parser.add_argument("--port", type=int, default=8423)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=40000)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--tokens-per-worker", type=int, default=3072)
    parser.add_argument("--pairs-per-batch", type=int, default=16)
    parser.add_argument("--blocks-per-forward", type=int, default=4)
    parser.add_argument("--request-concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--boot-timeout", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-pool")
    parser.add_argument(
        "--cache-clean",
        action="store_true",
        help="Keep frozen clean features on disk; may require hundreds of GB",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if set(args.server_npus) & set(args.trainer_npus):
        parser.error("Target and trainer NPU IDs must be disjoint")
    if args.smoke:
        args.max_records, args.epochs, args.pairs_per_batch = 48, 1, 4
        args.tokens_per_worker = 1536
    for field in (
        "epochs",
        "max_records",
        "prompt_length",
        "response_length",
        "tokens_per_worker",
        "pairs_per_batch",
        "blocks_per_forward",
        "request_concurrency",
        "request_timeout",
        "boot_timeout",
        "lr",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.data_workers < 0 or not 0 < args.port < 65536:
        parser.error("Invalid data worker count or port")
    if args.tokens_per_worker < args.prompt_length + args.response_length:
        parser.error("Token budget must fit the largest selected stored row")
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    for path in (
        root / "speculators/scripts/launch_vllm.py",
        Path(args.target) / "config.json",
        Path(args.data) / "state.json",
        Path(args.base_dflash) / "config.json",
    ):
        if not path.is_file():
            parser.error(f"Required local file missing: {path}")
    if output.exists() and not args.resume:
        parser.error(
            "Use a new output directory, or --resume for this stored-response run"
        )
    if args.resume and not (output / "stored_run.json").is_file():
        parser.error("--resume needs an existing stored-response run")
    # Direct loopback requests must not go through the user's authenticated proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = f"http://127.0.0.1:{args.port}/v1"
    run = vars(args).copy()
    for name in ("resume", "dry_run", "boot_timeout"):
        run.pop(name)
    run.update(
        output=str(output),
        root=str(root),
        endpoint=endpoint,
        served_model="retrace-stored-target",
        stored_data=str(output / "stored_data"),
        initial_checkpoint=str(output / "initial/retrace"),
        protocol="stored_pair_kl_v1",
        table2_reproduction_validated=False,
        response_source="stored Arrow tokens; no complete fresh continuations",
        pair_rule="actual greedy verification; require committed-prefix agreement",
        objective="stock DFlash clean-target KL, exp(-proposal_index/4)",
        batch_unit="packed tokens per worker; not global 32 prompts",
        conditioning_history="one sampled unconditioned source + aligned conditioned successor",
        checkpoint_cadence="each epoch; stock Trainer checkpoint format",
    )
    if args.resume:
        previous = json.loads((output / "stored_run.json").read_text())
        if previous != run:
            parser.error(
                "Resume configuration differs. Use this run's resume_stored.sh"
            )
    if args.dry_run:
        print(json.dumps(run, indent=2))
        print("Dry run only; no server, data preparation, or training launched.")
        return
    server_env = source_environment(root, os.environ, args.server_npus)
    trainer_env = source_environment(root, os.environ, args.trainer_npus)
    print("Checking vLLM package origins and SamplingParams before launch", flush=True)
    subprocess.run(
        [sys.executable, "-c", IMPORT_CHECK, str(root)],
        cwd=root / "speculators",
        env=server_env,
        check=True,
        timeout=180,
    )
    with socket.socket() as check:
        try:
            check.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error(f"Port {args.port} is already in use")
    if not args.resume:
        output.mkdir(parents=True)
        prepare_weights(args.base_dflash, args.target, output / "initial")
        config = ReTraceSpeculatorConfig.from_pretrained(output / "initial/retrace")
        manifest = prepare_data(
            args.data,
            output / "stored_data",
            count=args.max_records,
            vocabulary=config.target_vocab_size,
            block_size=config.block_size,
            prompt_limit=args.prompt_length,
            response_limit=args.response_length,
            seed=args.seed,
            prompt_pool=args.prompt_pool,
        )
        clean_bytes = (
            manifest["total_tokens"]
            * (len(config.aux_hidden_state_layer_ids) + 1)
            * config.transformer_layer_config.hidden_size
            * 2
        )
        print(
            f"Clean feature cache estimate: {clean_bytes / 2**30:.1f} GiB. "
            f"Retention {'enabled' if args.cache_clean else 'disabled (DFlash delete mode)'}. ",
            flush=True,
        )
        if args.cache_clean:
            if shutil.disk_usage(output).free < clean_bytes * 1.15:
                raise ValueError(
                    "Insufficient disk space for requested clean feature cache"
                )
        (output / "stored_run.json").write_text(json.dumps(run, indent=2) + "\n")
    else:
        config = ReTraceSpeculatorConfig.from_pretrained(output / "initial/retrace")
    max_len = args.prompt_length + args.response_length + config.block_size + 1
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
        str(output / "transfer"),
        "--target-layer-ids",
        *map(str, config.aux_hidden_state_layer_ids),
        "--",
        "--api-server-count",
        "1",
        "--renderer-num-workers",
        "1",
        # Target KV-cache pages: the pinned Ascend backend supports 128.
        # This is independent of the pretrained drafter's config.block_size.
        "--block-size",
        "128",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        run["served_model"],
        "--data-parallel-size",
        str(len(args.server_npus)),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_len),
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "64",
        "--gpu-memory-utilization",
        "0.85",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
    ]
    trainer_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(args.trainer_npus)),
        "-m",
        "speculators.models.retrace.stored_train",
        "--run-config",
        str(output / "stored_run.json"),
    ]
    if args.resume:
        trainer_command.append("--resume")
    # Bash wrapper restores CANN; the run-specific environment is inherited.
    wrapper = root / "speculators/examples/train/train_retrace_dflash_stored_vllm.sh"
    resume_args = [x for x in sys.argv[1:] if x != "--resume"]
    (output / "resume_stored.sh").write_text(
        "#!/usr/bin/env bash\nset -eo pipefail\nexec bash "
        + shlex.quote(str(wrapper))
        + " "
        + shlex.join(resume_args + ["--resume"])
        + "\n"
    )
    server = trainer = None
    log_path = output / "vllm_server.log"

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with log_path.open("a") as log:
            print("Server: " + shlex.join(server_command), flush=True)
            server = subprocess.Popen(
                server_command,
                env=server_env,
                cwd=root / "speculators",
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            start = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited: read {log_path}")
                if time.monotonic() - start > args.boot_timeout:
                    raise TimeoutError(f"vLLM startup timeout: {log_path}")
                try:
                    with opener.open(endpoint + "/models", timeout=3) as response:
                        advertised = json.load(response)["data"]
                    if not any(m["id"] == run["served_model"] for m in advertised):
                        raise RuntimeError("Unexpected model served on target port")
                    break
                except (urllib.error.URLError, TimeoutError):
                    print(
                        f"Waiting for vLLM ({time.monotonic() - start:.0f}s): {log_path}",
                        flush=True,
                    )
                    time.sleep(5)
            print("Trainer: " + shlex.join(trainer_command), flush=True)
            trainer = subprocess.Popen(
                trainer_command,
                env=trainer_env,
                cwd=root / "speculators",
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
