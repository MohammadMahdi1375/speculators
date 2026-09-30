"""Supervise vLLM target NPUs and standard DFlash trainers for stored ReTrace."""

import argparse
import hashlib
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

from transformers import AutoTokenizer

from ..config import ReTraceSpeculatorConfig
from ..paper_recipe.pretrained import prepare as prepare_weights
from .stored_data import prepare as prepare_data


def stop_group(process):
    if process is None:
        return
    # Own child process group only. Do not kill other training or serving jobs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def ids(value):
    result = value.split(",")
    if not all(x.isdecimal() for x in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Use distinct comma-separated integer NPU IDs")
    return result


def source_environment(root, inherited, visible_devices):
    """Resolve real source packages before editable-install namespace portions."""
    env = inherited.copy()
    paths = [root / "vllm", root / "vllm-ascend", root / "speculators/src", root / "speculators/hs_connectors/src"]
    env["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([*map(str, paths), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))])
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
from acl.rt import memcpy
origins["SamplingParams"] = SamplingParams.__module__
origins["EngineArgs"] = EngineArgs.__module__
print("Target import check passed: " + json.dumps(origins), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "target", "base-dflash", "data", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--server-npus", type=ids, default=ids("8,9"))
    parser.add_argument("--trainer-npus", type=ids, default=ids("10,11,12,13,14,15"))
    parser.add_argument("--port", type=int, default=8623)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=40000)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--global-prompt-batch", type=int, default=32)
    parser.add_argument("--chain-rounds", type=int, default=4)
    parser.add_argument("--chains-per-prompt", type=int, default=1)
    parser.add_argument("--blocks-per-forward", type=int, default=4)
    parser.add_argument("--request-concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--boot-timeout", type=int, default=1800)
    parser.add_argument("--server-batched-tokens", type=int, default=16384)
    parser.add_argument("--server-max-seqs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-pool")
    parser.add_argument(
        "--cache-clean",
        action="store_true",
        help="Keep frozen clean features on disk; may require hundreds of GB",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if set(args.server_npus) & set(args.trainer_npus):
        parser.error("Target and trainer NPU IDs must be disjoint")
    if args.smoke:
        args.max_records, args.epochs = args.global_prompt_batch * 3, 1
    for field in (
        "epochs",
        "max_records",
        "prompt_length",
        "response_length",
        "global_prompt_batch",
        "chain_rounds",
        "chains_per_prompt",
        "blocks_per_forward",
        "request_concurrency",
        "request_timeout",
        "boot_timeout",
        "server_batched_tokens",
        "server_max_seqs",
        "lr",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.data_workers < 0 or not 0 < args.port < 65536:
        parser.error("Invalid data worker count or port")
    if args.chain_rounds < 2:
        parser.error("At least two rounds per chain are required")
    if args.global_prompt_batch < len(args.trainer_npus) or args.max_records % args.global_prompt_batch:
        parser.error("Use complete global batches and at least one prompt per trainer")
    if args.server_batched_tokens < args.prompt_length + args.response_length + 17:
        parser.error("Server prefill token budget must fit the largest branch")
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    for path in (
        root / "speculators/scripts/launch_vllm.py",
        Path(args.target) / "config.json",
        Path(args.data) / "state.json",
        Path(args.base_dflash) / "config.json",
    ):
        if not path.is_file():
            parser.error(f"Required local file missing: {path}")
    if output.exists():
        parser.error("This launcher starts fresh from pretrained DFlash; use a new output directory")
    # Direct loopback requests must not go through the user's authenticated proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = f"http://127.0.0.1:{args.port}/v1"
    run = vars(args).copy()
    for name in ("dry_run", "boot_timeout"):
        run.pop(name)
    run.update(
        output=str(output),
        root=str(root),
        endpoint=endpoint,
        served_model="retrace-stored-ce-target",
        stored_data=str(output / "stored_data"),
        initial_checkpoint=str(output / "initial/retrace"),
        protocol="stored_chain_clean_ce_v2",
        table2_reproduction_validated=False,
        response_source="stored Arrow tokens; no complete fresh continuations",
        pair_rule="actual greedy branch verification; exact committed-prefix agreement including correction/bonus",
        objective="hard stored-token CE, exp(-proposal_index/4); per-prompt valid-label count then global prompt mean",
        batch_unit="exact global prompt count, packed per rank without record truncation",
        conditioning_history="bounded recurrent chains with detached one-round memory; not full online continuations",
        checkpoint_cadence="each epoch; stock Trainer checkpoint format",
        planned_updates=args.epochs * (args.max_records // args.global_prompt_batch),
        updates_per_epoch=args.max_records // args.global_prompt_batch,
        maximum_branch_requests_per_update=args.global_prompt_batch * args.chains_per_prompt * (args.chain_rounds - 1),
        clean_requests_per_update=args.global_prompt_batch,
        beta_max=1.0,
        beta_warmup_updates=50,
        optimizer="AdamW",
        weight_decay=0.01,
        lr_warmup_ratio=0.05,
        seed_author_row_ids_known=False,
        training_verification="greedy; author training sampling temperature unspecified",
        outstanding_differences=["stored response selection", "bounded anchor/chain sampler", "eight Ascend NPUs with DDP vs 32 A800 with FSDP", "unvalidated native evaluation"],
    )
    inspected = [*Path(__file__).parent.glob("*.py"),
                 *Path(__file__).parent.parent.joinpath("paper_recipe").glob("*.py")]
    run["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in inspected if path.is_relative_to(root)}
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
    output.mkdir(parents=True)
    prepare_weights(args.base_dflash, args.target, output / "initial")
    config = ReTraceSpeculatorConfig.from_pretrained(output / "initial/retrace")
    if config.block_size != 16 or config.transformer_layer_config.num_hidden_layers != 5:
        raise ValueError("This recipe requires the public five-layer DFlash-b16 checkpoint")
    config.training_objective = "clean_block_ce"
    config.stored_training_recipe = "stored_chain_clean_ce_v2"
    config.retrace_enabled = True
    config.retrace_beta_max, config.retrace_warmup_steps = 1.0, 50
    config.save_pretrained(output / "initial/retrace")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    empty_think_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
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
            empty_think_ids=empty_think_ids,
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
    if args.cache_clean and shutil.disk_usage(output).free < clean_bytes * 1.15:
        raise ValueError("Insufficient disk space for requested clean feature cache")
    (output / "stored_run.json").write_text(json.dumps(run, indent=2) + "\n")
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
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_len),
        "--max-num-batched-tokens",
        str(args.server_batched_tokens),
        "--max-num-seqs",
        str(args.server_max_seqs),
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
        "speculators.models.retrace.stored_recipe.stored_train",
        "--run-config",
        str(output / "stored_run.json"),
    ]
    (output / "commands.json").write_text(json.dumps({"server": server_command, "trainer": trainer_command}, indent=2) + "\n")
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
