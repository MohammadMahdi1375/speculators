#!/usr/bin/env python3
"""Bounded whole-pipeline tracing, using vLLM-Ascend's installed profiler.

This is a diagnostic client, not an inference implementation or a benchmark.
It reuses the selector diagnostic's standard-library HTTP/lifecycle helpers.
No Torch is imported in this parent process. The serving child uses the normal
compiled launcher, full proposals, target verification, and production parity.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile


def load_helpers():
    path = Path(__file__).with_name("profile_server.py")
    if not path.is_file():
        raise ValueError("Install the compiled selector profiler first: profile_server.py is missing")
    spec = importlib.util.spec_from_file_location("_compiled_profile_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profiler_config(directory, steps):
    return {
        "profiler": "torch", "torch_profiler_dir": str(directory),
        "ignore_frontend": True, "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False, "torch_profiler_record_shapes": False,
        "torch_profiler_dump_cuda_time_total": False,
        # WorkerProfiler.step runs before model execution. Skip request prefill
        # and begin at the second engine iteration, then stop after N iterations.
        "delay_iterations": 2, "max_iterations": steps,
    }


def profile_control(opener, base_url, action):
    if action not in {"start_profile", "stop_profile"}:
        raise ValueError("Invalid profile action")
    request = urllib.request.Request(base_url + "/" + action, data=b"", method="POST")
    try:
        with opener.open(request, timeout=600) as response:
            # These routes normally return an EMPTY HTTP 200, not JSON.
            response.read()
    except urllib.error.HTTPError as error:
        message = error.read(8192).decode("utf-8", errors="replace")
        raise RuntimeError(f"{action}: HTTP {error.code}: {message}") from error


def readable_trace_files(directory):
    # Omit potentially large raw binary device dumps. These analyzed artifacts
    # include device kernels, CPU scopes, operator summaries, and the timeline.
    return sorted(path for path in directory.rglob("*") if path.is_file()
                  and not path.is_symlink()
                  and (path.suffix in {".json", ".csv", ".log"}
                       or path.name.endswith(".json.gz")))


def has_kernel_rows(paths):
    for path in paths:
        if path.name != "kernel_details.csv":
            continue
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            if next(csv.DictReader(stream), None) is not None:
                return True
    return False


def source_digests(root, helpers):
    paths = [
        "speculators/examples/serve_dflash_prefix_compiled.sh",
        "speculators/examples/trace_dflash_prefix_compiled.sh",
        "speculators/scripts/dflash_prefix_compiled/trace_server.py",
        "speculators/scripts/dflash_prefix_compiled/profile_server.py",
        "speculators/src/speculators/models/dflash_prefix/compiled_inference.py",
        "speculators/src/speculators/models/dflash_prefix/compiled_selector.py",
        "vllm/vllm/config/profiler.py",
        "vllm/vllm/profiler/wrapper.py",
        "vllm/vllm/model_executor/models/qwen3_dflash_prefix.py",
        "vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py",
        "vllm-ascend/vllm_ascend/spec_decode/dflash_prefix_proposer.py",
        "vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
    ]
    return {name: helpers.digest(root / name) for name in paths}


def run(args):
    helpers = load_helpers()
    root = args.spec_main.expanduser().resolve(strict=True)
    draft = args.draft.expanduser().resolve(strict=True)
    target = args.target.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if not 1 <= args.steps <= 32 or not 1 <= args.port <= 65535:
        raise ValueError("Steps must be 1..32; port must be 1..65535")
    if not args.npu.isdecimal():
        raise ValueError("NPU must name one physical device, e.g. NPU=6")
    config = json.loads((draft / "config.json").read_text())
    if config.get("prefix_selector_kind") != "compiled_prefix_v3":
        raise ValueError("DRAFT must be the trained compiled_prefix_v3 checkpoint")
    block_size = config.get("block_size")
    if type(block_size) is not int or not 2 <= block_size <= 32:
        raise ValueError("This diagnostic supports checkpoint block sizes 2..32")
    prompts = helpers.frozen_prompts(manifest, target)
    sources = source_digests(root, helpers)
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise ValueError(f"Port {args.port} is busy; use an idle port and NPU") from error
    output.mkdir(parents=True, exist_ok=False)
    traces = output / "traces"
    traces.mkdir()
    log_path = output / "server.log"
    profile = profiler_config(traces, args.steps)
    command = ["bash", str(root / "speculators/examples/serve_dflash_prefix_compiled.sh"),
               "--profiler-config", json.dumps(profile)]
    env = os.environ.copy()
    env.update(SPEC_MAIN=str(root), DRAFT=str(draft), TARGET_MODEL=str(target),
               PYTHON_BIN=sys.executable, NPU=args.npu, PORT=str(args.port), MODE="fast",
               VALIDATE_STEPS="32", SELECTOR_PROFILE_STEPS="0",
               DFLASH_PREFIX_SELECTOR_PROFILE_STEPS="0", DFLASH_PREFIX_PROFILE_STEPS="0",
               VLLM_CUSTOM_SCOPES_FOR_PROFILING="1", VLLM_NVTX_SCOPES_FOR_PROFILING="0",
               SERVED_MODEL_NAME=args.model)
    env.pop("DFLASH_PREFIX_PROFILE_DIR", None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f"http://127.0.0.1:{args.port}"
    record = {"diagnostic_only": True, "complete": False, "draft": str(draft),
              "target": str(target), "draft_config_sha256": helpers.digest(draft / "config.json"),
              "manifest_sha256": helpers.digest(manifest), "manifest_indices": [0, 1],
              "npu": args.npu, "port": args.port, "profiler_config": profile,
              "source_sha256": sources, "requests": [], "server_command": command,
              "note": "Instrumented timeline, not serving throughput. CPU and NPU intervals overlap; do not sum all operator durations."}
    process = None
    started = time.monotonic()
    print(f"Diagnostic output: {output}\nStarting temporary server on NPU {args.npu}, port {args.port}.", flush=True)
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record["server_pid"] = process.pid
            deadline = time.monotonic() + 1800
            announced = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}; see {log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 1800 seconds")
                try:
                    models = helpers.read_json(opener, base_url + "/v1/models", timeout=5)
                except (urllib.error.URLError, TimeoutError, RuntimeError):
                    models = None
                if models is not None:
                    if args.model not in [entry.get("id") for entry in models.get("data", [])]:
                        raise RuntimeError("Unexpected served model name")
                    break
                if time.monotonic() - announced > 30:
                    print(f"Waiting for server; log: {log_path}", flush=True)
                    announced = time.monotonic()
                time.sleep(1)
            print("Warmup: production parity and graph capture finish before tracing.", flush=True)
            for attempt in range(3):
                request = helpers.completion(opener, base_url, args.model, prompts[0], 512)
                record["requests"].append(dict(request, phase="warmup"))
                text = log_path.read_text(errors="replace")
                if any("COMPILED_PREFIX_PARITY_PASS" in line
                       and "steady-state fast path enabled" in line for line in text.splitlines()):
                    break
            else:
                raise RuntimeError("Production parity gate did not finish; tracing was not started")
            if "COMPILED_PREFIX_PROFILE_READY" in text:
                raise RuntimeError("Isolated selector profiling is unexpectedly enabled")
            record["production_fast_gate_passed"] = True
            log_offset = len(text)
            print(f"Tracing {args.steps} worker iterations after prefill; no benchmark is run.", flush=True)
            # Try stop even if start's HTTP response is lost after enabling the
            # worker profiler. Its endpoint is idempotent in the installed code.
            probe_error = None
            try:
                profile_control(opener, base_url, "start_profile")
                tokens = max(256, (args.steps + 4) * block_size)
                request = helpers.completion(opener, base_url, args.model, prompts[1], tokens)
                record["requests"].append(dict(request, phase="trace_probe"))
            except (Exception, KeyboardInterrupt) as error:
                probe_error = error
                raise
            finally:
                try:
                    profile_control(opener, base_url, "stop_profile")
                except Exception as error:
                    record["profiler_stop_error"] = f"{type(error).__name__}: {error}"
                    if probe_error is None:
                        raise
            text = log_path.read_text(errors="replace")[log_offset:]
            if "Failed to start profiler" in text or "Failed to stop profiler" in text:
                raise RuntimeError("Worker profiler reported a failure; see server.log")
            # Ascend trace analysis can finish asynchronously. Wait for analyzed
            # NPU kernels and the timeline before terminating our server group.
            deadline = time.monotonic() + 300
            previous = None
            stable = 0
            while True:
                files = readable_trace_files(traces)
                snapshot = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                            for path in files]
                stable = stable + 1 if snapshot == previous else 0
                previous = snapshot
                if (stable >= 3 and has_kernel_rows(files)
                        and any(path.name == "trace_view.json" and path.stat().st_size > 32
                                for path in files)):
                    break
                if process.poll() is not None:
                    raise RuntimeError("Server exited before trace analysis finished")
                if time.monotonic() > deadline:
                    raise TimeoutError("No complete analyzed NPU trace within 300 seconds; see server.log and traces/")
                time.sleep(1)
            record["trace_files"] = [{"path": str(path.relative_to(output)),
                                      "bytes": path.stat().st_size} for path in files]
            record["complete"] = True
            print(f"Trace complete: {len(files)} analyzed files. Model weights and token policy were not modified.", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        print(record["error"], file=sys.stderr, flush=True)
    finally:
        if process is not None:
            try:
                helpers.stop_owned_server(process)
            except Exception as error:
                record["cleanup_error"] = f"{type(error).__name__}: {error}"
                record["complete"] = False
        record["diagnostic_wall_seconds"] = time.monotonic() - started
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        archive = output.parent / (output.name + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
            for path in readable_trace_files(output):
                stream.write(path, path.relative_to(output))
        print(f"Upload this diagnostic ZIP: {archive}", flush=True)
        if "cleanup_error" in record:
            print("Could not confirm server cleanup; see run.json.", flush=True)
        else:
            print("Only the server started by this diagnostic was stopped.", flush=True)
    return 0 if record["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec-main", "draft", "target", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--npu", default="6")
    parser.add_argument("--port", type=int, default=8213)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--model", default="qwen3-4b-dflash")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
