#!/usr/bin/env python3
"""Run the eager/target-graph output check sequentially on one idle NPU.

Only child servers started by this script are stopped. Existing servers are
never attached to or stopped. No model training or throughput test is run.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error


def helpers(filename="profile_server.py"):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location("_target_graph_http_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_dispatch_observed(text, block):
    # vLLM's CUDAGraphLogging table: unpadded | padded | paddings | mode | count.
    pattern = rf"\|\s*{block}\s*\|\s*{block}\s*\|\s*0\s*\|\s*FULL\s*\|\s*[1-9][0-9]*\s*\|"
    return re.search(pattern, text) is not None


def run_phase(args, phase, output, helper, reference=None):
    log_path = output / f"{phase}.log"
    run_dir = output / phase
    probe_path = output / f"{phase}_probe.json"
    command = ["bash", str(args.spec_main / "Evaluator/run_server_dflash.sh")]
    env = dict(os.environ, SPEC_MAIN=str(args.spec_main), DRAFT_MODEL=str(args.draft),
               TARGET_MODEL=str(args.target), NPU=args.npu, PORT=str(args.port),
               TARGET_EXECUTION=phase, MODE="fast", GRAPH_METRICS="1", CHECK_ONLY="0",
               PYTHON_BIN=sys.executable, RUN_DIR=str(run_dir),
               SERVED_MODEL_NAME=args.served_model_name,
               SPECULATIVE_METHOD=args.speculative_method,
               NUM_SPECULATIVE_TOKENS=args.num_speculative_tokens,
               DRAFT_ENFORCE_EAGER=args.draft_enforce_eager)
    opener = helper.urllib.request.build_opener(helper.urllib.request.ProxyHandler({}))
    process = None
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline, announced = time.monotonic() + 1800, time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{phase} server exited; see {log_path}")
                try:
                    models = helper.read_json(opener, f"http://127.0.0.1:{args.port}/v1/models", timeout=5)
                except (urllib.error.URLError, TimeoutError, RuntimeError):
                    models = None
                if models is not None:
                    if args.served_model_name not in [r["id"] for r in models["data"]]:
                        raise RuntimeError("Unexpected model on the requested port")
                    break
                now = time.monotonic()
                if now > deadline:
                    raise TimeoutError(f"{phase} startup exceeded 1800 seconds; see {log_path}")
                if now - announced >= 30:
                    print(f"Waiting for {phase} server; log: {log_path}", flush=True)
                    announced = now
                time.sleep(1)
            probe = [sys.executable, str(Path(__file__).with_name("probe_target_graph.py")),
                     "--launch-record", str(run_dir / "launch.json"), "--manifest", str(args.manifest),
                     "--output", str(probe_path), "--num-prompts", str(args.num_prompts),
                     "--tokens", str(args.tokens)]
            if reference is not None:
                probe += ["--reference", str(reference)]
            try:
                subprocess.run(probe, check=True, env=env)
            except subprocess.CalledProcessError as error:
                detail = "probe did not save a report"
                if probe_path.is_file():
                    try:
                        failure = json.loads(probe_path.read_text())
                        detail = (f"stage={failure.get('stage', 'unknown')}: "
                                  f"{failure.get('error', failure.get('comparison', {}))}")
                    except (ValueError, OSError):
                        detail = "probe report could not be read"
                raise RuntimeError(
                    f"{phase} probe failed: {detail}; server traceback: {log_path}"
                ) from error
            launch = json.loads((run_dir / "launch.json").read_text())
            # Allow the normal metrics logger to flush after short probes.
            deadline = time.monotonic() + 15
            while True:
                text = log_path.read_text(errors="replace")
                graph_ok = phase == "eager" or full_dispatch_observed(text, launch["proposals"] + 1)
                if graph_ok or time.monotonic() >= deadline:
                    break
                time.sleep(1)
            if not graph_ok:
                raise RuntimeError(f"No FULL target decode dispatch observed; see {log_path}. Do not assume the requested graph ran.")
            if launch["kind"] in {"local_prefix_v2", "compiled_prefix_v3"} and not any(
                "PREFIX_PARITY_PASS" in line and "steady-state fast path enabled" in line
                for line in text.splitlines()
            ):
                raise RuntimeError(f"Selector fast-path parity was not confirmed; see {log_path}")
            return probe_path
    finally:
        if process is not None:
            helper.stop_owned_server(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-main", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=os.environ.get("TARGET_MODEL", "/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--npu", default="6")
    parser.add_argument("--port", type=int, default=8213)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--speculative-method", choices=("auto", "dflash", "dspark"), default="auto")
    parser.add_argument("--num-speculative-tokens", default="auto")
    parser.add_argument("--draft-enforce-eager", choices=("omit", "1", "0"), default="omit")
    parser.add_argument("--served-model-name", default="qwen3-4b-spec-check")
    args = parser.parse_args()
    args.spec_main = args.spec_main.expanduser().resolve(strict=True)
    args.draft = args.draft.expanduser().resolve(strict=True)
    args.target = args.target.expanduser().resolve(strict=True)
    args.manifest = (args.manifest or args.spec_main / "Evaluator/retrace_eval_prompts_all_v2.json").expanduser().resolve(strict=True)
    if not args.npu.isdecimal() or not 1 <= args.port <= 65535:
        parser.error("Use one physical NPU and a port in [1, 65535]")
    if not 1 <= args.num_prompts <= 128 or not 64 <= args.tokens <= 2048:
        parser.error("Use 1..128 prompts and 64..2048 tokens")
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError as error:
            parser.error(f"Port {args.port} is busy. Use an idle port and NPU: {error}")
    output = (args.output or args.spec_main / "output" / f"target_graph_check_{time.time_ns()}").expanduser().resolve()
    helper = helpers()
    output.mkdir(parents=True, exist_ok=False)
    record = {"complete": False, "diagnostic_only": True, "draft": str(args.draft),
              "npu": args.npu, "port": args.port, "num_prompts": args.num_prompts, "tokens": args.tokens,
              "speculative_method_requested": args.speculative_method,
              "num_speculative_tokens_requested": args.num_speculative_tokens,
              "draft_enforce_eager_requested": args.draft_enforce_eager}
    print(f"Output check: {output}\nTwo servers run sequentially on NPU {args.npu}; no benchmark.", flush=True)
    try:
        eager = run_phase(args, "eager", output, helper)
        graph = run_phase(args, "graph", output, helper, reference=eager)
        record["comparison"] = json.loads(graph.read_text())["comparison"]
        record["target_full_dispatch_observed"] = True
        record["complete"] = bool(record["comparison"]["pass"])
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        print(record["error"], file=sys.stderr, flush=True)
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["complete"] and (output / "graph.log").is_file():
        try:
            collector = helpers("collect_target_graph_failure.py")
            record["diagnostic_zip"] = str(collector.collect(output))
        except Exception as error:
            record["collection_error"] = f"{type(error).__name__}: {error}"
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["complete"]:
        print("TARGET_GRAPH_CHECK_PASS: observed token IDs and acceptance counters match; "
              "target FULL dispatch confirmed; prefix selector checks applied only to prefix checkpoints. "
              "This covers the observed prompts, not universal parity or speedup.")
    print(f"Small result file: {output / 'result.json'}")
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
