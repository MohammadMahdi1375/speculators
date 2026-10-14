#!/usr/bin/env python3
"""Own a temporary diagnostic server, profile a few blocks, and collect results.

Standard library only. Does not import Torch, change checkpoints, or replace the
evaluator. The serving child sources the existing CANN environment. Only this
child's process group is stopped, even on HTTP failure or interruption.
"""

import argparse
import hashlib
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_prompts(path, target):
    document = json.loads(path.read_text())
    if document.get("thinking") is not False:
        raise ValueError("Use the same no-thinking manifest as your evaluation")
    if document.get("target_config_sha256") != digest(target / "config.json"):
        raise ValueError("Manifest target configuration differs from TARGET_MODEL")
    for name, expected in document.get("tokenizer_sha256", {}).items():
        if Path(name).name != name or digest(target / name) != expected:
            raise ValueError(f"Manifest tokenizer differs: {name}")
    rows = document["datasets"]["gsm8k"]["results"]
    if len(rows) < 2:
        raise ValueError("Manifest must contain at least two GSM8K prompts")
    result = []
    for row in rows[:2]:
        ids = row["prompt_token_ids"]
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("Invalid frozen prompt token IDs")
        result.append(ids)
    return result


def read_json(opener, url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            document = json.load(response)
    except urllib.error.HTTPError as error:
        message = error.read(8192).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message}") from error
    if "error" in document:
        raise RuntimeError(f"Server error: {document['error']}")
    return document


def completion(opener, base_url, model, ids, tokens):
    response = read_json(opener, base_url + "/v1/completions", {
        "model": model, "prompt": ids, "temperature": 0, "top_p": 1.0, "top_k": -1,
        "seed": 42, "max_tokens": tokens, "min_tokens": tokens, "ignore_eos": True,
        "stream": False,
    }, timeout=600)
    usage = response.get("usage") or {}
    if not response.get("choices") or usage.get("completion_tokens", 0) < tokens:
        raise RuntimeError("Probe ended before its requested token count")
    # Do not save prompt/generated text, or publish this request as throughput.
    return {"completion_tokens": usage["completion_tokens"], "diagnostic_only": True}


def stop_owned_server(process):
    # start_new_session=True makes this group exclusive to the diagnostic child.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Parent may have exited while workers still exist in the same owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def collect_sources(root):
    paths = [
        "speculators/src/speculators/models/dflash_prefix/compiled_profile.py",
        "speculators/src/speculators/models/dflash_prefix/compiled_inference.py",
        "speculators/src/speculators/models/dflash_prefix/compiled_selector.py",
        "speculators/src/speculators/models/dflash_prefix/npu_graph_selector_v2.py",
        "speculators/examples/serve_dflash_prefix_compiled.sh",
        "speculators/examples/profile_dflash_prefix_compiled.sh",
        "speculators/scripts/dflash_prefix_compiled/profile_server.py",
    ]
    return {name: digest(root / name) for name in paths}


def run(args):
    root = args.spec_main.expanduser().resolve(strict=True)
    draft = args.draft.expanduser().resolve(strict=True)
    target = args.target.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    config = json.loads((draft / "config.json").read_text())
    if config.get("prefix_selector_kind") != "compiled_prefix_v3":
        raise ValueError("DRAFT must be the trained compiled_prefix_v3 checkpoint")
    prompts = frozen_prompts(manifest, target)
    if not 1 <= args.steps <= 32 or not 1 <= args.port <= 65535:
        raise ValueError("Steps must be 1..32; port must be 1..65535")
    # Refuse an occupied port; never attach to or terminate an unrelated server.
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise ValueError(f"Port {args.port} is busy. Stop your existing serving process first.") from error
    sources = collect_sources(root)
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "server.log"
    record = {"diagnostic_only": True, "complete": False, "draft": str(draft),
              "target": str(target), "draft_config_sha256": digest(draft / "config.json"),
              "manifest_sha256": digest(manifest), "manifest_indices": [0, 1],
              "npu": args.npu, "port": args.port, "steps": args.steps,
              "source_sha256": sources, "requests": []}
    env = os.environ.copy()
    env.update(SPEC_MAIN=str(root), DRAFT=str(draft), TARGET_MODEL=str(target),
               PYTHON_BIN=sys.executable, NPU=args.npu, PORT=str(args.port), MODE="fast",
               VALIDATE_STEPS="32", SELECTOR_PROFILE_STEPS=str(args.steps),
               DFLASH_PREFIX_PROFILE_DIR=str(output), SERVED_MODEL_NAME=args.model)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f"http://127.0.0.1:{args.port}"
    process = None
    started = time.monotonic()
    print(f"Diagnostic output: {output}\nStarting temporary server on NPU {args.npu}, port {args.port}.", flush=True)
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(
                ["bash", str(root / "speculators/examples/serve_dflash_prefix_compiled.sh")],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            record["server_pid"] = process.pid
            deadline = time.monotonic() + 1800
            announced = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}; see {log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 1800 seconds")
                try:
                    models = read_json(opener, base_url + "/v1/models", timeout=5)
                except (urllib.error.URLError, TimeoutError, RuntimeError):
                    # vLLM may briefly answer HTTP 503 during startup.
                    models = None
                if models is not None:
                    if args.model not in [entry.get("id") for entry in models.get("data", [])]:
                        raise RuntimeError("Unexpected served model name")
                    break
                if time.monotonic() - announced > 30:
                    print(f"Waiting for server; log: {log_path}", flush=True)
                    announced = time.monotonic()
                time.sleep(1)
            print("Warmup: 512 tokens; no profiling or throughput measurement yet.", flush=True)
            for attempt in range(3):
                record["requests"].append(completion(opener, base_url, args.model, prompts[0], 512))
                text = log_path.read_text(errors="replace")
                if any("COMPILED_PREFIX_PARITY_PASS" in line and "steady-state fast path enabled" in line
                       for line in text.splitlines()):
                    break
            else:
                raise RuntimeError("Production parity gate did not finish; profiling was not armed")
            if "COMPILED_PREFIX_PROFILE_READY" not in text:
                raise RuntimeError("Diagnostic module was not loaded; check installed source paths")
            (output / "ARM").write_text("HTTP ready and production parity passed\n")
            print(f"Profiling {args.steps} real proposal blocks; synchronized timings, not a benchmark.", flush=True)
            tokens = max(256, (args.steps + 2) * config["block_size"])
            record["requests"].append(completion(opener, base_url, args.model, prompts[1], tokens))
            reports = sorted(output.glob("selector_profile_*.json"))
            if len(reports) != 1:
                raise RuntimeError(f"Expected one worker profile, found {len(reports)}; see server.log")
            report = json.loads(reports[0].read_text())
            if report.get("status") != "complete" or report.get("blocks") != args.steps:
                raise RuntimeError(f"Incomplete diagnostic: {report.get('error')}")
            record["complete"] = True
            print("Isolated synchronized means (ms):", flush=True)
            for name, value in report["timings_ms"].items():
                print(f"  {name}: {value['mean']:.4f}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        print(record["error"], file=sys.stderr, flush=True)
    finally:
        if process is not None:
            stop_owned_server(process)
        (output / "ARM").unlink(missing_ok=True)
        record["diagnostic_wall_seconds"] = time.monotonic() - started
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        archive = output.parent / (output.name + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.suffix in {".json", ".log"}:
                    stream.write(path, path.name)
        print(f"Diagnostic report ZIP: {archive}", flush=True)
        print("Only the server started by this diagnostic was stopped.", flush=True)
    return 0 if record["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec-main", "draft", "target", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--npu", default="7")
    parser.add_argument("--port", type=int, default=8213)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--model", default="qwen3-4b-dflash")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
