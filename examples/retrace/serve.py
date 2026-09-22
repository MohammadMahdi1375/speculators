#!/usr/bin/env python
"""Start the native, one-NPU ReTrace OpenAI-compatible server."""

import argparse
import json
import os
import shlex
import sys

from speculators.models.retrace.serving import speculative_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = speculative_config(args.draft)
    if config["method"] != "dflash":
        parser.error("This launcher is for isolated ReTrace/DFlash")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.target,
        "--served-model-name",
        "qwen3-retrace",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        "0.85",
        "--additional-config",
        json.dumps({"enable_reduce_sample": False}),
        "--speculative-config",
        json.dumps(config),
    ]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        os.execvpe(command[0], command, dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="0"))


if __name__ == "__main__":
    main()
