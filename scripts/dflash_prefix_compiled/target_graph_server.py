#!/usr/bin/env python3
"""Launch Qwen3 with DFlash, DFlash-Prefix, or DSpark on Ascend.

No Torch import, monkey patch, checkpoint edit, or model-parameter change.
The optional graph metrics report actual runtime dispatch in the server log.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


PREFIX_KINDS = {"local_prefix_v2", "compiled_prefix_v3"}


def eager_override(value):
    if value is None or value == "omit":
        return None
    if type(value) is bool:
        return value
    if value in ("1", "true"):
        return True
    if value in ("0", "false"):
        return False
    raise ValueError("DRAFT_ENFORCE_EAGER must be omit, 1, or 0")


def checkpoint(draft, target, method="auto", proposals="auto", draft_enforce_eager="omit"):
    draft = Path(draft).expanduser().resolve(strict=True)
    target = Path(target).expanduser().resolve(strict=True)
    cfg = json.loads((draft / "config.json").read_text())
    target_cfg = json.loads((target / "config.json").read_text())
    if draft == target or "speculators_model_type" in target_cfg:
        raise ValueError("TARGET_MODEL must be the original Qwen3 target, not a draft")
    target_cfg = target_cfg.get("text_config", target_cfg)
    if target_cfg.get("model_type") != "qwen3":
        raise ValueError("This controlled launcher supports Qwen3 text targets")
    family = cfg.get("speculators_model_type")
    if family in {"dflash", "dspark"}:
        kind = family
    elif family == "dflash_prefix" and cfg.get("prefix_selector_kind") in {
        "local_prefix_v2", "compiled_prefix_v3"
    }:
        kind = cfg["prefix_selector_kind"]
        if cfg.get("prefix_top_k") != 16:
            raise ValueError("Expected the trained top-K=16 selector")
        if cfg.get("prefix_disable_selector", False) or cfg.get("prefix_inference_gate_scale", 1) != 1:
            raise ValueError("Selector must be enabled at its trained scale")
        if cfg.get("prefix_walk_backend", "torch") != "torch":
            raise ValueError("Attention selectors require prefix_walk_backend=torch")
    else:
        raise ValueError("Expected a Speculators DFlash, attention-prefix, or DSpark checkpoint")
    expected_method = "dspark" if family == "dspark" else "dflash"
    if method == "auto":
        method = expected_method
    if method != expected_method:
        raise ValueError(f"Checkpoint family={family} requires method={expected_method}; got {method}")
    block = cfg.get("block_size")
    if type(block) is not int or block < 2:
        raise ValueError("Expected an integer checkpoint block_size >= 2")
    if family != "dspark" and block not in (8, 16):
        raise ValueError("This DFlash graph recipe supports block_size 8 or 16")
    sample_anchor = cfg.get("sample_from_anchor", False)
    if type(sample_anchor) is not bool:
        raise ValueError("sample_from_anchor must be a boolean")
    available = block if family == "dspark" and sample_anchor else block - 1
    if proposals == "auto":
        proposals = available
    else:
        if type(proposals) is bool or str(proposals) != str(int(proposals)):
            raise ValueError("NUM_SPECULATIVE_TOKENS must be auto or a positive decimal integer")
        proposals = int(proposals)
    if not 1 <= proposals <= 15:
        raise ValueError(f"Resolved proposals={proposals}; this Ascend graph recipe requires 1..15 "
                         "(proposals + verifier bonus <= 16). Set NUM_SPECULATIVE_TOKENS explicitly; "
                         "the launcher does not silently truncate DSpark proposals.")
    if family == "dspark":
        if proposals > available:
            raise ValueError(f"DSpark checkpoint provides {available} trained proposal positions")
    elif proposals != available:
        raise ValueError("DFlash/Prefix NUM_SPECULATIVE_TOKENS must equal checkpoint block_size - 1")
    layer = cfg.get("transformer_layer_config", {})
    for key in ("hidden_size", "vocab_size"):
        expected = (cfg.get("target_hidden_size") or layer.get(key)) if key == "hidden_size" else layer.get(key)
        if not target_cfg.get(key) or expected != target_cfg[key]:
            raise ValueError(f"Target/draft {key} mismatch")
    if not any(draft.glob("*.safetensors")):
        raise ValueError("Draft safetensors weights are missing")
    spec = {
        "method": method, "model": str(draft), "num_speculative_tokens": proposals,
        "draft_sample_method": "greedy", "disable_padded_drafter_batch": False,
        "enable_adaptive_verification": False,
    }
    override = eager_override(draft_enforce_eager)
    if override is not None:
        spec["enforce_eager"] = override
    return draft, target, kind, block, spec


def execution_config(execution, block):
    if execution not in {"graph", "eager"}:
        raise ValueError("TARGET_EXECUTION must be graph or eager")
    # This pinned Ascend v1 runner requires VLLM_COMPILE (3) for _use_aclgraph.
    # Mode 0 plus FULL_DECODE_ONLY would leave graph-parameter setup disabled.
    # Disable the optional arithmetic fusions; capture one verification block.
    compilation = {
        "mode": 3 if execution == "graph" else 0,
        "cudagraph_mode": "FULL_DECODE_ONLY" if execution == "graph" else "NONE",
        "custom_ops": ["all"],
    }
    if execution == "graph":
        compilation.update(cudagraph_capture_sizes=[block], max_cudagraph_capture_size=block)
    additional = {
        "enable_reduce_sample": False,
        "ascend_compilation_config": {
            "enable_npugraph_ex": False, "enable_static_kernel": False,
            "enable_super_kernel": False, "fuse_norm_quant": False,
            "fuse_qknorm_rope": False, "fuse_muls_add": False,
        },
    }
    return compilation, additional


def serving_environment(inherited, kind, mode):
    if mode not in {"reference", "validate", "fast"}:
        raise ValueError("MODE must be reference, validate, or fast")
    env = {k: v for k, v in inherited.items() if not k.startswith("DFLASH_PREFIX_")}
    for key in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "ASCEND_LAUNCH_BLOCKING"):
        env.pop(key, None)
    # Profiling and validation settings from previous terminals must not leak.
    env.update(VLLM_CUSTOM_SCOPES_FOR_PROFILING="0", VLLM_NVTX_SCOPES_FOR_PROFILING="0",
               VLLM_USE_BREAKABLE_CUDAGRAPH="0")
    if kind not in PREFIX_KINDS:
        env.update(DFLASH_PREFIX_INFERENCE="reference", DFLASH_PREFIX_NPU_GRAPH="0")
        return env
    env.update(
        DFLASH_PREFIX_INFERENCE=mode, DFLASH_PREFIX_NPU_GRAPH="0" if mode == "reference" else "1",
        DFLASH_PREFIX_TOKEN_CACHE="0", DFLASH_PREFIX_SCORE_CHECK="strict",
        DFLASH_PREFIX_VALIDATE_STEPS="32", DFLASH_PREFIX_PROFILE_STEPS="0",
        DFLASH_PREFIX_SELECTOR_PROFILE_STEPS="0", DFLASH_PREFIX_FUSION_PROFILE_STEPS="0",
        DFLASH_PREFIX_HISTORY_GROUP_SIZE="1",
        DFLASH_PREFIX_FUSION="local_unpaired" if kind == "local_prefix_v2" and mode != "reference" else "off",
    )
    if mode != "reference":
        env["DFLASH_PREFIX_GRAPH_SCOPE"] = "selector"
    return env


def command_and_record(args):
    root = args.spec_main.expanduser().resolve(strict=True)
    draft, target, kind, block, spec = checkpoint(
        args.draft, args.target, getattr(args, "speculative_method", "auto"),
        getattr(args, "num_speculative_tokens", "auto"),
        getattr(args, "draft_enforce_eager", "omit"),
    )
    proposals = spec["num_speculative_tokens"]
    verification_tokens = proposals + 1
    if not args.npu.isdecimal() or not 1 <= args.port <= 65535:
        raise ValueError("Use one physical NPU index and a port in [1, 65535]")
    if not math.isfinite(args.memory_utilization) or not 0 < args.memory_utilization < 1:
        raise ValueError("MEMORY_UTILIZATION must be finite and between zero and one")
    if not max(block, verification_tokens) <= args.max_model_len <= args.max_batched_tokens:
        raise ValueError("Unchunked prefill requires MAX_BATCHED_TOKENS >= MAX_MODEL_LEN >= block")
    compilation, additional = execution_config(args.target_execution, verification_tokens)
    paths = [
        "vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
        "vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py",
        "vllm-ascend/vllm_ascend/compilation/acl_graph.py",
        "vllm-ascend/vllm_ascend/platform.py",
        "vllm/vllm/config/compilation.py",
    ]
    if kind == "dspark":
        paths += ["vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py",
                  "vllm/vllm/model_executor/models/qwen3_dspark.py"]
    if kind in PREFIX_KINDS:
        paths += [
            "vllm-ascend/vllm_ascend/spec_decode/dflash_prefix_proposer.py",
            "vllm/vllm/model_executor/models/qwen3_dflash_prefix.py",
            "speculators/src/speculators/models/dflash_prefix/selector_v2.py",
            "speculators/src/speculators/models/dflash_prefix/" + (
                "compiled_inference.py" if kind == "compiled_prefix_v3" else "inference_v2.py"
            ),
        ]
    sources = {name: digest(root / name) for name in paths}
    base = (root / paths[1]).read_text()
    if spec.get("enforce_eager") is not None and "self.runner._use_aclgraph() and not self.speculative_config.enforce_eager" not in base:
        raise ValueError("Installed proposer no longer has the reviewed independent draft-eager switch")
    seed, kv_block = int(getattr(args, "seed", 42)), int(getattr(args, "kv_block_size", 128))
    if not 0 <= seed < 2**32 or kv_block < 1:
        raise ValueError("SEED must be in [0, 2**32); KV_BLOCK_SIZE must be positive")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(target),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", args.served_model_name,
        "--tensor-parallel-size", "1", "--data-parallel-size", "1",
        "--dtype", "bfloat16", "--seed", str(seed), "--generation-config", "vllm",
        "--max-num-seqs", "1", "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_batched_tokens), "--block-size", str(kv_block),
        "--gpu-memory-utilization", str(args.memory_utilization),
        "--no-async-scheduling", "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
        "--api-server-count", "1", "--renderer-num-workers", "1",
        "--additional-config", json.dumps(additional),
        "--compilation-config", json.dumps(compilation),
        "--enable-per-request-metrics", "--per-request-spec-decode-metrics", "summary",
        "--speculative-config", json.dumps(spec),
    ]
    if args.target_execution == "eager":
        command.append("--enforce-eager")
    if args.graph_metrics:
        command.append("--cudagraph-metrics")
    env = serving_environment(os.environ, kind, args.mode)
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.npu
    record = {
        "target": str(target), "draft": str(draft), "kind": kind,
        "target_config_sha256": digest(target / "config.json"),
        "draft_config_sha256": digest(draft / "config.json"), "source_sha256": sources,
        "block_size": block, "proposals": proposals, "verification_tokens": verification_tokens,
        "target_execution_requested": args.target_execution, "speculative_method": spec["method"],
        "speculative_method_requested": getattr(args, "speculative_method", "auto"),
        "draft_enforce_eager": spec.get("enforce_eager"),
        "draft_execution_requested": ("backend_default" if "enforce_eager" not in spec
                                      else "eager" if spec["enforce_eager"] else "graph_allowed"),
        "draft_backend_note": ("AscendDSparkProposer currently disables its own draft graph."
                               if kind == "dspark" else "Runtime dispatch must be checked in the server log."),
        "npu": args.npu, "port": args.port, "served_model_name": args.served_model_name,
        "command": command, "compilation_config": compilation, "speculative_config": spec,
        "additional_config": additional, "selector_environment": {
            k: v for k, v in env.items() if k.startswith("DFLASH_PREFIX_")
        },
        "graph_metrics": bool(args.graph_metrics), "check_only": bool(args.check_only),
        "max_model_len": args.max_model_len, "max_batched_tokens": args.max_batched_tokens,
        "memory_utilization": args.memory_utilization,
        "seed": seed, "kv_block_size": kv_block,
        "note": "Requested configuration; runtime graph use must be confirmed in the server log. No NPU speedup or target-output parity is certified by this launcher.",
    }
    return root, command, env, record


def add_evaluator_record(record, args, log):
    prefix = record["kind"] in PREFIX_KINDS
    mode = record["selector_environment"]["DFLASH_PREFIX_INFERENCE"]
    process_stat = Path("/proc/self/stat").read_text()
    fields = process_stat.rsplit(")", 1)[1].split()
    record.update(
        format="dflash_evaluator_integration_v3",
        base_url=f"http://127.0.0.1:{args.port}", served_model=args.served_model_name,
        variant="prefix" if prefix else "dspark" if record["kind"] == "dspark" else "vanilla",
        prefix_mode=mode,
        prefix_npu_graph=record["selector_environment"]["DFLASH_PREFIX_NPU_GRAPH"] == "1",
        prefix_token_cache=False, prefix_validate_steps=32 if prefix and mode != "reference" else 0,
        num_speculative_tokens=record["proposals"], server_log=str(log),
        launcher_pid=int(process_stat.split(" ", 1)[0]), launcher_start_ticks=fields[19],
        suggested_method=f"{record['kind']}_bs{record['block_size']}_k{record['proposals']}_{args.target_execution}_{mode}",
        device=f"npu:{args.npu}", tp_size=1, dp_size=1, dtype="bfloat16",
        max_num_seqs=1, prefix_caching=False,
        enforce_eager=args.target_execution == "eager", model_runner="ascend_v1",
        helper_sha256=digest(Path(__file__)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec-main", "target", "draft", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--npu", default="6")
    parser.add_argument("--port", type=int, default=8213)
    parser.add_argument("--target-execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--mode", choices=("fast", "validate", "reference"), default="fast")
    parser.add_argument("--speculative-method", choices=("auto", "dflash", "dspark"),
                        default=os.environ.get("SPECULATIVE_METHOD", "auto"))
    parser.add_argument("--num-speculative-tokens", default=os.environ.get("NUM_SPECULATIVE_TOKENS", "auto"))
    parser.add_argument("--draft-enforce-eager", choices=("omit", "1", "0"),
                        default=os.environ.get("DRAFT_ENFORCE_EAGER", "omit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-block-size", type=int, default=128)
    parser.add_argument("--register-evaluator", type=int, choices=(0, 1), default=0,
                        help="Record this process and its log for Evaluator/evaluator.py")
    parser.add_argument("--served-model-name", default="qwen3-4b-dflash")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-batched-tokens", type=int, default=32768)
    parser.add_argument("--memory-utilization", type=float, default=.96)
    parser.add_argument("--graph-metrics", type=int, choices=(0, 1), default=0)
    parser.add_argument("--check-only", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    root, command, env, record = command_and_record(args)
    if not args.check_only:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", args.port))
            except OSError as error:
                raise ValueError(f"Port {args.port} is unavailable: {error}") from error
    output = args.run_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    log = output / "server.log"
    if args.register_evaluator:
        add_evaluator_record(record, args, log)
    (output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    if args.register_evaluator and not args.check_only:
        tee = subprocess.Popen(["tee", str(log)], stdin=subprocess.PIPE)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
        tee.stdin.close()
        registry = root / "output/evaluator_servers"
        registry.mkdir(parents=True, exist_ok=True)
        temporary = registry / f"port_{args.port}.{os.getpid()}.tmp"
        temporary.symlink_to(output / "launch.json")
        os.replace(temporary, registry / f"port_{args.port}.json")
    print(f"TARGET_GRAPH_REQUEST mode={args.target_execution} kind={record['kind']} "
          f"block={record['block_size']} proposals={record['proposals']} "
          f"method={record['speculative_method']} draft_execution={record['draft_execution_requested']} "
          "full_verification=True", flush=True)
    print(f"Target: {record['target']}\nDraft: {record['draft']}", flush=True)
    print(f"Draft backend: {record['draft_backend_note']}", flush=True)
    print(f"Launch record: {output / 'launch.json'}", flush=True)
    if args.register_evaluator:
        print(f"Server log: {log}", flush=True)
    print(f"Command: {shlex.join(command)}", flush=True)
    if args.check_only:
        print("Configuration prepared. No server started. Use a fresh RUN_DIR for serving.")
        return
    os.chdir(root / "speculators")
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error

