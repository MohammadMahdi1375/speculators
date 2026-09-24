#!/usr/bin/env bash
# DRAFT=/checkpoint for prefix or plain DFlash; omit DRAFT for target-only.
# PREFIX_WALK_BACKEND=triton enables the fused walk after passing its NPU check.
# PREFIX_DISABLE_SELECTOR=1 evaluates the same backbone with the head bypassed.
set -eo pipefail
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:?Activate vllm-dflash2-main first}/bin/python}"
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY DFLASH_TP_GATHER
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export VLLM_USE_V2_MODEL_RUNNER=0
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ASCEND_RT_VISIBLE_DEVICES="${SERVE_NPUS:-8}"
set -u
if [[ "$ASCEND_RT_VISIBLE_DEVICES" == *,* ]]; then
    echo "Initial evaluation uses TP=1 on one NPU; set SERVE_NPUS to one device." >&2; exit 2
fi
MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/}"
PORT="${PORT:-8094}"
DRAFT="${DRAFT:-}"
export PREFIX_WALK_BACKEND="${PREFIX_WALK_BACKEND:-torch}"
export PREFIX_DISABLE_SELECTOR="${PREFIX_DISABLE_SELECTOR:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPEC_MAIN/output/prefix_serve_$(date +%Y%m%d_%H%M%S)_$PORT}"
mkdir -p "$OUTPUT_DIR"
args=(--model "$MODEL" --served-model-name qwen3-4b --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size 1 --max-model-len "${MAX_MODEL_LEN:-4096}"
    --max-num-seqs "${MAX_NUM_SEQS:-8}" --gpu-memory-utilization 0.85 --enforce-eager)
if [[ -n "$DRAFT" ]]; then
    "$PYTHON_BIN" - "$DRAFT" "$OUTPUT_DIR/speculative.json" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1]).resolve()
c = json.loads((p / "config.json").read_text())
if c.get("speculators_model_type") not in {"dflash", "dflash_prefix"}:
    raise SystemExit("Expected a speculators DFlash or dflash_prefix checkpoint")
if c.get("sample_from_anchor", False):
    raise SystemExit("This launcher expects sample_from_anchor=False")
if c.get("draft_vocab_size") != 151936:
    raise SystemExit("This launcher expects the full Qwen3-4B vocabulary")
if os.environ["PREFIX_WALK_BACKEND"] not in {"torch", "triton", "auto"}:
    raise SystemExit("PREFIX_WALK_BACKEND must be torch, triton, or auto")
if os.environ["PREFIX_DISABLE_SELECTOR"] not in {"0", "1"}:
    raise SystemExit("PREFIX_DISABLE_SELECTOR must be 0 or 1")
if c["speculators_model_type"] == "dflash_prefix":
    # Create a run-local config view and symlink weights. The trained checkpoint
    # stays unchanged when choosing a backend or disabling the head for ablation.
    source = p
    p = Path(sys.argv[2]).parent.resolve() / "draft_view"
    p.mkdir(exist_ok=False)
    c["prefix_walk_backend"] = os.environ["PREFIX_WALK_BACKEND"]
    c["prefix_disable_selector"] = os.environ["PREFIX_DISABLE_SELECTOR"] == "1"
    (p / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    for item in source.iterdir():
        if item.is_file() and item.name != "config.json":
            (p / item.name).symlink_to(item.resolve())
    (p.parent / "source_checkpoint.txt").write_text(str(source) + "\n")
out = {"method": "dflash", "model": str(p), "num_speculative_tokens": c["block_size"] - 1,
       "draft_sample_method": "greedy", "enforce_eager": True}
Path(sys.argv[2]).write_text(json.dumps(out))
PY
    args+=(--speculative-config "$(cat "$OUTPUT_DIR/speculative.json")")
fi
"$PYTHON_BIN" "$SPEC_MAIN/speculators/scripts/snapshot_dflash_prefix.py" --spec-main "$SPEC_MAIN" --output "$OUTPUT_DIR/source"
cp "$0" "$OUTPUT_DIR/launcher.sh"
for repo in speculators vllm vllm-ascend; do
    git -C "$SPEC_MAIN/$repo" rev-parse HEAD > "$OUTPUT_DIR/${repo}_commit.txt"
done
"$PYTHON_BIN" - > "$OUTPUT_DIR/packages.txt" <<'PY'
from importlib.metadata import distributions
print("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})))
PY
printf '%q ' "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$OUTPUT_DIR/command.txt"
echo "Serving on one NPU ($ASCEND_RT_VISIBLE_DEVICES); logs: $OUTPUT_DIR/server.log"
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server "${args[@]}" 2>&1 | tee "$OUTPUT_DIR/server.log"
