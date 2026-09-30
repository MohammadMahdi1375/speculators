#!/usr/bin/env bash
# From speculators/: CHECK_ONLY=1 bash examples/train/train_dflash_prefix_tau.sh
set -eo pipefail

# Check the entire file before preparing a run or starting a process.
bash -n -- "${BASH_SOURCE[0]}"

# ============================================================
# Paths -- helpers live inside Speculators, not the old ZIP folder.
# ============================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
SPEC_MAIN="$(cd -- "$SPEC_MAIN" && pwd)"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-}}"

if [[ -z "$ENV_ROOT" ]]; then
    echo "ERROR: No Conda environment is active and ENV_ROOT was not provided." >&2
    exit 1
fi

CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "Using ENV_ROOT=$ENV_ROOT"
echo "Using PYTHON_BIN=$PYTHON_BIN"
HELPER_DIR="$SPEC_MAIN/speculators/scripts/dflash_prefix_tau"

export MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATASET="${DATASET:-/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
# Only read the original recipe. No saved draft weights are loaded.
export BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/run.yaml}"
if [[ ! -f "$BASE_TRAIN_CONFIG" || ! -r "$BASE_TRAIN_CONFIG" ]]; then
    printf 'ERROR: Base training recipe is missing or unreadable: %s\n' "$BASE_TRAIN_CONFIG" >&2
    printf 'Copy the original run.yaml, or set BASE_TRAIN_CONFIG to its actual path.\n' >&2
    exit 1
fi

# ============================================================
# Training settings -- edit these defaults or override via env.
# ============================================================
export BLOCK_SIZE="${BLOCK_SIZE:-16}"       # 1 anchor + 15 proposed tokens
export PREFIX_TOP_K="${PREFIX_TOP_K:-16}"   # candidates per position, not block size
export PREFIX_RANK="${PREFIX_RANK:-256}"
export PREFIX_HEADS="${PREFIX_HEADS:-4}"
export PREFIX_LAYERS="${PREFIX_LAYERS:-2}"
export EPOCHS="${EPOCHS:-5}"
export SELECTOR_DELAY_STEPS="${SELECTOR_DELAY_STEPS:-2000}"
export SELECTOR_WARMUP_STEPS="${SELECTOR_WARMUP_STEPS:-2000}"
export HEAD_LR="${HEAD_LR:-0.0003}"
export PREFIX_LOSS_ALPHA="${PREFIX_LOSS_ALPHA:-0.25}"
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export VAL_BATCHES="${VAL_BATCHES:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CHECK_ONLY="${CHECK_ONLY:-0}"
# Preserve the backbone optimizer, LR, data split, and sequence length from run.yaml.

if [[ ! "$BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]] || ((BLOCK_SIZE < 2 || BLOCK_SIZE > 3072)); then
    echo "BLOCK_SIZE must be an integer from 2 through 3072 (anchor included)." >&2
    exit 2
fi
if [[ "$CHECK_ONLY" != 0 && "$CHECK_ONLY" != 1 ]]; then
    echo "CHECK_ONLY must be 0 or 1" >&2
    exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs${BLOCK_SIZE}_layer${PREFIX_LAYERS}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT}"
if [[ "${OUTPUT_DIR%/}" == "$OUTPUT_ROOT" ]]; then
    export OUTPUT_DIR="$OUTPUT_ROOT/tau_v2_scratch_$(date +%Y%m%d_%H%M%S)_$$"
fi

# ============================================================
# NPU allocation
# ============================================================
export VLLM_NPUS="${VLLM_NPUS:-0,1}"
export VLLM_DP="${VLLM_DP:-2}"
export VLLM_PORT="${VLLM_PORT:-8092}"
export VLLM_RPC_PORT="${VLLM_RPC_PORT:-29692}"
export TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6}"
export NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-5}"

# ============================================================
# Ascend environment
# ============================================================
unset PYTHONPATH ASCEND_LAUNCH_BLOCKING
CANN_ENV_SH="${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}"
ATB_ENV_SH="${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
for env_file in "$CANN_ENV_SH" "$ATB_ENV_SH"; do
    if [[ ! -r "$env_file" ]]; then
        printf 'ERROR: Ascend environment file is unreadable: %s\n' "$env_file" >&2
        printf 'Set CANN_ROOT, or set CANN_ENV_SH and ATB_ENV_SH explicitly.\n' >&2
        exit 1
    fi
done
source "$CANN_ENV_SH"
source "$ATB_ENV_SH"
# Do not force the old server chip. An explicit SOC_VERSION, if needed,
# must match this A2 server and its installed vllm-ascend build.
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY DFLASH_TP_GATHER
export NO_PROXY="localhost,127.0.0.1,::1" no_proxy="localhost,127.0.0.1,::1"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
set -u

# ============================================================
# Prepare a scratch run with BLOCK_SIZE tokens and record its configuration.
# ============================================================
for name in MODEL DATASET BASE_TRAIN_CONFIG OUTPUT_DIR; do
    resolved_path="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${!name}")"
    printf -v "$name" '%s' "$resolved_path"
    export "$name"
done
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" "$HELPER_DIR/prepare_experiment.py"
cd "$SPEC_MAIN/speculators"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" scripts/preflight_dflash_prefix.py \
    --model "$MODEL" --dataset "$DATASET" --output "$OUTPUT_DIR" \
    --target-layers 1 9 17 25 33 --mask-token-id 151669
mkdir -p "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/hidden_states"
SNAPSHOT_DIR="$OUTPUT_DIR/provenance/tau_training"
mkdir -p "$SNAPSHOT_DIR"
cp -- "$HELPER_DIR/prepare_experiment.py" "$HELPER_DIR/train_entry.py" \
    "$HELPER_DIR/check_selector.py" "$HELPER_DIR/checkpoint_config.py" "$SNAPSHOT_DIR/"
cp -- "$SCRIPT_PATH" "$SNAPSHOT_DIR/train_dflash_prefix_tau.sh"
cp -- "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_tau.sh" "$SNAPSHOT_DIR/"
# Snapshot the installed model too: git diff alone omits untracked source files.
cp -a -- "$SPEC_MAIN/speculators/src/speculators/models/dflash_prefix" \
    "$OUTPUT_DIR/provenance/dflash_prefix_model"
for repo in speculators vllm vllm-ascend; do
    git -C "$SPEC_MAIN/$repo" rev-parse HEAD > "$OUTPUT_DIR/provenance/${repo}_commit.txt"
    git -C "$SPEC_MAIN/$repo" diff > "$OUTPUT_DIR/provenance/${repo}.patch"
done
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Prepared scratch configuration: $OUTPUT_DIR"
    echo "BLOCK_SIZE=$BLOCK_SIZE; speculative proposals=$((BLOCK_SIZE - 1)); top-K=$PREFIX_TOP_K"
    echo "No NPU server or training process started. Use a fresh OUTPUT_DIR for training."
    exit 0
fi
"$PYTHON_BIN" - <<'PY_PORTS'
import os, socket
for name in ("VLLM_PORT", "VLLM_RPC_PORT"):
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", int(os.environ[name])))
        except OSError as exc:
            raise SystemExit(f"{name} is busy; finish the existing server first: {exc}")
PY_PORTS
ASCEND_RT_VISIBLE_DEVICES="${TRAIN_NPUS%%,*}" "$PYTHON_BIN" "$SNAPSHOT_DIR/check_selector.py" \
    --device npu:0 --block-size "$BLOCK_SIZE" --top-k "$PREFIX_TOP_K" \
    2>&1 | tee "$OUTPUT_DIR/logs/selector_check.log"

# ============================================================
# Target hidden-state server and training
# ============================================================
setsid env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON_BIN" scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids 1 9 17 25 33 \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --provenance-dir "$OUTPUT_DIR/provenance" -- \
    --served-model-name "$MODEL" \
    --data-parallel-size "$VLLM_DP" --tensor-parallel-size 1 \
    --data-parallel-rpc-port "$VLLM_RPC_PORT" --port "$VLLM_PORT" \
    --max-model-len 3328 --gpu-memory-utilization 0.85 \
    > "$OUTPUT_DIR/logs/hidden_states_server.log" 2>&1 &
VLLM_PID=$!
cleanup() {
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
    for ((i=0; i<15; i++)); do
        if ! kill -0 -- "-$VLLM_PID" 2>/dev/null; then break; fi
        sleep 1
    done
    kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + 1800))
until curl --noproxy '*' --connect-timeout 3 --max-time 5 -fsS \
    "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null || ((SECONDS >= deadline)); then
        tail -n 80 "$OUTPUT_DIR/logs/hidden_states_server.log"
        exit 1
    fi
    sleep 3
done
"$PYTHON_BIN" - <<'PY_ID'
import json, os, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f"http://127.0.0.1:{os.environ['VLLM_PORT']}/v1/models", timeout=15) as r:
    advertised = json.load(r)["data"][0]["id"]
if advertised != os.environ["MODEL"]:
    raise SystemExit(f"Hidden-state model ID mismatch: {advertised!r}")
PY_ID

ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    "$SNAPSHOT_DIR/train_entry.py" --config "$OUTPUT_DIR/train_config.yaml" \
    2>&1 | tee "$OUTPUT_DIR/logs/train.log"

mkdir -p "$OUTPUT_ROOT"
LATEST="$OUTPUT_ROOT/tau_v2_scratch_latest"
if [[ -e "$LATEST" && ! -L "$LATEST" ]]; then
    echo "Leaving existing ordinary path untouched: $LATEST"
else
    ln -sfn -- "$OUTPUT_DIR" "$LATEST"
fi
echo "Best checkpoint: $OUTPUT_DIR/checkpoints/checkpoint_best"
echo "Tau history: $OUTPUT_DIR/checkpoints/tau_history.jsonl"
printf 'Serve: BLOCK_SIZE=%q DRAFT=%q bash %q\n' \
    "$BLOCK_SIZE" "$OUTPUT_DIR/checkpoints/checkpoint_best" \
    "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_tau.sh"
