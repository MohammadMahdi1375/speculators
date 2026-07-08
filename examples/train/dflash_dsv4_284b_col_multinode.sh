#!/bin/bash
# ============================================================================
# DFlash CO-LOCATED, 2-NODE — target TP=16 + draft DP=16 across 16 NPUs.
#
# RUN ON BOTH NODES:
#   parent 80.5.5.108:
#   MAX_SAMPLES=9920 DEBUG_LOGS=0 LOG_FILTER=1 DFLASH_TARGET_MAX_MODEL_LEN=2048 \ 
#   CACHE_HS_ONLY=1 MAX_SAMPLES="$MAX_SAMPLES" DEBUG_LOGS="$DEBUG_LOGS" LOG_FILTER="$LOG_FILTER" DFLASH_TARGET_MAX_MODEL_LEN="$DFLASH_TARGET_MAX_MODEL_LEN" bash examples/train/dflash_dsv4_284b_col_multinode.sh 0 2>&1 | tee ./logs/train_dsv4_108.log

#   MAX_SAMPLES="$MAX_SAMPLES" DEBUG_LOGS="$DEBUG_LOGS" LOG_FILTER="$LOG_FILTER" DFLASH_TARGET_MAX_MODEL_LEN="$DFLASH_TARGET_MAX_MODEL_LEN" bash examples/train/dflash_dsv4_284b_col_multinode.sh 0 2>&1 | tee ./logs/train_dsv4_108.log

#
#   child 80.5.5.109:
#   MAX_SAMPLES=9920 DEBUG_LOGS=0 LOG_FILTER=1 DFLASH_TARGET_MAX_MODEL_LEN=2048 \ 
#   CACHE_HS_ONLY=1 MAX_SAMPLES="$MAX_SAMPLES" DEBUG_LOGS="$DEBUG_LOGS" LOG_FILTER="$LOG_FILTER" DFLASH_TARGET_MAX_MODEL_LEN="$DFLASH_TARGET_MAX_MODEL_LEN" bash examples/train/dflash_dsv4_284b_col_multinode.sh 1 2>&1 | tee ./logs/train_dsv4_109.log

#   MAX_SAMPLES="$MAX_SAMPLES" DEBUG_LOGS="$DEBUG_LOGS" LOG_FILTER="$LOG_FILTER" DFLASH_TARGET_MAX_MODEL_LEN="$DFLASH_TARGET_MAX_MODEL_LEN" bash examples/train/dflash_dsv4_284b_col_multinode.sh 1 2>&1 | tee ./logs/train_dsv4_109.log
#
# QUIET MODE, default:
#   normal training logs, loss lines, tracebacks, and key errors are kept;
#   noisy CANN/vLLM spam is suppressed.
#
# DEBUG MODE:
#   DEBUG_LOGS=1 bash examples/train/dflash_dsv4_284b_col_multinode.sh 0  2>&1 | tee ./logs/train_dsv4_0_debug.log
#
# RAW UNFILTERED LOG:
#   LOG_FILTER=0 bash examples/train/dflash_dsv4_284b_col_multinode.sh 0  2>&1 | tee ./logs/train_dsv4_0_raw.log
# ============================================================================

set -eo pipefail

NODE_RANK="${1:?usage: bash dflash_dsv4_284b_col_multinode.sh <node_rank 0|1>   (0=parent, 1=child)}"

SPEC_MAIN="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
CONDA_ENV="/home/n84449292/m84379596/conda/vllm-ascend-0202"

# Force imports to come from this checkout/submodules, not stale site-packages.
unset PYTHONPATH
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

# ===================== Logging controls =====================
# Default: quiet training logs.
# Set DEBUG_LOGS=1 for full Ascend/vLLM debug output.
DEBUG_LOGS="${DEBUG_LOGS:-0}"

# Default: filter known noisy torchrun lines before they reach console/tee.
# Set LOG_FILTER=0 to keep raw torchrun output.
LOG_FILTER="${LOG_FILTER:-1}"
CACHE_HS_ONLY="${CACHE_HS_ONLY:-0}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"

# Cleanup controls:
#   CLEAN_HS_CACHE=auto  -> clean final hidden-state cache only in CACHE_HS_ONLY=1 mode
#   CLEAN_HS_CACHE=0     -> keep existing final cache, useful for resuming/debugging
#   CLEAN_HS_CACHE=1     -> force clean final cache; only allowed in CACHE_HS_ONLY=1 mode
#   CLEAN_SHARED_STORAGE=1 -> clean /dev/shm hidden-state scratch files
CLEAN_HS_CACHE="${CLEAN_HS_CACHE:-auto}"
CLEAN_SHARED_STORAGE="${CLEAN_SHARED_STORAGE:-1}"

if [ "$DEBUG_LOGS" = "1" ]; then
    export ASCEND_LAUNCH_BLOCKING=1
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export VLLM_LOGGING_LEVEL=INFO
    unset PYTHONWARNINGS
    LOG_FILTER=0
    echo "[node $NODE_RANK] DEBUG_LOGS=1: verbose Ascend/vLLM logging enabled"
else
    unset ASCEND_LAUNCH_BLOCKING

    # Do not stream CANN SLOG to stdout during normal training.
    export ASCEND_SLOG_PRINT_TO_STDOUT=0

    # Quiet CANN host logging as much as possible.
    # If your CANN build rejects level 4, change this to 3.
    export ASCEND_GLOBAL_LOG_LEVEL=4
    export ASCEND_GLOBAL_EVENT_ENABLE=0

    # Quiet vLLM INFO logs. Tracebacks and Python exceptions still show.
    export VLLM_LOGGING_LEVEL=WARNING

    # Suppress repeated Python warning spam.
    export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning"

    echo "[node $NODE_RANK] quiet logging enabled; use DEBUG_LOGS=1 for full logs"
fi

# Filter only known high-volume noise. Keep Traceback, RuntimeError,
# OutOfMemoryError, FileNotFoundError, SafetensorError, loss, and checkpoints.
QUIET_LOG_FILTER='TypedStorage is deprecated|pin_memory.py:57|Qwen2VLImageProcessorFast|`rope_parameters`|Get a block from the existing pool failed|This error log can be ignored|Dumping input data for V1 LLM engine|Dumping scheduler output for model execution|^\[INFO\] (DRV|HCCL|HCCP|ASCENDCL)\(|^\[WARNING\] .*warnings.py:110|^\[INFO\] RUNTIME\(.*SetWatchDogDevStatus|^\[INFO\] HCCL\(.*HCCL_TRACE'

# ===================== Ascend / vLLM-Ascend env =====================
export HCCL_BUFFSIZE=128
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
unset VLLM_ASCEND_ENABLE_FLASHCOMM1

export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Smoke-test / current workaround: bypass npu_quant_lightning_indexer in dsa_v1.py.
# Keep this only if your DFLASH_DISABLE_QLI patch is applied.
export DFLASH_DISABLE_QLI=1
export DSV4_VLLM_SERVE_PATCH=1

# ===================== CONFIG, identical on both nodes =====================
PARENT_IP="80.5.5.108"
CHILD_IP="80.5.5.109"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=8
TARGET_TP_SIZE=16
LOCAL_NPUS="0,1,2,3,4,5,6,7"

MODEL="/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16"
DATASET="/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend_full.jsonl"

DATA_OUT="/home/n84449292/m84379596/dflash_dsv4_col_multinode"
SHARED_STORAGE_PATH="/dev/shm/hidden_states"

MAX_SAMPLES="${MAX_SAMPLES:-32}"
SEQ_LENGTH=1024
EPOCHS=1
LR=6e-4
SEED=42
HS_CACHE_PATH="/home/n84449292/m84379596/dflash_dsv4_col_multinode_hidden_states_ms${MAX_SAMPLES}_sl${SEQ_LENGTH}_seed${SEED}"

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [ -n "${MAX_SAMPLES:-}" ] && (( MAX_SAMPLES % WORLD_SIZE != 0 )); then
    echo "ERROR: MAX_SAMPLES=$MAX_SAMPLES must be divisible by WORLD_SIZE=$WORLD_SIZE"
    echo "Use 32, 992, 1008, 1024, etc."
    exit 1
fi

SPECULATOR_TYPE="dflash"
BLOCK_SIZE=10
MAX_ANCHORS=128
VERIFIER_VOCAB=129280
DRAFT_VOCAB_SIZE=129280
# DRAFT_VOCAB_SIZE=32768
NUM_LAYERS=3
TARGET_LAYER_IDS="2 20 40"

# ============================================================================

export no_proxy="localhost,127.0.0.1,::1,${PARENT_IP},${CHILD_IP}"
export NO_PROXY="$no_proxy"

export DFLASH_TP_GATHER=1
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
# Hidden-state extraction can render seq + anchors + extra prompt/control tokens.
# Use a conservative target-engine context by default. Override if needed.
export DFLASH_TARGET_MAX_MODEL_LEN="${DFLASH_TARGET_MAX_MODEL_LEN:-2048}"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Cross-node Gloo/HCCL must bind to the routable NIC, not loopback.
NET_IFACE="$(ip -o -4 addr show | awk '/80\.5\.5\./{print $2; exit}')"
if [ -z "$NET_IFACE" ]; then
    echo "ERROR: no NIC with an 80.5.5.x address on this node."
    echo "Run: ip -o -4 addr show"
    exit 1
fi

echo "[node $NODE_RANK] binding distributed traffic to NIC: $NET_IFACE ($(ip -o -4 addr show dev "$NET_IFACE" | awk '{print $4}'))"

export GLOO_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"

cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators
mkdir -p logs

echo "=== [node $NODE_RANK] repo/import sanity ==="
echo "[node $NODE_RANK] parent branch: $(git -C "$SPEC_MAIN" branch --show-current) @ $(git -C "$SPEC_MAIN" rev-parse --short HEAD)"
echo "[node $NODE_RANK] speculators branch: $(git -C "$SPEC_MAIN/speculators" branch --show-current) @ $(git -C "$SPEC_MAIN/speculators" rev-parse --short HEAD)"
if [ "$(git -C "$SPEC_MAIN/speculators" branch --show-current)" != "spec_dsv4_test" ]; then
    echo "ERROR: speculators must be on spec_dsv4_test for this test run"
    exit 1
fi
python - <<'PY'
import sys
import torch
import torch_npu
import vllm
import vllm_ascend
import speculators
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
print("speculators:", speculators.__file__)
print("npu_count:", torch.npu.device_count())
PY

echo "=== [node $NODE_RANK] cleanup old local processes and hidden-state files ==="
pkill -9 -f "scripts/train.py" 2>/dev/null || true
pkill -9 -f "torchrun"        2>/dev/null || true
pkill -9 -f "EngineCore"      2>/dev/null || true

rm -rf "$SHARED_STORAGE_PATH"
mkdir -p "$SHARED_STORAGE_PATH"

echo "[node $NODE_RANK] /dev/shm usage after cleanup:"
df -h /dev/shm || true
du -sh "$SHARED_STORAGE_PATH" 2>/dev/null || true
sleep 2

# ---- each node tokenizes its OWN local copy ----
mkdir -p "$DATA_OUT"
mkdir -p "$HS_CACHE_PATH"

echo "=== [node $NODE_RANK] prepare_data, node-local copy at $DATA_OUT ==="
PREP_MAX_SAMPLES_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
    PREP_MAX_SAMPLES_ARGS=(--max-samples "$MAX_SAMPLES")
fi

echo "=== [node $NODE_RANK] prepare_data, node-local copy at $DATA_OUT ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$DATA_OUT" \
    "${PREP_MAX_SAMPLES_ARGS[@]}" \
    --seq-length "$SEQ_LENGTH" \
    --seed "$SEED" \
    --overwrite

rm -f "$DATA_OUT/d2t.npy" "$DATA_OUT/t2d.npy"
# Content fingerprint, order-independent over files.
FP=$(find "$DATA_OUT" -type f ! -name '.*' ! -path '*/checkpoints/*' -exec sha256sum {} \; \
     | awk '{print $1}' | sort | sha256sum | awk '{print $1}')

echo "=================================================================="
echo "[node $NODE_RANK] DATA_FINGERPRINT: $FP"
echo "  -> this MUST match the other node's fingerprint before trusting the run"
echo "=================================================================="

echo "=== [node $NODE_RANK] train: in-process co-located, target TP=$TARGET_TP_SIZE, draft DP=$((NNODES*NPROC_PER_NODE)) ==="
echo "[node $NODE_RANK] DEBUG_LOGS=$DEBUG_LOGS LOG_FILTER=$LOG_FILTER"
echo "[node $NODE_RANK] DFLASH_DISABLE_QLI=${DFLASH_DISABLE_QLI-UNSET}"
echo "[node $NODE_RANK] CACHE_HS_ONLY=$CACHE_HS_ONLY"
echo "[node $NODE_RANK] HS_CACHE_PATH=$HS_CACHE_PATH"
echo "[node $NODE_RANK] CLEAN_HS_CACHE=$CLEAN_HS_CACHE CLEAN_SHARED_STORAGE=$CLEAN_SHARED_STORAGE"
echo "[node $NODE_RANK] DFLASH_TARGET_MAX_MODEL_LEN=${DFLASH_TARGET_MAX_MODEL_LEN-UNSET}"
echo "[node $NODE_RANK] VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1-UNSET}"
echo "[node $NODE_RANK] ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT-UNSET}"
echo "[node $NODE_RANK] ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL-UNSET}"


TARGET_ENGINE_ARGS=()
CACHE_ARGS=()

if [ "$CACHE_HS_ONLY" = "1" ]; then
    echo "[node $NODE_RANK] CACHE_HS_ONLY=1: generate hidden-state cache only; no training"
    if [ "$CLEAN_HS_CACHE" = "auto" ] || [ "$CLEAN_HS_CACHE" = "1" ]; then
        echo "[node $NODE_RANK] cleaning hidden-state cache before cache build: $HS_CACHE_PATH"
        rm -rf "$HS_CACHE_PATH"
    else
        echo "[node $NODE_RANK] keeping existing hidden-state cache: $HS_CACHE_PATH"
    fi
    mkdir -p "$HS_CACHE_PATH"

    if [ "$CLEAN_SHARED_STORAGE" = "1" ]; then
        echo "[node $NODE_RANK] cleaning vLLM hidden-state scratch storage: $SHARED_STORAGE_PATH"
        rm -rf "$SHARED_STORAGE_PATH"
        mkdir -p "$SHARED_STORAGE_PATH"
    fi
    TARGET_ENGINE_ARGS=(
        --in-process-target
        --target-tp-size "$TARGET_TP_SIZE"
        --enable-expert-parallel
        --gpu-memory-utilization 0.75
        --shared-storage-path "$SHARED_STORAGE_PATH"
    )
    CACHE_ARGS=(
        --hidden-states-path "$HS_CACHE_PATH"
        --on-missing generate
        --on-generate cache
        --cache-hidden-states-only
    )
else
    echo "[node $NODE_RANK] training from cached hidden states only; in-process target disabled"

    if [ "$CLEAN_HS_CACHE" = "1" ]; then
        echo "ERROR: CLEAN_HS_CACHE=1 would delete the cache before training."
        echo "Use CACHE_HS_ONLY=1 to rebuild cache, then run training without CLEAN_HS_CACHE=1."
        exit 1
    fi

    CACHE_COUNT=$(find "$HS_CACHE_PATH" -maxdepth 1 -name 'hs_*.safetensors' 2>/dev/null | wc -l)
    echo "[node $NODE_RANK] cached hidden-state files found: $CACHE_COUNT in $HS_CACHE_PATH"
    if [ "$CACHE_COUNT" -eq 0 ]; then
        echo "ERROR: no hidden-state cache files found for MAX_SAMPLES=$MAX_SAMPLES."
        echo "Run cache phase first:"
        echo "  CACHE_HS_ONLY=1 MAX_SAMPLES=$MAX_SAMPLES bash examples/train/dflash_dsv4_284b_col_multinode.sh $NODE_RANK"
        exit 1
    fi

    TARGET_ENGINE_ARGS=()
    CACHE_ARGS=(
        --hidden-states-path "$HS_CACHE_PATH"
        --on-missing raise
        --on-generate cache
    )
fi

run_train() {
    ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" torchrun \
        --nnodes "$NNODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$PARENT_IP" \
        --master_port "$MASTER_PORT" \
        --nproc_per_node "$NPROC_PER_NODE" \
        scripts/train.py \
        "${TARGET_ENGINE_ARGS[@]}" \
        --verifier-name-or-path "$MODEL" \
        --data-path "$DATA_OUT" \
        --save-path "$DATA_OUT/checkpoints" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --total-seq-len "$SEQ_LENGTH" \
        --speculator-type "$SPECULATOR_TYPE" \
        --block-size "$BLOCK_SIZE" \
        --max-anchors "$MAX_ANCHORS" \
        --num-layers "$NUM_LAYERS" \
        --target-layer-ids $TARGET_LAYER_IDS \
        --draft-arch qwen3 \
        --draft-hidden-act silu \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --draft-attn-impl sdpa \
        --mask-token-id 1 \
        --noise-std 0.0 \
        --scheduler-type cosine \
        --logger tensorboard \
        --run-name dflash_colo_2node \
        --log-dir ./logs/colo_2node \
        "${CACHE_ARGS[@]}" \
        --log-freq 10 \
        --checkpoint-freq 1.0 \
        --num-workers "$TRAIN_NUM_WORKERS" \
        --no-resume-from-checkpoint \
        --seed "$SEED"
}
# --no-resume-from-checkpoint
if [ "$LOG_FILTER" = "1" ]; then
    # Preserve torchrun exit status while filtering high-volume known-noise lines.
    set +e
    run_train 2>&1 | stdbuf -oL grep -Ev "$QUIET_LOG_FILTER"
    TORCH_STATUS=${PIPESTATUS[0]}
    set -e

    if [ "$TORCH_STATUS" -ne 0 ]; then
        echo "[node $NODE_RANK] torchrun failed with exit code $TORCH_STATUS"
        echo "[node $NODE_RANK] To capture raw logs, rerun with: LOG_FILTER=0 or DEBUG_LOGS=1"
        exit "$TORCH_STATUS"
    fi
else
    run_train
fi

echo "[node $NODE_RANK] done. Checkpoints, rank 0 local home: $DATA_OUT/checkpoints/"
