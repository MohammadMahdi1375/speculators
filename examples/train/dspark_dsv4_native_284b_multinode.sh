#!/bin/bash
# ============================================================================
# Native DeepSeek-V4-Flash DSpark, 2-node NPU training.
#
# Parent 80.5.5.108:
#   bash examples/train/dspark_dsv4_native_284b_multinode.sh 0 cache
    # LOG_FILTER=0 \
    # MAX_ANCHORS=128 \
    # NUM_MTP_LAYERS=3 \
    # DSPARK_NATIVE_N_ROUTED_EXPERTS=256 \
    # DSPARK_NATIVE_N_ACTIVATED_EXPERTS=6 \
    # DSPARK_NATIVE_FSDP2_EXPERT_WRAP=1 \
    # DSPARK_NATIVE_TOUCH_ALL_EXPERTS=1 \
    # DSPARK_NATIVE_TOUCH_ALL_EXPERTS_MODE=forward \
    # DSPARK_NATIVE_TOUCH_CONFIDENCE_HEAD=1 \
    # DSPARK_NATIVE_TOUCH_CONFIDENCE_LOSS=1 \
    # DSPARK_NATIVE_TOUCH_FULL_LOGITS_LOSS=1 \
    # DSPARK_NATIVE_FORCE_VALID_ANCHOR=0 \
    # bash examples/train/dspark_dsv4_native_284b_multinode.sh 0 train \
    # 2>&1 | tee ./logs/native_train_dspark_full.log
#   bash examples/train/dspark_dsv4_native_284b_multinode.sh 0 export
#
# Child 80.5.5.109:
#   bash examples/train/dspark_dsv4_native_284b_multinode.sh 1 cache
#   bash examples/train/dspark_dsv4_native_284b_multinode.sh 1 train
#
# Fine-tune official mtp.* weights instead of scratch:
#   INIT_MTP_FROM=/home/.../DeepSeek-V4-Flash-DSpark bash ... 0 train
# ============================================================================

set -eo pipefail

NODE_RANK="${1:?usage: bash dspark_dsv4_native_284b_multinode.sh <node_rank 0|1> <cache|train|export|all>}"
PHASE="${2:-all}"

if [ "$NODE_RANK" != "0" ] && [ "$NODE_RANK" != "1" ]; then
    echo "ERROR: NODE_RANK must be 0 or 1, got: $NODE_RANK"
    exit 1
fi

case "$PHASE" in
    cache|train|export|all) ;;
    *) echo "ERROR: PHASE must be cache, train, export, or all; got: $PHASE"; exit 1 ;;
esac

# ===================== repo/env paths =====================
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_HOME="${CANN_HOME:-/home/n84449292/m84379596/CANN/CANN9.0.0}"
ENV_PREFIX="${ENV_PREFIX:-/home/n84449292/m84379596/conda/vllm-ascend-0202}"

PY="$ENV_PREFIX/bin/python"
TORCHRUN=("$PY" -m torch.distributed.run)

unset PYTHONPATH
source "$CANN_HOME/ascend-toolkit/set_env.sh"
source "$CANN_HOME/nnal/atb/set_env.sh"

export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export PATH="$ENV_PREFIX/bin:$PATH"

# ===================== logging controls =====================
DEBUG_LOGS="${DEBUG_LOGS:-0}"
LOG_FILTER="${LOG_FILTER:-1}"
if [ "$DEBUG_LOGS" = "1" ]; then
    export ASCEND_LAUNCH_BLOCKING=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export VLLM_LOGGING_LEVEL=INFO
    unset PYTHONWARNINGS
    LOG_FILTER=0
else
    unset ASCEND_LAUNCH_BLOCKING
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=4
    export ASCEND_GLOBAL_EVENT_ENABLE=0
    export VLLM_LOGGING_LEVEL=WARNING
    export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning"
fi

QUIET_LOG_FILTER='TypedStorage is deprecated|pin_memory.py:57|Qwen2VLImageProcessorFast|`rope_parameters`|Get a block from the existing pool failed|This error log can be ignored|Dumping input data for V1 LLM engine|Dumping scheduler output for model execution|^\[INFO\] (DRV|HCCL|HCCP|ASCENDCL)\(|^\[WARNING\] .*warnings.py:110|^\[INFO\] RUNTIME\(.*SetWatchDogDevStatus|^\[INFO\] HCCL\(.*HCCL_TRACE'

# ===================== Ascend / vLLM-Ascend =====================
export HCCL_BUFFSIZE=128
export HCCL_CONNECT_TIMEOUT=1800
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DFLASH_TP_GATHER=1
export DFLASH_DISABLE_QLI=1
export DSV4_VLLM_SERVE_PATCH=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# ===================== cluster =====================
PARENT_IP="80.5.5.108"
CHILD_IP="80.5.5.109"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES=2
NPROC_PER_NODE=8
TARGET_TP_SIZE=16
LOCAL_NPUS="0,1,2,3,4,5,6,7"

export no_proxy="localhost,127.0.0.1,::1,${PARENT_IP},${CHILD_IP}"
export NO_PROXY="$no_proxy"

if [ "$NODE_RANK" = "0" ]; then
    EXPECTED_IP="$PARENT_IP"
else
    EXPECTED_IP="$CHILD_IP"
fi
NET_IFACE="$(ip -o -4 addr show | awk -v ip="$EXPECTED_IP" '{split($4,a,"/"); if (a[1] == ip) {print $2; exit}}')"
if [ -z "$NET_IFACE" ]; then
    echo "ERROR: NODE_RANK=$NODE_RANK expects IP $EXPECTED_IP, but no NIC has that IP."
    ip -o -4 addr show
    exit 1
fi
LOCAL_IP="$(ip -o -4 addr show dev "$NET_IFACE" | awk -v ip="$EXPECTED_IP" '{split($4,a,"/"); if (a[1] == ip) {print a[1]; exit}}')"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_FAMILY=AF_INET
export HCCL_IF_IP="$LOCAL_IP"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-17777}"
export TP_SOCKET_IFNAME="$NET_IFACE"

# ===================== data/model =====================
MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16}"
DATASET="${DATASET:-/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend_full.jsonl}"
DATA_OUT="${DATA_OUT:-/home/n84449292/m84379596/dspark_dsv4_native_multinode}"
HIDDEN_STATES_PATH="$DATA_OUT/hidden_states"
SHARED_STORAGE_PATH="${SHARED_STORAGE_PATH:-/dev/shm/hidden_states}"

MAX_SAMPLES="${MAX_SAMPLES:-100000}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
SEED="${SEED:-42}"

# Official native DSpark config.
SPECULATOR_TYPE="dspark_dsv4_native"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-5}"
DSPARK_NOISE_TOKEN_ID="${DSPARK_NOISE_TOKEN_ID:-128799}"
NUM_MTP_LAYERS="${NUM_MTP_LAYERS:-3}"
export DSPARK_NATIVE_NUM_MTP_LAYERS="$NUM_MTP_LAYERS"
echo "[node $NODE_RANK] NUM_MTP_LAYERS=$NUM_MTP_LAYERS"
echo "[node $NODE_RANK] DSPARK_NATIVE_N_ROUTED_EXPERTS=${DSPARK_NATIVE_N_ROUTED_EXPERTS:-full}"
echo "[node $NODE_RANK] DSPARK_NATIVE_N_ACTIVATED_EXPERTS=${DSPARK_NATIVE_N_ACTIVATED_EXPERTS:-default}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-40 41 42}"
MARKOV_RANK="${MARKOV_RANK:-256}"
MAX_ANCHORS="${MAX_ANCHORS:-8}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
INIT_MTP_FROM="${INIT_MTP_FROM:-}"

cd "$SPEC_MAIN/speculators"
mkdir -p logs "$DATA_OUT" "$HIDDEN_STATES_PATH"

echo "[node $NODE_RANK] NIC=$NET_IFACE ($LOCAL_IP), SPEC_MAIN=$SPEC_MAIN"
echo "[node $NODE_RANK] PHASE=$PHASE MODEL=$MODEL DATA_OUT=$DATA_OUT"

echo "=== [node $NODE_RANK] import sanity check ==="
"$PY" - <<'PY'
import sys
import torch
import torch_npu
import vllm
import vllm_ascend
import speculators
import speculators.models.dspark_dsv4_native
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
print("speculators:", speculators.__file__)
print("native dspark import OK")
PY

echo "=== [node $NODE_RANK] cleanup ==="
pkill -9 -f "scripts/train.py"      2>/dev/null || true
pkill -9 -f "torchrun"              2>/dev/null || true
pkill -9 -f "torch.distributed.run" 2>/dev/null || true
pkill -9 -f "EngineCore"            2>/dev/null || true
rm -rf "$SHARED_STORAGE_PATH"
mkdir -p "$SHARED_STORAGE_PATH"

npu-smi info || true
IFS=',' read -ra DEV_ARR <<< "$LOCAL_NPUS"
if [ "${#DEV_ARR[@]}" -ne "$NPROC_PER_NODE" ]; then
    echo "ERROR: LOCAL_NPUS has ${#DEV_ARR[@]} entries but NPROC_PER_NODE=$NPROC_PER_NODE"
    exit 1
fi
for r in $(seq 0 $((NPROC_PER_NODE - 1))); do
    ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" TEST_LOCAL_RANK="$r" "$PY" - <<'PY'
import os, torch
r = int(os.environ["TEST_LOCAL_RANK"])
torch.accelerator.set_device_index(r)
print(f"OK local_rank={r}")
PY
done

if [ "$PHASE" = "cache" ] || [ "$PHASE" = "all" ]; then
    # Cache phase owns DATA_OUT and may overwrite it.
    # Train phase must NOT overwrite DATA_OUT because that deletes hidden_states/.
echo "=== [node $NODE_RANK] prepare_data ==="
"$PY" scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$DATA_OUT" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --seed "$SEED" \
    --overwrite
else
    echo "=== [node $NODE_RANK] Skipping prepare_data for PHASE=$PHASE; reusing $DATA_OUT ==="
    if [ ! -d "$DATA_OUT" ]; then
        echo "ERROR: DATA_OUT does not exist: $DATA_OUT"
        echo "Run cache phase first on both nodes."
        exit 1
    fi
fi


# Full-vocab native DSpark.  Do not train with draft/target vocab maps.
rm -f "$DATA_OUT/d2t.npy" "$DATA_OUT/t2d.npy"

FP=$(find "$DATA_OUT" -type f ! -name '.*' ! -path '*/checkpoints/*' ! -path '*/hidden_states/*' -exec sha256sum {} \; \
     | awk '{print $1}' | sort | sha256sum | awk '{print $1}')
echo "=================================================================="
echo "[node $NODE_RANK] DATA_FINGERPRINT: $FP"
echo "This should match the other node before trusting the run."
echo "=================================================================="

run_cache() {
    echo "=== [node $NODE_RANK] cache target hidden states, TP=$TARGET_TP_SIZE ==="
    # Cache uses lightweight DFlash speculator just to drive hidden-state extraction.
    # It does NOT train DFlash and exits before training.
    ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" "${TORCHRUN[@]}" \
        --nnodes "$NNODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$PARENT_IP" \
        --master_port "$MASTER_PORT" \
        --nproc_per_node "$NPROC_PER_NODE" \
        scripts/train.py \
        --in-process-target \
        --target-tp-size "$TARGET_TP_SIZE" \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.75 \
        --shared-storage-path "$SHARED_STORAGE_PATH" \
        --verifier-name-or-path "$MODEL" \
        --data-path "$DATA_OUT" \
        --hidden-states-path "$HIDDEN_STATES_PATH" \
        --save-path "$DATA_OUT/cache_dummy_checkpoints" \
        --epochs 1 \
        --lr 1e-6 \
        --total-seq-len "$SEQ_LENGTH" \
        --speculator-type dflash \
        --block-size "$DSPARK_BLOCK_SIZE" \
        --draft-vocab-size 129280 \
        --max-anchors "$MAX_ANCHORS" \
        --num-layers 1 \
        --target-layer-ids $TARGET_LAYER_IDS \
        --draft-attn-impl sdpa \
        --mask-token-id "$DSPARK_NOISE_TOKEN_ID" \
        --noise-std 0.0 \
        --on-missing generate \
        --on-generate cache \
        --cache-hidden-states-only \
        --logger tensorboard \
        --run-name native_dspark_cache \
        --log-dir ./logs/native_dspark_cache \
        --checkpoint-freq 1 \
        --no-resume-from-checkpoint \
        --seed "$SEED"
}

check_hidden_states_before_train() {
    echo "=== [node $NODE_RANK] checking cached hidden states before train ==="
    "$PY" - <<PY
from pathlib import Path
from datasets import load_from_disk
import sys

data_out = Path("$DATA_OUT")
hs = Path("$HIDDEN_STATES_PATH")

data = load_from_disk(str(data_out))
n = len(data)

have = set()
if hs.exists():
    for p in hs.glob("hs_*.safetensors"):
        try:
            have.add(int(p.stem.split("_")[1]))
        except Exception:
            pass

missing = [i for i in range(n) if i not in have]

print("DATA_OUT:", data_out)
print("HIDDEN_STATES_PATH:", hs)
print("dataset samples:", n)
print("hidden-state files found:", len(have))
print("missing count:", len(missing))
print("missing first 50:", missing[:50])

if missing:
    print("ERROR: hidden-state cache is incomplete. Run cache phase first and verify missing count is 0.")
    sys.exit(2)
PY
}

run_train() {
    echo "=== [node $NODE_RANK] train native DSpark mtp.* ==="
    check_hidden_states_before_train
    INIT_ARGS=()
    SHAPE_ARGS=(--num-layers "$NUM_MTP_LAYERS")
    if [ -n "$INIT_MTP_FROM" ]; then
        INIT_ARGS=(--from-pretrained "$INIT_MTP_FROM")
        # validate_draft_init_args rejects --from-pretrained + --num-layers.
        # The native loader defaults n_mtp_layers=3 from the DSpark config path.
        SHAPE_ARGS=()
    fi

    ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" "${TORCHRUN[@]}" \
        --nnodes "$NNODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$PARENT_IP" \
        --master_port "$MASTER_PORT" \
        --nproc_per_node "$NPROC_PER_NODE" \
        scripts/train.py \
        --verifier-name-or-path "$MODEL" \
        --data-path "$DATA_OUT" \
        --hidden-states-path "$HIDDEN_STATES_PATH" \
        --save-path "$DATA_OUT/checkpoints" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --total-seq-len "$SEQ_LENGTH" \
        --speculator-type "$SPECULATOR_TYPE" \
        --block-size "$DSPARK_BLOCK_SIZE" \
        --draft-vocab-size 129280 \
        --max-anchors "$MAX_ANCHORS" \
        --target-layer-ids $TARGET_LAYER_IDS \
        --mask-token-id "$DSPARK_NOISE_TOKEN_ID" \
        --markov-rank "$MARKOV_RANK" \
        --loss-fn kl_div \
        --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
        --noise-std 0.0 \
        --on-missing raise \
        --on-generate cache \
        --scheduler-type cosine \
        --logger tensorboard \
        --run-name native_dspark_dsv4_2node \
        --log-dir ./logs/native_dspark_dsv4_2node \
        --log-freq 10 \
        --checkpoint-freq 1 \
        --no-resume-from-checkpoint \
        --seed "$SEED" \
        "${SHAPE_ARGS[@]}" \
        "${INIT_ARGS[@]}"
}

run_export() {
    if [ "$NODE_RANK" != "0" ]; then
        echo "[node $NODE_RANK] export skipped; run export on parent only."
        return
    fi
    echo "=== [node $NODE_RANK] export HF-compatible DeepSeek-V4-Flash-DSpark checkpoint ==="
    if [ -n "${CHECKPOINT:-}" ]; then
        CKPT="$CHECKPOINT"
    else
        CKPT="$(find "$DATA_OUT/checkpoints" -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V | tail -1)"
        if [ -z "$CKPT" ]; then
            echo "ERROR: no numeric checkpoint directory found under $DATA_OUT/checkpoints"
            exit 1
        fi
    fi
    OUT="${EXPORT_OUT:-$DATA_OUT/hf_dsv4_flash_dspark_native}"
    "$PY" -m speculators.models.dspark_dsv4_native.export_hf \
        --target-model "$MODEL" \
        --checkpoint "$CKPT" \
        --output "$OUT" \
        --dspark-block-size "$DSPARK_BLOCK_SIZE" \
        --dspark-noise-token-id "$DSPARK_NOISE_TOKEN_ID" \
        --dspark-target-layer-ids $TARGET_LAYER_IDS \
        --dspark-markov-rank "$MARKOV_RANK" \
        --num-mtp-layers "$NUM_MTP_LAYERS"
}

run_filtered() {
    local fn="$1"
    if [ "$LOG_FILTER" = "1" ]; then
        set +e
        "$fn" 2>&1 | stdbuf -oL grep -Ev "$QUIET_LOG_FILTER"
        STATUS=${PIPESTATUS[0]}
        set -e
        if [ "$STATUS" -ne 0 ]; then
            echo "[node $NODE_RANK] $fn failed with exit code $STATUS"
            echo "[node $NODE_RANK] rerun with LOG_FILTER=0 or DEBUG_LOGS=1 for raw logs"
            exit "$STATUS"
        fi
    else
        "$fn"
    fi
}

case "$PHASE" in
    cache) run_filtered run_cache ;;
    train) run_filtered run_train ;;
    export) run_export ;;
    all) run_filtered run_cache; run_filtered run_train; run_export ;;
esac

echo "[node $NODE_RANK] done"
