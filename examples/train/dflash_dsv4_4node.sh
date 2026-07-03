#!/bin/bash
# ============================================================================
# DFlash CO-LOCATED, 4-NODE — target TP=32 + draft DP=32 across 32 NPUs.
# Nodes:
#   rank 0: 80.5.5.108
#   rank 1: 80.5.5.109
#   rank 2: 80.5.5.136
#   rank 3: 80.5.5.60
#
# Run the SAME file on all four nodes. It auto-detects local 80.5.5.x IP.
# Optional: pass node rank explicitly; script verifies it matches the local IP.
# bash examples/train/dflash_dsv4_4node.sh   2>&1 | tee ./logs/train_dsv4_4node_rank0_109.log
# ============================================================================

set -eo pipefail

# --------------------- Global 4-node topology ---------------------
PARENT_IP="80.5.5.108"
ALL_NODE_IPS_CSV="80.5.5.108,80.5.5.109,80.5.5.136,80.5.5.60"
MASTER_PORT="${MASTER_PORT:-29500}"
FINGERPRINT_PORT="${FINGERPRINT_PORT:-29501}"
NNODES=4
NPROC_PER_NODE=8
WORLD_SIZE_TOTAL=$((NNODES * NPROC_PER_NODE))

# For this current train.py path, keep target DP=1 and make target TP == world size.
# vLLM effective EP = TP * target DP = 32.
TARGET_TP_SIZE=32
TARGET_DP_SIZE=1
EFFECTIVE_EP_SIZE=$((TARGET_TP_SIZE * TARGET_DP_SIZE))
DRAFT_DP_SIZE="$WORLD_SIZE_TOTAL"
LOCAL_NPUS="0,1,2,3,4,5,6,7"

# --------------------- Detect local node ---------------------
LOCAL_IP="$(ip -o -4 addr show | awk '/80\.5\.5\./{split($4,a,"/"); print a[1]; exit}')"
NET_IFACE="$(ip -o -4 addr show | awk '/80\.5\.5\./{print $2; exit}')"

if [ -z "${LOCAL_IP:-}" ] || [ -z "${NET_IFACE:-}" ]; then
    echo "ERROR: no NIC with an 80.5.5.x address on this node."
    echo "Run: ip -o -4 addr show"
    exit 1
fi

case "$LOCAL_IP" in
    80.5.5.108)
        NODE_RANK_DEFAULT=0
        BASE="/home/n84449292/m84379596"
        REPO="$BASE/DFlash/vLLM_NPU"
        CANN_TOOLKIT_ENV="$BASE/CANN/CANN9.0.0/ascend-toolkit/set_env.sh"
        CANN_ATB_ENV="$BASE/CANN/CANN9.0.0/nnal/atb/set_env.sh"
        MODEL_DEFAULT="$BASE/Huggingface/DeepSeek-V4-Flash-bf16"
        DATASET_DEFAULT="$BASE/Huggingface/datasets/open_perfectblend_full.jsonl"
        EXTRA_LD_LIBRARY_PATH=""
        ;;
    80.5.5.109)
        NODE_RANK_DEFAULT=1
        BASE="/home/n84449292/m84379596"
        REPO="$BASE/DFlash/vLLM_NPU"
        CANN_TOOLKIT_ENV="$BASE/CANN/CANN9.0.0/ascend-toolkit/set_env.sh"
        CANN_ATB_ENV="$BASE/CANN/CANN9.0.0/nnal/atb/set_env.sh"
        MODEL_DEFAULT="$BASE/Huggingface/DeepSeek-V4-Flash-bf16"
        DATASET_DEFAULT="$BASE/Huggingface/datasets/open_perfectblend_full.jsonl"
        EXTRA_LD_LIBRARY_PATH=""
        ;;
    80.5.5.136)
        NODE_RANK_DEFAULT=2
        BASE="/home/f00518697/m84379596"
        REPO="$BASE/DFlash/vLLM_NPU"
        CANN_TOOLKIT_ENV="/home/a00652497/CANN/9.0.0.0430/ascend-toolkit/set_env.sh"
        CANN_ATB_ENV="/home/a00652497/CANN/9.0.0.0430/nnal/atb/set_env.sh"
        MODEL_DEFAULT="/mnt/old_home/Huggingface/DeepSeek-V4-Flash-bf16"
        # Prefer the same dataset name as the other nodes. If it is absent, set DATASET explicitly.
        DATASET_DEFAULT="$BASE/Huggingface/datasets/open_perfectblend_full.jsonl"
        EXTRA_LD_LIBRARY_PATH="/home/a00652497/CANN/9.0.0.0430/cann-9.0.0/aarch64-linux/lib64"
        ;;
    80.5.5.60)
        NODE_RANK_DEFAULT=3
        BASE="/home/f00518697/m84379596"
        REPO="$BASE/DFlash/vLLM_NPU"
        # Mixed CANN layout on 60: toolkit from CANN9.0.0_full, ATB from CANN9.0.0.
        CANN_TOOLKIT_ENV="$BASE/CANN/CANN9.0.0_full/ascend-toolkit/set_env.sh"
        CANN_ATB_ENV="$BASE/CANN/CANN9.0.0/nnal/atb/set_env.sh"
        MODEL_DEFAULT="$BASE/Huggingface/DeepSeek-V4-Flash-bf16"
        DATASET_DEFAULT="$BASE/Huggingface/datasets/open_perfectblend_full.jsonl"
        EXTRA_LD_LIBRARY_PATH=""
        ;;
    *)
        echo "ERROR: unsupported local 80.5.5.x IP: $LOCAL_IP"
        echo "Expected one of: $ALL_NODE_IPS_CSV"
        exit 1
        ;;
esac

NODE_RANK="${1:-$NODE_RANK_DEFAULT}"
if [ "$NODE_RANK" != "$NODE_RANK_DEFAULT" ]; then
    echo "ERROR: local IP $LOCAL_IP should use node_rank=$NODE_RANK_DEFAULT, but got node_rank=$NODE_RANK"
    echo "Use ALLOW_NODE_RANK_OVERRIDE=1 only if you are intentionally overriding this."
    if [ "${ALLOW_NODE_RANK_OVERRIDE:-0}" != "1" ]; then
        exit 2
    fi
fi

# --------------------- Source CANN / ATB and local repo packages ---------------------
if [ ! -f "$CANN_TOOLKIT_ENV" ]; then
    echo "ERROR: missing toolkit env: $CANN_TOOLKIT_ENV"
    exit 1
fi
if [ ! -f "$CANN_ATB_ENV" ]; then
    echo "ERROR: missing ATB env: $CANN_ATB_ENV"
    exit 1
fi

source "$CANN_TOOLKIT_ENV"
source "$CANN_ATB_ENV"

if [ -n "$EXTRA_LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="$REPO/vllm:$REPO/vllm-ascend:$REPO/speculators:$REPO/speculators/src:${PYTHONPATH:-}"

# --------------------- Logging controls ---------------------
DEBUG_LOGS="${DEBUG_LOGS:-0}"
LOG_FILTER="${LOG_FILTER:-1}"

if [ "$DEBUG_LOGS" = "1" ]; then
    export ASCEND_LAUNCH_BLOCKING=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export VLLM_LOGGING_LEVEL=INFO
    unset PYTHONWARNINGS
    LOG_FILTER=0
    echo "[node $NODE_RANK/$LOCAL_IP] DEBUG_LOGS=1: verbose Ascend/vLLM logging enabled"
else
    unset ASCEND_LAUNCH_BLOCKING
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=4
    export ASCEND_GLOBAL_EVENT_ENABLE=0
    export VLLM_LOGGING_LEVEL=WARNING
    export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning"
    echo "[node $NODE_RANK/$LOCAL_IP] quiet logging enabled; use DEBUG_LOGS=1 for full logs"
fi

QUIET_LOG_FILTER='TypedStorage is deprecated|pin_memory.py:57|Qwen2VLImageProcessorFast|`rope_parameters`|Get a block from the existing pool failed|This error log can be ignored|Dumping input data for V1 LLM engine|Dumping scheduler output for model execution|^\[INFO\] (DRV|HCCL|HCCP|ASCENDCL)\(|^\[WARNING\] .*warnings.py:110|^\[INFO\] RUNTIME\(.*SetWatchDogDevStatus|^\[INFO\] HCCL\(.*HCCL_TRACE'

# --------------------- Ascend / vLLM-Ascend env ---------------------
export HCCL_BUFFSIZE=128
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
unset VLLM_ASCEND_ENABLE_FLASHCOMM1

export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DFLASH_DISABLE_QLI=1
export DSV4_VLLM_SERVE_PATCH=1

export no_proxy="localhost,127.0.0.1,::1,${ALL_NODE_IPS_CSV}"
export NO_PROXY="$no_proxy"

export DFLASH_TP_GATHER=1
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

export GLOO_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"

# --------------------- Training config ---------------------
MODEL="${MODEL:-$MODEL_DEFAULT}"
DATASET="${DATASET:-$DATASET_DEFAULT}"
DATA_OUT="${DATA_OUT:-$BASE/dflash_dsv4_col_4node}"
SHARED_STORAGE_PATH="${SHARED_STORAGE_PATH:-/dev/shm/hidden_states_dflash_4node}"

# Keep MAX_SAMPLES=1000 for first 4-node smoke run. For full data, launch with: MAX_SAMPLES= bash ...
MAX_SAMPLES="${MAX_SAMPLES-1000}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-6e-4}"
SEED="${SEED:-42}"

SPECULATOR_TYPE="dflash"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_ANCHORS="${MAX_ANCHORS:-128}"
VERIFIER_VOCAB=129280
DRAFT_VOCAB_SIZE=129280
NUM_LAYERS=3
TARGET_LAYER_IDS="2 20 40"

# --------------------- Preflight checks ---------------------
echo "[node $NODE_RANK/$LOCAL_IP] binding distributed traffic to NIC: $NET_IFACE ($(ip -o -4 addr show dev "$NET_IFACE" | awk '{print $4}'))"
echo "[node $NODE_RANK/$LOCAL_IP] TP=$TARGET_TP_SIZE target_DP=$TARGET_DP_SIZE effective_EP=$EFFECTIVE_EP_SIZE draft_DP=$DRAFT_DP_SIZE world=$WORLD_SIZE_TOTAL"
echo "[node $NODE_RANK/$LOCAL_IP] REPO=$REPO"
echo "[node $NODE_RANK/$LOCAL_IP] MODEL=$MODEL"
echo "[node $NODE_RANK/$LOCAL_IP] DATASET=$DATASET"
echo "[node $NODE_RANK/$LOCAL_IP] DATA_OUT=$DATA_OUT"

if [ ! -d "$REPO/speculators" ]; then
    echo "ERROR: missing repo/speculators directory: $REPO/speculators"
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "ERROR: missing model directory: $MODEL"
    exit 1
fi
if [ ! -f "$DATASET" ]; then
    echo "ERROR: missing dataset file: $DATASET"
    echo "All four nodes must use the same dataset content. Set DATASET=/path/to/file if needed."
    exit 1
fi

PATCH_FILE="$REPO/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py"
if ! grep -q 'cpu:gloo,npu:hccl' "$PATCH_FILE"; then
    echo "ERROR: missing cpu:gloo,npu:hccl patch in $PATCH_FILE"
    echo "Patch all nodes before 4-node training."
    exit 1
fi

REPO_CHECK="$REPO" python - <<'PY'
import os
import torch
import torch_npu
import vllm
import vllm_ascend
repo = os.environ["REPO_CHECK"]
print("[env-check] torch:", torch.__version__, torch.__file__)
print("[env-check] torch_npu:", torch_npu.__version__, torch_npu.__file__)
print("[env-check] npu count:", torch.npu.device_count())
print("[env-check] vllm:", vllm.__file__)
print("[env-check] vllm_ascend:", vllm_ascend.__file__)
if not vllm.__file__.startswith(repo + "/vllm/"):
    raise SystemExit(f"ERROR: vllm is not loaded from local repo {repo}/vllm")
if not vllm_ascend.__file__.startswith(repo + "/vllm-ascend/"):
    raise SystemExit(f"ERROR: vllm_ascend is not loaded from local repo {repo}/vllm-ascend")
if torch.npu.device_count() < 8:
    raise SystemExit("ERROR: fewer than 8 NPUs visible")
PY

cd "$REPO/speculators"
mkdir -p logs

# --------------------- Cleanup ---------------------
echo "=== [node $NODE_RANK/$LOCAL_IP] cleanup old local processes and hidden-state files ==="
pkill -9 -f "scripts/train.py" 2>/dev/null || true
pkill -9 -f "torchrun"        2>/dev/null || true
pkill -9 -f "EngineCore"      2>/dev/null || true

rm -rf "$SHARED_STORAGE_PATH"
mkdir -p "$SHARED_STORAGE_PATH"

echo "[node $NODE_RANK/$LOCAL_IP] /dev/shm usage after cleanup:"
df -h /dev/shm || true
du -sh "$SHARED_STORAGE_PATH" 2>/dev/null || true
sleep 2

# --------------------- Prepare data ---------------------
mkdir -p "$DATA_OUT"

PREP_MAX_SAMPLES_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
    PREP_MAX_SAMPLES_ARGS=(--max-samples "$MAX_SAMPLES")
fi

echo "=== [node $NODE_RANK/$LOCAL_IP] prepare_data, node-local copy at $DATA_OUT ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$DATA_OUT" \
    "${PREP_MAX_SAMPLES_ARGS[@]}" \
    --seq-length "$SEQ_LENGTH" \
    --seed "$SEED" \
    --overwrite

rm -f "$DATA_OUT/d2t.npy" "$DATA_OUT/t2d.npy"
FP=$(DATA_OUT="$DATA_OUT" python - <<'PYFP'
import hashlib
import json
import os
from datasets import load_from_disk

def norm(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, bytes):
        return list(x)
    if isinstance(x, dict):
        return {k: norm(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    return x

obj = load_from_disk(os.environ["DATA_OUT"])
h = hashlib.sha256()

# Usually this is a Dataset, but handle DatasetDict too.
if hasattr(obj, "keys") and not hasattr(obj, "num_rows"):
    items = [(split, obj[split]) for split in sorted(obj.keys())]
else:
    items = [("dataset", obj)]

for split, ds in items:
    h.update(split.encode("utf-8"))
    h.update(b"\n")
    h.update(str(len(ds)).encode("utf-8"))
    h.update(b"\n")

    for row in ds:
        for key in sorted(row.keys()):
            h.update(key.encode("utf-8"))
            h.update(b"=")
            h.update(json.dumps(norm(row[key]), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            h.update(b"\n")
        h.update(b"\0")

print(h.hexdigest())
PYFP
)

cat <<MSG
==================================================================
[node $NODE_RANK/$LOCAL_IP] DATA_FINGERPRINT: $FP
  -> this MUST match all four nodes before trusting the run.
==================================================================
MSG

# --------------------- Cross-node fingerprint check ---------------------
cat > /tmp/dflash_check_fingerprint.py <<'PY'
import os
import sys
import torch.distributed as dist

dist.init_process_group("gloo")
rank = dist.get_rank()
world = dist.get_world_size()
item = {
    "rank": rank,
    "node_rank": os.environ.get("NODE_RANK"),
    "local_ip": os.environ.get("LOCAL_IP"),
    "fp": os.environ.get("DATA_FP"),
    "dataset": os.environ.get("DATASET"),
}
items = [None for _ in range(world)]
dist.all_gather_object(items, item)
mismatch = len({x["fp"] for x in items}) != 1
if rank == 0:
    print("=== 4-node DATA_FINGERPRINT check ===", flush=True)
    for x in sorted(items, key=lambda z: int(z["node_rank"])):
        print(f"node_rank={x['node_rank']} ip={x['local_ip']} fp={x['fp']} dataset={x['dataset']}", flush=True)
    if mismatch:
        print("ERROR: data fingerprints do not match across nodes.", flush=True)
    else:
        print("OK: all data fingerprints match.", flush=True)
dist.barrier()
dist.destroy_process_group()
sys.exit(3 if mismatch else 0)
PY

export NODE_RANK LOCAL_IP DATA_FP="$FP" DATASET

echo "=== [node $NODE_RANK/$LOCAL_IP] cross-node fingerprint check ==="
torchrun \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$PARENT_IP" \
    --master_port "$FINGERPRINT_PORT" \
    --nproc_per_node 1 \
    /tmp/dflash_check_fingerprint.py

# --------------------- Train ---------------------
echo "=== [node $NODE_RANK/$LOCAL_IP] train: target TP=$TARGET_TP_SIZE, target DP=$TARGET_DP_SIZE, EP=$EFFECTIVE_EP_SIZE, draft DP=$DRAFT_DP_SIZE ==="
echo "[node $NODE_RANK/$LOCAL_IP] DEBUG_LOGS=$DEBUG_LOGS LOG_FILTER=$LOG_FILTER"
echo "[node $NODE_RANK/$LOCAL_IP] DFLASH_DISABLE_QLI=${DFLASH_DISABLE_QLI-UNSET}"
echo "[node $NODE_RANK/$LOCAL_IP] VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1-UNSET}"
echo "[node $NODE_RANK/$LOCAL_IP] ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT-UNSET}"
echo "[node $NODE_RANK/$LOCAL_IP] ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL-UNSET}"

run_train() {
    ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" torchrun \
        --nnodes "$NNODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$PARENT_IP" \
        --master_port "$MASTER_PORT" \
        --nproc_per_node "$NPROC_PER_NODE" \
        scripts/train.py \
        --in-process-target \
        --target-tp-size "$TARGET_TP_SIZE" \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.6 \
        --shared-storage-path "$SHARED_STORAGE_PATH" \
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
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --draft-arch qwen3 \
        --draft-hidden-act silu \
        --mask-token-id 1 \
        --noise-std 0.0 \
        --scheduler-type cosine \
        --logger tensorboard \
        --run-name dflash_colo_4node \
        --log-dir ./logs/colo_4node \
        --on-missing generate \
        --on-generate delete \
        --log-freq 10 \
        --save-steps 1000 \
        --no-resume-from-checkpoint \
        --seed "$SEED"
}

if [ "$LOG_FILTER" = "1" ]; then
    set +e
    run_train 2>&1 | stdbuf -oL grep -Ev "$QUIET_LOG_FILTER"
    TORCH_STATUS=${PIPESTATUS[0]}
    set -e

    if [ "$TORCH_STATUS" -ne 0 ]; then
        echo "[node $NODE_RANK/$LOCAL_IP] torchrun failed with exit code $TORCH_STATUS"
        echo "[node $NODE_RANK/$LOCAL_IP] To capture raw logs, rerun with: LOG_FILTER=0 or DEBUG_LOGS=1"
        exit "$TORCH_STATUS"
    fi
else
    run_train
fi

echo "[node $NODE_RANK/$LOCAL_IP] done. Checkpoints, rank 0 local home: $DATA_OUT/checkpoints/"
