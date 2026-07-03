#!/bin/bash
# ============================================================================
# DFlash CO-LOCATED, 2-NODE — target TP=16 + draft DP=16 across 16 NPUs.
# HOME IS NODE-LOCAL on this cluster, so there is NO shared filesystem:
#   * Each node tokenizes its OWN identical local copy of the dataset.
#   * Hidden states stay on local /dev/shm (per-rank, per-node).
#   * Checkpoints are written by rank 0 to the PARENT's local home.
#
# RUN ON BOTH NODES:
#     parent 80.5.5.108:   bash dflash_colo_2node.sh 0
#     child  80.5.5.109:   bash dflash_colo_2node.sh 1
#
# BEFORE trusting the training run, compare the line
#     [node 0] DATA_FINGERPRINT: <hash>
#     [node 1] DATA_FINGERPRINT: <hash>
# from the two logs. They MUST be identical. If they differ, prepare_data is
# non-deterministic (shuffling) — stop and we pin its seed, or rsync node 0's
# copy to node 1 instead (see note at the bottom).
# ============================================================================
set -eo pipefail

NODE_RANK="${1:?usage: bash dflash_colo_2node.sh <node_rank 0|1>   (0=parent, 1=child)}"

source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

# ===================== CONFIG (identical on both nodes) =====================
PARENT_IP="80.5.5.108"        # = MASTER_ADDR (node 0)
CHILD_IP="80.5.5.109"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=8
TARGET_TP_SIZE=16
LOCAL_NPUS="0,1,2,3,4,5,6,7"

MODEL="/share/canada_group_folder/ckpt/Qwen3-8B"                              # shared, read-only OK
DATASET="/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl"  # shared, read-only OK

# NODE-LOCAL output (home is NOT shared); each node makes its own identical copy:
DATA_OUT="/home/n84449292/m84379596/dflash_colo_2node"
# Hidden states on local /dev/shm (per node):
SHARED_STORAGE_PATH="/dev/shm/hidden_states"

MAX_SAMPLES=10000
SEQ_LENGTH=3072
EPOCHS=1
LR=6e-4
SEED=42
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=16
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"
# VOCAB: full 151936 (matches SpecForge) — needs the model.py full-vocab patch;
# we OMIT --draft-vocab-size so the trainer uses the full verifier vocab.
# ============================================================================

export no_proxy="localhost,127.0.0.1,::1,${PARENT_IP},${CHILD_IP}"
export NO_PROXY="$no_proxy"
export DFLASH_TP_GATHER=1
export HCCL_CONNECT_TIMEOUT=1800
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
# Cross-node gloo/HCCL MUST bind to the routable NIC, not loopback. Both nodes are
# named "localhost" (resolves to ::1), which is why ranks tried to reach [::1] and got
# "Connection refused". Auto-detect the interface carrying the 80.5.5.x address here:
NET_IFACE="$(ip -o -4 addr show | awk '/80\.5\.5\./{print $2; exit}')"
if [ -z "$NET_IFACE" ]; then
    echo "ERROR: no NIC with an 80.5.5.x address on this node. Run 'ip -o -4 addr show'"
    echo "       and set NET_IFACE manually below."; exit 1
fi
echo "[node $NODE_RANK] binding distributed traffic to NIC: $NET_IFACE ($(ip -o -4 addr show dev "$NET_IFACE" | awk '{print $4}'))"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"

cd /home/n84449292/m84379596/DFlash/vLLM_NPU/speculators

pkill -9 -f "scripts/train.py" 2>/dev/null || true
pkill -9 -f "EngineCore"       2>/dev/null || true
rm -rf "$SHARED_STORAGE_PATH"
sleep 2

# ---- each node tokenizes its OWN local copy ----
mkdir -p "$DATA_OUT"
echo "=== [node $NODE_RANK] prepare_data (node-local copy at $DATA_OUT) ==="
python scripts/prepare_data.py \
    --model "$MODEL" --data "$DATASET" --output "$DATA_OUT" \
    --max-samples "$MAX_SAMPLES" --seq-length "$SEQ_LENGTH" --overwrite

# content fingerprint (order-independent over files) — compare across the two nodes
FP=$(find "$DATA_OUT" -type f ! -name '.*' ! -path '*/checkpoints/*' -exec sha256sum {} \; \
     | awk '{print $1}' | sort | sha256sum | awk '{print $1}')
echo "=================================================================="
echo "[node $NODE_RANK] DATA_FINGERPRINT: $FP"
echo "  -> this MUST match the other node's fingerprint before you trust the run"
echo "=================================================================="

echo "=== [node $NODE_RANK] Step 2: train (in-process co-located, TP=$TARGET_TP_SIZE, DP=$((NNODES*NPROC_PER_NODE))) ==="
ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" torchrun \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$PARENT_IP" \
    --master_port "$MASTER_PORT" \
    --nproc_per_node "$NPROC_PER_NODE" \
    scripts/train.py \
    --in-process-target \
    --target-tp-size "$TARGET_TP_SIZE" \
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
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --mask-token-id 151669 \
    --noise-std 0.0 \
    --scheduler-type cosine \
    --logger tensorboard \
    --run-name dflash_colo_2node \
    --log-dir ./logs/colo_2node \
    --on-missing generate \
    --on-generate delete \
    --log-freq 10 \
    --seed "$SEED"

echo "[node $NODE_RANK] done. Checkpoints (on node 0's local home): $DATA_OUT/checkpoints/"

# ---- If the two DATA_FINGERPRINT lines DIFFER, prepare_data shuffled. Two fixes: ----
#   (1) pin its seed / disable shuffle (send me prepare_data.py), OR
#   (2) tokenize ONLY on the parent, then on the CHILD copy it over instead of tokenizing:
#         rsync -a ${PARENT_IP}:${DATA_OUT}/ ${DATA_OUT}/
#       (requires inter-node ssh; guarantees byte-identical copies.)
