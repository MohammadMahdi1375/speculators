#!/bin/bash
# ============================================================================
# DFlash IN-PROCESS co-located trainer (speculators + vLLM external_launcher)
# Target + drafter in ONE torchrun job, sharing ONE HCCL group, on all 8 NPUs.
#
# MATCHED PAIR with dflash_separate.sh: identical vocab, LR, seq, block, layers,
# epochs, seed, mask, noise, scheduler. The ONLY intended difference is the
# plumbing: target in-process TP=8 on all 8 cards -> draft trainer DP=8.
# (separate uses DP=4). DP is structural and left as-is; neutralize it at
# analysis time with:  plot_dflash.py --xaxis tokens --tokens-per-step colo=... sep=...
# ============================================================================
set -eo pipefail
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
set -u
cd /home/n84449292/m84379596/DFlash/vLLM_NPU/speculators

# ============ Configuration (matched to SpecForge / to the separate run) ============
MODEL="/share/canada_group_folder/ckpt/Qwen3-8B"
DATASET="/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl"
OUTPUT_DIR="./output/dflash_colocated"
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

# VOCAB: set DRAFT_VOCAB_SIZE to the verifier full vocab (151936) to MATCH
# SpecForge, or to a smaller value (e.g. 8192) for a reduced head.
# NOTE: the DFlash code raises if draft_vocab_size == verifier vocab WITH
# mappings ("mappings not needed"), so FULL vocab must be requested by OMITTING
# the flag. This block does that automatically: it passes --draft-vocab-size
# only when the value is < verifier vocab; at == it omits the flag so the
# trainer falls back to the full vocab. Keep identical in both scripts.
VERIFIER_VOCAB=151936
DRAFT_VOCAB_SIZE=151936
VOCAB_FLAG=""
if [ "$DRAFT_VOCAB_SIZE" -lt "$VERIFIER_VOCAB" ]; then
    VOCAB_FLAG="--draft-vocab-size $DRAFT_VOCAB_SIZE"
fi

ALL_NPUS="0,1,2,3,4,5,6,7"
NUM_TRAIN_NPUS=8
TARGET_TP_SIZE=8
SHARED_STORAGE_PATH="/dev/shm/hidden_states"
# ===================================================================================

export DFLASH_TP_GATHER=1                       # REQUIRED for TP>1 co-location
export HCCL_CONNECT_TIMEOUT=1800
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1

pkill -9 -f "scripts/train.py" 2>/dev/null || true
pkill -9 -f "EngineCore"       2>/dev/null || true
rm -rf "$SHARED_STORAGE_PATH"
sleep 2

echo "=== Step 1: prepare_data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --overwrite

# Drop stale vocab mappings from any prior run so they REGENERATE at the
# DRAFT_VOCAB_SIZE set above (a cached 8192 d2t/t2d would otherwise size-mismatch).
rm -f "$OUTPUT_DIR"/d2t.npy "$OUTPUT_DIR"/t2d.npy

echo "=== Step 2: train (in-process co-located, 8 NPUs, draft DP=8) ==="
ASCEND_RT_VISIBLE_DEVICES="$ALL_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --in-process-target \
    --target-tp-size "$TARGET_TP_SIZE" \
    --shared-storage-path "$SHARED_STORAGE_PATH" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    $VOCAB_FLAG \
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
    --run-name dflash_colocated \
    --log-dir ./logs/colocated \
    --on-missing generate \
    --on-generate delete \
    --log-freq 10 \
    --seed "$SEED"

echo "Done. Checkpoints: $OUTPUT_DIR/checkpoints/  |  TB logs: ./logs/colocated"
