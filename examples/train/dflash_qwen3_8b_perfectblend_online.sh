#!/bin/bash
# ============================================================================
# DFlash SEPARATE (online) trainer — vLLM hidden-states server + FSDP trainer
# on disjoint NPUs (server 0-3 DP=4, trainer 4-7 DP=4).
#
# MATCHED PAIR with dflash_colocated.sh: identical vocab, LR, seq, block, layers,
# epochs, seed, mask, noise, scheduler. Intended difference: plumbing (live
# server vs in-process) -> trainer DP=4 here (vs 8). DP left as-is on purpose.
#
# NOTE: the curl health URL and --vllm-endpoint are PLAIN URLs (an earlier paste
# had them wrapped in markdown "[...](...)", which breaks curl and the trainer).
# ============================================================================
set -eo pipefail
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
unset DFLASH_TP_GATHER                          # online mode: gather/scatter MUST be OFF
set -u
cd /home/n84449292/m84379596/DFlash/vLLM_NPU/speculators

# ============ Configuration (matched to the co-located run) ============
MODEL="/share/canada_group_folder/ckpt/Qwen3-8B"
DATASET="/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl"
OUTPUT_DIR="./output/dflash_separate"
VLLM_PORT=8000
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

VLLM_NPUS="0,1,2,3"         # 4 NPUs serve hidden states (DP=4)
TRAIN_NPUS="4,5,6,7"        # 4 NPUs train (draft DP=4)
NUM_TRAIN_NPUS=4
VLLM_DP=4
# =======================================================================

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

echo "=== Step 2: launch vLLM server (NPUs $VLLM_NPUS) ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
       --max-model-len 4096 \
       --gpu-memory-utilization 0.85 &

VLLM_PID=$!
cleanup() { echo "Stopping vLLM..."; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "Waiting for server..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do sleep 2; done
echo "Server ready."

export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1

echo "=== Step 3: train (online, NPUs $TRAIN_NPUS, draft DP=4) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
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
    --run-name dflash_separate \
    --log-dir ./logs/separate \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 180 \
    --max-retries 8 \
    --log-freq 10 \
    --seed "$SEED"

echo "Done. Checkpoints: $OUTPUT_DIR/checkpoints/  |  TB logs: ./logs/separate"
