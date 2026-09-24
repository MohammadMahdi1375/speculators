#!/bin/bash
# ============================================================================
# Online DFlash training on Ascend NPU
# Qwen3-4B + preprocessed Open-PerfectBlend + CANN 9.1.0
#
# Matched to the DSpark baseline:
#   - same target model
#   - same preprocessed dataset
#   - same sequence length
#   - same epoch count
#   - same learning rate
#   - same target hidden-state layers
#   - same full vocabulary
#   - same NPU allocation
#
# NPU split:
#   vLLM hidden-state server : NPUs 8,9
#   DFlash trainer           : NPUs 10,11,12,13,14,15
#
# IMPORTANT:
#   Dataset is already preprocessed with Hugging Face save_to_disk.
#   DO NOT run prepare-data again.
# ============================================================================

set -eo pipefail

# ============================================================================
# Environment
# ============================================================================

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

CANN_ROOT=/home/n84449292/m84379596/CANN/9.1.0

unset PYTHONPATH

source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"

unset ASCEND_LAUNCH_BLOCKING

# Prefer Conda C++ runtime over system libstdc++.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export SOC_VERSION=ascend910_9372

export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

# ============================================================================
# Disable proxies for local vLLM communication
# ============================================================================

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ftp_proxy
unset FTP_PROXY
unset all_proxy
unset ALL_PROXY

export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="$no_proxy"

unset DFLASH_TP_GATHER

set -u

cd "$SPEC_MAIN/speculators"

# ============================================================================
# Configuration
# ============================================================================

MODEL="/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"

# Already-preprocessed Open-PerfectBlend dataset.
DATASET="/home/n84449292/m84379596/Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072/"

# Use a separate output directory from DSpark.
OUTPUT_DIR="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_qwen3_4b_openperfectblend_bs16"

# ============================================================================
# vLLM hidden-state server
# ============================================================================

VLLM_PORT=8080

VLLM_ENDPOINT="http://127.0.0.1:${VLLM_PORT}"
VLLM_API_ENDPOINT="${VLLM_ENDPOINT}/v1"
VLLM_READY_ENDPOINT="${VLLM_API_ENDPOINT}/models"

# ============================================================================
# Training
# ============================================================================

SEQ_LENGTH=3072

# Same as DSpark baseline.
EPOCHS=5
LR=6e-4
SEED=42

VLLM_MAX_MODEL_LEN=$((SEQ_LENGTH + 256))

# ============================================================================
# DFlash
# ============================================================================

SPECULATOR_TYPE="dflash"

# Keep the same block size as the DSpark training configuration.
BLOCK_SIZE=16

# Same anchor budget as DSpark for a controlled comparison.
MAX_ANCHORS=512

NUM_LAYERS=5

# Must exactly match hidden-state layers exposed by launch_vllm.py.
TARGET_LAYER_IDS="1 9 17 25 33"

# ============================================================================
# Vocabulary
# ============================================================================

VERIFIER_VOCAB=151936
DRAFT_VOCAB_SIZE=151936

MASK_TOKEN_ID=151669

# ============================================================================
# NPU assignments
# ============================================================================

VLLM_NPUS="0,1"
VLLM_DP=2

TRAIN_NPUS="2,3,4,5,6,7"
NUM_TRAIN_NPUS=6

# ============================================================================
# Step 0: Environment / model / dataset sanity checks
# ============================================================================

echo
echo "============================================================"
echo " Step 0: Environment / dataset / vocab sanity check"
echo "============================================================"
echo

MODEL="$MODEL" \
EXPECTED_VOCAB="$VERIFIER_VOCAB" \
DRAFT_VOCAB_SIZE="$DRAFT_VOCAB_SIZE" \
DATASET="$DATASET" \
MASK_TOKEN_ID="$MASK_TOKEN_ID" \
python - <<'PY'
import json
import os
from pathlib import Path

import speculators

model = Path(os.environ["MODEL"])
dataset = Path(os.environ["DATASET"])

expected_vocab = int(os.environ["EXPECTED_VOCAB"])
draft_vocab = int(os.environ["DRAFT_VOCAB_SIZE"])
mask_token_id = int(os.environ["MASK_TOKEN_ID"])

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

config_path = model / "config.json"

if not config_path.exists():
    raise SystemExit(
        f"ERROR: verifier config.json does not exist:\n{config_path}"
    )

with config_path.open() as f:
    cfg = json.load(f)

cfg = cfg.get("text_config", cfg)

if "vocab_size" not in cfg:
    raise SystemExit("ERROR: vocab_size not found in verifier config.")

actual_vocab = int(cfg["vocab_size"])

print("Speculators path :", speculators.__file__)
print("Verifier vocab   :", actual_vocab)
print("Requested draft  :", draft_vocab)
print("Mask token ID    :", mask_token_id)

if actual_vocab != expected_vocab:
    raise SystemExit(
        f"ERROR: expected verifier vocab {expected_vocab}, "
        f"but config.json has {actual_vocab}"
    )

if draft_vocab != actual_vocab:
    raise SystemExit(
        f"ERROR: draft vocab {draft_vocab} does not equal "
        f"verifier vocab {actual_vocab}"
    )

if mask_token_id >= actual_vocab:
    raise SystemExit(
        f"ERROR: mask token ID {mask_token_id} is outside "
        f"vocabulary size {actual_vocab}"
    )

print()
print("Vocabulary check : OK")
print(f"Using FULL vocabulary: draft={draft_vocab}, verifier={actual_vocab}")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

if not dataset.exists():
    raise SystemExit(
        f"ERROR: dataset does not exist:\n{dataset}"
    )

dataset_info = dataset / "dataset_info.json"
state_file = dataset / "state.json"
arrow_files = sorted(dataset.glob("data-*.arrow"))

if not dataset_info.exists():
    raise SystemExit(
        "ERROR: dataset_info.json not found. "
        "Expected a Hugging Face save_to_disk dataset."
    )

if not state_file.exists():
    raise SystemExit(
        "ERROR: state.json not found. "
        "Expected a Hugging Face save_to_disk dataset."
    )

if not arrow_files:
    raise SystemExit("ERROR: no data-*.arrow files found.")

print()
print("Dataset type     : PREPROCESSED Hugging Face Arrow")
print("Arrow shards     :", len(arrow_files))
print("Dataset path     :", dataset)

with dataset_info.open() as f:
    info = json.load(f)

features = info.get("features", {})

print(
    "Dataset features :",
    ", ".join(features.keys()),
)

required = {
    "input_ids",
    "loss_mask",
}

missing = required - set(features)

if missing:
    raise SystemExit(
        f"ERROR: prepared dataset is missing required features: "
        f"{sorted(missing)}"
    )

print("Dataset check    : OK")
print("CANN/NPU mode    : Ascend")
PY

# ============================================================================
# Print configuration
# ============================================================================

echo
echo "============================================================"
echo " DFlash Training Configuration"
echo "============================================================"
echo

echo "MODEL                 : $MODEL"
echo "DATASET               : $DATASET"
echo "DATASET TYPE          : PREPROCESSED ARROW"
echo
echo "OUTPUT_DIR            : $OUTPUT_DIR"
echo
echo "CANN_ROOT             : $CANN_ROOT"
echo "SOC_VERSION           : $SOC_VERSION"
echo
echo "SPECULATOR_TYPE       : $SPECULATOR_TYPE"
echo "BLOCK_SIZE            : $BLOCK_SIZE"
echo "MAX_ANCHORS           : $MAX_ANCHORS"
echo
echo "NUM_LAYERS            : $NUM_LAYERS"
echo "TARGET_LAYER_IDS      : $TARGET_LAYER_IDS"
echo
echo "VERIFIER_VOCAB        : $VERIFIER_VOCAB"
echo "DRAFT_VOCAB_SIZE      : $DRAFT_VOCAB_SIZE"
echo "MASK_TOKEN_ID         : $MASK_TOKEN_ID"
echo
echo "SEQ_LENGTH            : $SEQ_LENGTH"
echo "EPOCHS                : $EPOCHS"
echo "LR                    : $LR"
echo
echo "vLLM NPUs             : $VLLM_NPUS"
echo "vLLM DP               : $VLLM_DP"
echo
echo "TRAIN NPUs            : $TRAIN_NPUS"
echo "NUM TRAIN NPUs        : $NUM_TRAIN_NPUS"
echo
echo "vLLM ENDPOINT         : $VLLM_ENDPOINT"
echo

mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

# ============================================================================
# Step 1: Check vLLM port
# ============================================================================

echo
echo "============================================================"
echo " Step 1: Checking vLLM port $VLLM_PORT"
echo "============================================================"
echo

if ss -ltnH 2>/dev/null \
    | awk '{print $4}' \
    | grep -Eq "(^|:)${VLLM_PORT}$"; then

    echo
    echo "ERROR: Port ${VLLM_PORT} is already occupied."
    echo

    ss -ltnp 2>/dev/null \
        | grep ":${VLLM_PORT}" \
        || true

    exit 1
fi

echo "Port ${VLLM_PORT} is free."

# ============================================================================
# Step 2: Launch hidden-state-aware vLLM server
# ============================================================================

echo
echo "============================================================"
echo " Step 2: Launching vLLM on NPUs $VLLM_NPUS"
echo "============================================================"
echo

VLLM_LOG="./logs/dflash_qwen3_4b_full_vocab_vllm_server.log"

echo "vLLM log -> $VLLM_LOG"
echo

setsid env \
    ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" \
    no_proxy="$no_proxy" \
    NO_PROXY="$NO_PROXY" \
    python scripts/launch_vllm.py "$MODEL" \
        --target-layer-ids $TARGET_LAYER_IDS \
        -- \
        --data-parallel-size "$VLLM_DP" \
        --port "$VLLM_PORT" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!

echo "vLLM PID/PGID: $VLLM_PID"

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    echo
    echo "============================================================"
    echo " Stopping DFlash hidden-state vLLM server"
    echo "============================================================"
    echo

    echo "PGID: $VLLM_PID"

    kill -TERM -- "-$VLLM_PID" 2>/dev/null || true

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM stopped."
            return
        fi

        sleep 1
    done

    echo "Force killing vLLM..."

    kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ============================================================================
# Wait for vLLM
# ============================================================================

echo
echo "Waiting for vLLM:"
echo "  $VLLM_READY_ENDPOINT"
echo

WAITED=0
BOOT_TIMEOUT=${VLLM_BOOT_TIMEOUT:-1800}

until curl \
    --noproxy "*" \
    --connect-timeout 5 \
    --max-time 10 \
    -sf \
    "$VLLM_READY_ENDPOINT" \
    >/dev/null 2>&1
do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo
        echo "ERROR: vLLM exited before becoming ready."
        echo

        tail -n 150 "$VLLM_LOG" || true
        exit 1
    fi

    if [ "$WAITED" -ge "$BOOT_TIMEOUT" ]; then
        echo
        echo "ERROR: vLLM startup timeout."
        echo

        tail -n 150 "$VLLM_LOG" || true
        exit 1
    fi

    sleep 5
    WAITED=$((WAITED + 5))

    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  ...still waiting (${WAITED}s)"

        if ss -ltnH 2>/dev/null \
            | awk '{print $4}' \
            | grep -Eq "(^|:)${VLLM_PORT}$"; then
            echo "     port ${VLLM_PORT}: LISTENING"
        else
            echo "     port ${VLLM_PORT}: not listening yet"
        fi
    fi
done

echo
echo "vLLM API ready after ${WAITED}s."

# ============================================================================
# API verification
# ============================================================================

HTTP_CODE=$(
    curl \
        --noproxy "*" \
        -s \
        -o /dev/null \
        -w "%{http_code}" \
        "$VLLM_READY_ENDPOINT"
)

echo "HTTP status: $HTTP_CODE"

if [ "$HTTP_CODE" != "200" ]; then
    echo
    echo "ERROR: vLLM API returned HTTP $HTTP_CODE"
    exit 1
fi

echo "vLLM API: OK"

# ============================================================================
# Step 3: Existing preprocessed dataset
# ============================================================================

echo
echo "============================================================"
echo " Step 3: Using existing preprocessed Open-PerfectBlend"
echo "============================================================"
echo

echo "Skipping prepare-data."
echo "Training data:"
echo "  $DATASET"
echo

# ============================================================================
# NPU compatibility
# ============================================================================

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# ============================================================================
# Step 4: Train DFlash
# ============================================================================

echo
echo "============================================================"
echo " Step 4: Training DFlash"
echo "============================================================"
echo

echo "Training NPUs          : $TRAIN_NPUS"
echo "Number of trainer NPUs : $NUM_TRAIN_NPUS"
echo
echo "Verifier vocabulary    : $VERIFIER_VOCAB"
echo "Draft vocabulary       : $DRAFT_VOCAB_SIZE"
echo

ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone \
    --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATASET" \
    --vllm-endpoint "$VLLM_API_ENDPOINT" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --checkpoint-freq 1 \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --draft-attn-impl sdpa \
    --mask-token-id "$MASK_TOKEN_ID" \
    --scheduler-type cosine \
    --logger tensorboard \
    --run-name dflash_qwen3_4b_full_vocab_openperfectblend_npu \
    --log-dir ./logs/dflash_qwen3_4b_full_vocab_openperfectblend_npu \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 180 \
    --max-retries 8 \
    --log-freq 10 \
    --seed "$SEED"

# ============================================================================
# Training finished
# ============================================================================

echo
echo "============================================================"
echo " DFlash training finished"
echo "============================================================"
echo

echo "Training data:"
echo "  $DATASET"
echo

echo "Checkpoints:"
echo "  $OUTPUT_DIR/checkpoints/"
echo

echo "TensorBoard logs:"
echo "  ./logs/dflash_qwen3_4b_full_vocab_openperfectblend_npu"
echo

echo "vLLM log:"
echo "  $VLLM_LOG"
echo

# ============================================================================
# Verify generated checkpoint config
# ============================================================================

LATEST_CONFIG=$(find "$OUTPUT_DIR/checkpoints" \
    -name config.json \
    -type f \
    2>/dev/null \
    | sort \
    | tail -n 1)

if [ -n "${LATEST_CONFIG:-}" ] && [ -f "$LATEST_CONFIG" ]; then

    echo "============================================================"
    echo " Verifying saved DFlash configuration"
    echo "============================================================"
    echo

    echo "Config:"
    echo "  $LATEST_CONFIG"
    echo

    EXPECTED_DRAFT_VOCAB="$DRAFT_VOCAB_SIZE" \
    CONFIG_PATH="$LATEST_CONFIG" \
    python - <<'PY'
import json
import os

path = os.environ["CONFIG_PATH"]
expected = int(os.environ["EXPECTED_DRAFT_VOCAB"])

with open(path) as f:
    cfg = json.load(f)

actual = cfg.get("draft_vocab_size")

print("Saved draft_vocab_size:", actual)

if actual is None:
    raise SystemExit(
        "ERROR: saved config does not contain draft_vocab_size."
    )

if int(actual) != expected:
    raise SystemExit(
        f"ERROR: expected saved draft_vocab_size={expected}, "
        f"but got {actual}"
    )

transformer_cfg = cfg.get(
    "transformer_layer_config",
    {},
)

transformer_vocab = transformer_cfg.get(
    "vocab_size"
)

print("Transformer vocab_size:", transformer_vocab)

if (
    transformer_vocab is not None
    and int(transformer_vocab) != expected
):
    raise SystemExit(
        f"ERROR: transformer_layer_config.vocab_size="
        f"{transformer_vocab}, expected {expected}"
    )

print()
print("Checkpoint vocabulary verification: OK")
print(f"draft_vocab_size = {expected}")
print(f"target vocab_size = {expected}")
PY

else
    echo
    echo "WARNING: Could not locate a saved config.json under:"
    echo "  $OUTPUT_DIR/checkpoints"
    echo
fi