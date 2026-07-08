#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DeepSeek-V4 DFlash online trainer
#
# Run only on node 115.
# It trains the drafter and requests hidden states from the live vLLM server
# running on 108 + 109.
# =============================================================================

# ===================== paths/config to modify if needed =====================
SPEC_MAIN="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
CONDA_ENV="/home/n84449292/m84379596/conda/vllm-ascend-0202"

MODEL="/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16"
DATASET="/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend_full.jsonl"

SERVER_ENDPOINT="${SERVER_ENDPOINT:-http://80.5.5.108:30000/v1}"

# Must match the server ONLINE_HS_PATH.
ONLINE_HS_PATH="${ONLINE_HS_PATH:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/dflash_online_hidden_states/dsv4_284b}"

MAX_SAMPLES="${MAX_SAMPLES:-9920}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-6e-4}"
SEED="${SEED:-42}"

DATA_OUT="/home/n84449292/m84379596/dflash_dsv4_online_train115_ms${MAX_SAMPLES}_sl${SEQ_LENGTH}"

SPECULATOR_TYPE="dflash"
BLOCK_SIZE=10
MAX_ANCHORS=128
NUM_LAYERS=3

DRAFT_VOCAB_SIZE=129280
VERIFIER_VOCAB=129280

# Must match server TARGET_LAYER_IDS_JSON.
TARGET_LAYER_IDS="2 20 40"

TRAIN_NPUS="${TRAIN_NPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
# ============================================================================

# CANN env scripts are not safe under `set -u` because they reference
# variables such as PYTHONPATH/ZSH_VERSION before defining them.
unset PYTHONPATH
export PYTHONPATH=""
export ZSH_VERSION="${ZSH_VERSION:-}"

set +u
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
set -u

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"

cd "$SPEC_MAIN/speculators"
mkdir -p logs
mkdir -p "$DATA_OUT"
mkdir -p "$ONLINE_HS_PATH"

echo "=================================================================="
echo "[trainer 115] DeepSeek-V4 online DFlash training"
echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "DATA_OUT=$DATA_OUT"
echo "SERVER_ENDPOINT=$SERVER_ENDPOINT"
echo "ONLINE_HS_PATH=$ONLINE_HS_PATH"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "SEQ_LENGTH=$SEQ_LENGTH"
echo "EPOCHS=$EPOCHS"
echo "LR=$LR"
echo "TRAIN_NPUS=$TRAIN_NPUS"
echo "TARGET_LAYER_IDS=$TARGET_LAYER_IDS"
echo "=================================================================="

echo "=== preflight: server health ==="
curl -sf "http://80.5.5.108:30000/health" >/dev/null
curl -sf "http://80.5.5.108:30000/v1/models" | head -c 500 || true
echo
echo

echo "=== prepare data on trainer node ==="
PREP_MAX_SAMPLES_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
    PREP_MAX_SAMPLES_ARGS=(--max-samples "$MAX_SAMPLES")
fi

python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$DATA_OUT" \
    "${PREP_MAX_SAMPLES_ARGS[@]}" \
    --seq-length "$SEQ_LENGTH"

echo "=== create full-vocab identity mappings ==="
python - <<PY
import numpy as np
from pathlib import Path

p = Path("$DATA_OUT")
p.mkdir(parents=True, exist_ok=True)

vocab_size = int("$DRAFT_VOCAB_SIZE")
arr = np.arange(vocab_size, dtype=np.int64)

np.save(p / "d2t.npy", arr)
np.save(p / "t2d.npy", arr)

print("d2t:", p / "d2t.npy", arr.shape, arr.dtype)
print("t2d:", p / "t2d.npy", arr.shape, arr.dtype)
PY

echo "=== train drafter on 115 against live vLLM server ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone \
    --nproc_per_node "$NPROC_PER_NODE" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_OUT" \
    --vllm-endpoint "$SERVER_ENDPOINT" \
    --hidden-states-path "$ONLINE_HS_PATH" \
    --save-path "$DATA_OUT/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
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
    --draft-attn-impl sdpa \
    --mask-token-id 1 \
    --noise-std 0.0 \
    --scheduler-type cosine \
    --logger tensorboard \
    --run-name dflash_dsv4_online_115 \
    --log-dir ./logs/online_115 \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 1200 \
    --max-retries 2 \
    --log-freq 10 \
    --checkpoint-freq 1.0 \
    --num-workers "$TRAIN_NUM_WORKERS" \
    --no-resume-from-checkpoint \
    --seed "$SEED"

echo "Done. Checkpoints saved to $DATA_OUT/checkpoints"
