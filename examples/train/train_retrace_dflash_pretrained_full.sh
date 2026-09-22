#!/usr/bin/env bash
# Pretrained DFlash-b16 + ReTrace fine-tuning for the user's pinned Ascend stack.
# Defaults: 40k prompts, 512/1024 tokens, prompt batch 32, 8 epochs, LR 5e-5.
# Checkpoints: end of each epoch (--save-every 0), plus completion/planned pause.
# SDPA + dynamic prompt work assignment; validate with retrace_dflash_execution_check.sh.
# No random-drafter fallback. Existing runs should resume from a saved step.
set -eo pipefail

RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
RETRACE_PYTHON="${RETRACE_PYTHON:-/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python}"
RETRACE_TARGET="${RETRACE_TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
RETRACE_BASE_DFLASH="${RETRACE_BASE_DFLASH:-/home/n84449292/m84379596/Huggingface/Qwen3-4B-DFlash-b16}"
RETRACE_ARROW="${RETRACE_ARROW:-/home/n84449292/m84379596/Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
RETRACE_CANN_ROOT="${RETRACE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
RETRACE_TRAIN_OUTPUT="${RETRACE_TRAIN_OUTPUT:-$RETRACE_ROOT/output/retrace_dflash_pretrained_$(date +%Y%m%d_%H%M%S)}"

# CANN environment scripts are not nounset-safe.
source "$RETRACE_CANN_ROOT/ascend-toolkit/set_env.sh"
source "$RETRACE_CANN_ROOT/nnal/atb/set_env.sh"
set -u
export PYTHONPATH="$RETRACE_ROOT/speculators/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_USE_V2_MODEL_RUNNER=0

if [[ ! -x "$RETRACE_PYTHON" ]]; then
    printf 'Python executable missing: %s\n' "$RETRACE_PYTHON" >&2
    exit 2
fi

# Local backend: each worker holds a frozen target, using its incremental KV
# cache. Distributed gradients average 32 COMPLETE prompts per update.
# Frozen target uses NPU RMSNorm; --target-norm reference retains stock math.
# This script starts a NEW pretrained run. Use the generated resume_fast.sh to
# retain an existing checkpoint's optimizer, update count and original schedule.
# Unknown inherited RETRACE_MODE/DRAFT/STEPS/ACCUM values are not consumed.
exec "$RETRACE_PYTHON" -m speculators.models.retrace.pretrained_launch \
    --root "$RETRACE_ROOT" \
    --target "$RETRACE_TARGET" \
    --base-dflash "$RETRACE_BASE_DFLASH" \
    --data "$RETRACE_ARROW" \
    --output "$RETRACE_TRAIN_OUTPUT" \
    --target-backend local \
    --trainer-npus 0,1,2,3,4,5,6,7 \
    --max-prompts 40000 \
    --prompt-length 512 \
    --response-length 1024 \
    --global-prompt-batch 32 \
    --epochs 8 \
    --lr 5e-5 \
    --performance-mode cached \
    --trace-storage device \
    --blocks-per-forward 16 \
    --rollout-batch-size 1 \
    --attention-backend sdpa \
    --target-norm npu \
    --work-distribution dynamic \
    --save-every 0 \
    "$@"
