#!/usr/bin/env bash
# Pretrained DSpark-block7 + ReTrace; separate from the DFlash training job.
# Full defaults: 40k prompts, global prompt batch 32, 8 epochs, LR 5e-5.
# Save at each epoch end plus completion/planned pause. No scratch fallback.
set -eo pipefail

RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
RETRACE_PYTHON="${RETRACE_PYTHON:-/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python}"
RETRACE_TARGET="${RETRACE_TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
RETRACE_DSPARK_BASE="${RETRACE_DSPARK_BASE:-/home/n84449292/m84379596/Huggingface/Qwen3-4B-DSpark-block7}"
RETRACE_ARROW="${RETRACE_ARROW:-/home/n84449292/m84379596/Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
RETRACE_CANN_ROOT="${RETRACE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
RETRACE_DSPARK_NPUS="${RETRACE_DSPARK_NPUS:-8,9,10,11,12,13,14,15}"
RETRACE_DSPARK_OUTPUT="${RETRACE_DSPARK_OUTPUT:-$RETRACE_ROOT/output/retrace_dspark_pretrained_$(date +%Y%m%d_%H%M%S)_$$}"

# CANN scripts are not nounset-safe; this shell cannot change another running job.
source "$RETRACE_CANN_ROOT/ascend-toolkit/set_env.sh"
source "$RETRACE_CANN_ROOT/nnal/atb/set_env.sh"
set -u
export PYTHONPATH="$RETRACE_ROOT/speculators/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Explicitly override any inherited DFlash visibility mask before Python imports.
export ASCEND_RT_VISIBLE_DEVICES="$RETRACE_DSPARK_NPUS"

if [[ ! -x "$RETRACE_PYTHON" ]]; then
    printf 'Python executable missing: %s\n' "$RETRACE_PYTHON" >&2
    exit 2
fi

# One frozen target per worker, local KV caches, no vLLM server/port is needed.
# DSpark-specific output/base variables deliberately do not consume DFlash values.
exec "$RETRACE_PYTHON" -m speculators.models.retrace_dspark.pretrained_launch \
    --root "$RETRACE_ROOT" \
    --target "$RETRACE_TARGET" \
    --base-dspark "$RETRACE_DSPARK_BASE" \
    --data "$RETRACE_ARROW" \
    --output "$RETRACE_DSPARK_OUTPUT" \
    --trainer-npus "$RETRACE_DSPARK_NPUS" \
    --max-prompts 40000 \
    --prompt-length 512 \
    --response-length 1024 \
    --global-prompt-batch 32 \
    --epochs 8 \
    --lr 5e-5 \
    --blocks-per-forward 2 \
    --save-every 0 \
    "$@"
