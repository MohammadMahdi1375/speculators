#!/usr/bin/env bash
# Stored responses + vLLM feature extraction + fresh verified ReTrace round pairs.
# A separate experiment: token-packed batches, not the old full-online protocol.
set -eo pipefail
RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
RETRACE_PYTHON="${RETRACE_PYTHON:-/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python}"
RETRACE_TARGET="${RETRACE_TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
RETRACE_BASE_DFLASH="${RETRACE_BASE_DFLASH:-    }"
RETRACE_ARROW="${RETRACE_ARROW:-/home/n84449292/m84379596/Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
RETRACE_CANN_ROOT="${RETRACE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
RETRACE_STORED_OUTPUT="${RETRACE_STORED_OUTPUT:-$RETRACE_ROOT/output/retrace_dflash_stored_$(date +%Y%m%d_%H%M%S)}"
source "$RETRACE_CANN_ROOT/ascend-toolkit/set_env.sh"
source "$RETRACE_CANN_ROOT/nnal/atb/set_env.sh"
set -u
# Preserve any user's LD_PRELOAD workaround; prefer the environment's C++ runtime.
RETRACE_ENV_LIB="$(dirname "$(dirname "$RETRACE_PYTHON")")/lib"
export LD_LIBRARY_PATH="$RETRACE_ENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Explicit source roots avoid importing the outer vllm repository directory as
# a namespace package before the editable-install finder can load __init__.py.
export PYTHONPATH="$RETRACE_ROOT/vllm:$RETRACE_ROOT/vllm-ascend:$RETRACE_ROOT/speculators/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VLLM_USE_V2_MODEL_RUNNER=0
cd "$RETRACE_ROOT/speculators"
exec "$RETRACE_PYTHON" -m speculators.models.retrace.stored_launch \
    --root "$RETRACE_ROOT" --target "$RETRACE_TARGET" \
    --base-dflash "$RETRACE_BASE_DFLASH" --data "$RETRACE_ARROW" \
    --output "$RETRACE_STORED_OUTPUT" \
    --server-npus "${RETRACE_STORED_SERVER_NPUS:-0,1}" \
    --trainer-npus "${RETRACE_STORED_TRAINER_NPUS:-2,3,4,5,6,7}" \
    "$@"
