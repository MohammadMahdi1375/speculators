#!/usr/bin/env bash
# Paired Table-2-style Qwen3-4B evaluation on one Ascend NPU, both temperatures.
# Usage: bash evaluate_retrace_dflash_table2.sh --manifest FILE --baseline DIR \
#          --retrace DIR --output NEW_DIR [--max-tokens 64 --temperatures 0]
set -eo pipefail
RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
RETRACE_PYTHON="${RETRACE_PYTHON:-/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python}"
RETRACE_TARGET="${RETRACE_TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
RETRACE_CANN_ROOT="${RETRACE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
source "$RETRACE_CANN_ROOT/ascend-toolkit/set_env.sh"
source "$RETRACE_CANN_ROOT/nnal/atb/set_env.sh"
set -u
export PYTHONPATH="$RETRACE_ROOT/speculators/src${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${RETRACE_EVAL_NPU:-0}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0
exec "$RETRACE_PYTHON" -m speculators.models.retrace.table2_benchmark suite \
    --target "$RETRACE_TARGET" "$@"
