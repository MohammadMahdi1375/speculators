#!/usr/bin/env bash
# Paired full-length training benchmark, preserving the original run/checkpoint.
set -eo pipefail
RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
RETRACE_PYTHON="${RETRACE_PYTHON:-/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python}"
RETRACE_CANN_ROOT="${RETRACE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
RETRACE_PERF_NPUS="${RETRACE_PERF_NPUS:-0,1,2,3,4,5,6,7}"
source "$RETRACE_CANN_ROOT/ascend-toolkit/set_env.sh"
source "$RETRACE_CANN_ROOT/nnal/atb/set_env.sh"
export PYTHONPATH="$RETRACE_ROOT/speculators/src${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="$RETRACE_PERF_NPUS"
export RETRACE_PERF_NPUS
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
exec "$RETRACE_PYTHON" -m speculators.models.retrace.performance_benchmark "$@"
