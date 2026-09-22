#!/usr/bin/env bash
# Independent ReTrace: 2 target-server NPUs and 6 online trajectory workers.
set -eo pipefail
export RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
RETRACE_CANN_ROOT="${RETRACE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
source "$RETRACE_CANN_ROOT/ascend-toolkit/set_env.sh"
source "$RETRACE_CANN_ROOT/nnal/atb/set_env.sh"
set -u
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
export PYTHONPATH="$RETRACE_ROOT/speculators/src:$RETRACE_ROOT/speculators/hs_connectors/src:$RETRACE_ROOT/vllm:$RETRACE_ROOT/vllm-ascend:${PYTHONPATH:-}"
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}"
export NO_PROXY="$no_proxy"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "$RETRACE_ROOT/speculators"
exec "${RETRACE_PYTHON:-python}" -m speculators.models.retrace.launch "$@"
