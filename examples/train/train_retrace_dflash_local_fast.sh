#!/usr/bin/env bash
# Local target + Speculators; opt-in continuous batching on logical NPUs 8–15.
set -eo pipefail
RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
RETRACE_PYTHON="${RETRACE_PYTHON:-$ENV_ROOT/bin/python}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$RETRACE_ROOT/vllm:$RETRACE_ROOT/vllm-ascend:$RETRACE_ROOT/speculators/src:$RETRACE_ROOT/speculators/hs_connectors/src"
export ASCEND_RT_VISIBLE_DEVICES="${RETRACE_NPUS:-8,9,10,11,12,13,14,15}"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
RETRACE_OUTPUT="${RETRACE_OUTPUT:-$RETRACE_ROOT/output/retrace_local_continuous_$(date +%Y%m%d_%H%M%S)}"
cd "$RETRACE_ROOT/speculators"
exec "$RETRACE_PYTHON" -m speculators.models.retrace.paper_recipe.utilization_launch \
  --root "$RETRACE_ROOT" --output "$RETRACE_OUTPUT" \
  --trainer-npus "$ASCEND_RT_VISIBLE_DEVICES" --target-backend local \
  --performance-mode cached --attention-backend sdpa --target-norm npu \
  --work-distribution dynamic --profile-performance \
  --rollout-scheduler continuous --rollout-batch-size "${RETRACE_ACTIVE_ROLLOUTS:-2}" \
  --trace-storage device --blocks-per-forward "${RETRACE_BLOCKS_PER_FORWARD:-16}" \
  --max-prompts 40000 --prompt-length 512 --response-length 1024 \
  --global-prompt-batch 32 --epochs 8 --lr 5e-5 --save-every 0 "$@"
