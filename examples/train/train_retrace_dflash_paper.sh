#!/usr/bin/env bash
# Pretrained DFlash + recurrent ReTrace: vLLM target and Speculators trainer.
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
# Keep CANN's Python bindings (including acl) after the selected repositories.
export PYTHONPATH="$RETRACE_ROOT/vllm:$RETRACE_ROOT/vllm-ascend:$RETRACE_ROOT/speculators/src:$RETRACE_ROOT/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
RETRACE_SERVER_NPUS="${RETRACE_SERVER_NPUS:-8,9}"
RETRACE_TRAINER_NPUS="${RETRACE_TRAINER_NPUS:-10,11,12,13,14,15}"
# The launcher sets a separate visibility mask for each child process group.
# RETRACE_NPUS from the old local-target launcher does not override this split.
export ASCEND_RT_VISIBLE_DEVICES="$RETRACE_SERVER_NPUS,$RETRACE_TRAINER_NPUS"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 OMP_PROC_BIND=false
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
RETRACE_OUTPUT="${RETRACE_OUTPUT:-$RETRACE_ROOT/output/retrace_paper_vllm_$(date +%Y%m%d_%H%M%S)}"
cd "$RETRACE_ROOT/speculators"
exec "$RETRACE_PYTHON" -m speculators.models.retrace.paper_recipe.launch \
  --root "$RETRACE_ROOT" --output "$RETRACE_OUTPUT" \
  --server-npus "$RETRACE_SERVER_NPUS" --trainer-npus "$RETRACE_TRAINER_NPUS" \
  --port "${RETRACE_PORT:-8523}" \
  --target-backend vllm --performance-mode cached --attention-backend sdpa \
  --target-norm reference --work-distribution dynamic --profile-performance \
  --max-prompts 40000 --prompt-length 512 --response-length 1024 \
  --global-prompt-batch 32 --epochs 8 --lr 5e-5 --save-every 0 "$@"
