#!/usr/bin/env bash
set -eo pipefail

# Qwen3-4B target + your existing trained prefix draft. No retraining.
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="$ENV_ROOT/bin/python"

TARGET="${TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${DRAFT:-$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs8/tau_v2_scratch_20260928_082808_3142334/checkpoints/0}"
NPU="${NPU:-11}"
PORT="${PORT:-8211}"
MODE="${MODE:-fast}"                 # reference | validate | fast
VALIDATE_STEPS="${VALIDATE_STEPS:-32}"

case "$MODE" in
    reference|validate|fast) ;;
    *) echo "MODE must be reference, validate, or fast" >&2; exit 1 ;;
esac
if ! [[ "$VALIDATE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "VALIDATE_STEPS must be a positive integer" >&2
    exit 1
fi

# This metadata check does not import torch or touch NPU memory.
SPEC_CONFIG="$("$PYTHON_BIN" "$SPEC_MAIN/speculators/scripts/dflash_prefix_tau/inference_config.py" \
    --target "$TARGET" --draft "$DRAFT")"
echo "Prefix inference MODE=$MODE; NPU=$NPU; PORT=$PORT; initial checks=$VALIDATE_STEPS"
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    echo "Configuration valid. No server started."
    exit 0
fi

unset PYTHONPATH ASCEND_LAUNCH_BLOCKING
source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export SOC_VERSION=ascend910_9372
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export DFLASH_PREFIX_INFERENCE="$MODE"
export DFLASH_PREFIX_VALIDATE_STEPS="$VALIDATE_STEPS"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"

cd "$SPEC_MAIN/speculators"
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main \
    serve "$TARGET" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name qwen3-4b-dflash \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --seed 42 \
    --generation-config vllm \
    --max-num-seqs 1 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --gpu-memory-utilization 0.96 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --api-server-count 1 \
    --renderer-num-workers 1 \
    --additional-config '{"enable_reduce_sample": false}' \
    --enable-per-request-metrics \
    --per-request-spec-decode-metrics summary \
    --speculative-config "$SPEC_CONFIG"
