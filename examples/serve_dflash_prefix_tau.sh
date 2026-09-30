#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SPEC_MAIN="$(cd -- "$SPEC_MAIN" && pwd)"
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
DRAFT="${DRAFT:-$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs${BLOCK_SIZE}/tau_v2_scratch_latest/checkpoints/checkpoint_best}"
MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
PORT="${PORT:-8103}"
NPU="${NPU:-9}"

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
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"

SPECULATIVE_CONFIG="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" \
    "$SPEC_MAIN/speculators/scripts/dflash_prefix_tau/checkpoint_config.py" \
    --draft "$DRAFT" --expected-block-size "$BLOCK_SIZE")"

# The --block-size 128 flag below is vLLM KV-cache allocation, not draft length.
cd "$SPEC_MAIN/speculators"
exec "$PYTHON_BIN" \
    -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --served-model-name qwen3-4b-dflash \
    --tensor-parallel-size 1 --data-parallel-size 1 \
    --dtype bfloat16 --seed 42 --generation-config vllm \
    --max-num-seqs 1 --max-model-len 32768 --max-num-batched-tokens 32768 \
    --block-size 128 \
    --gpu-memory-utilization 0.96 --enforce-eager --no-async-scheduling \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --api-server-count 1 --renderer-num-workers 1 \
    --additional-config '{"enable_reduce_sample": false}' \
    --enable-per-request-metrics --per-request-spec-decode-metrics summary \
    --speculative-config "$SPECULATIVE_CONFIG"
