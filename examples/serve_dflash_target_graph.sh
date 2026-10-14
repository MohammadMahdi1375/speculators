#!/usr/bin/env bash
# Native target decode graph experiment; edit defaults or override with env.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-}}"
if [[ -z "$ENV_ROOT" ]]; then
    echo 'Activate the working Ascend environment or set ENV_ROOT.' >&2
    exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${DRAFT:?Set DRAFT to your existing trained checkpoint}"
NPU="${NPU:-6}"
PORT="${PORT:-8213}"
TARGET_EXECUTION="${TARGET_EXECUTION:-graph}" # graph | eager
MODE="${MODE:-fast}"                         # selector: fast | validate | reference
RUN_DIR="${RUN_DIR:-$SPEC_MAIN/output/target_graph_${TARGET_EXECUTION}_$(date +%Y%m%d_%H%M%S)_$$}"

unset PYTHONPATH ASCEND_LAUNCH_BLOCKING
source "${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}"
source "${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Batch=1, TP=1, DP=1. Proposal count is read from checkpoint block_size.
# Draft backbone stays eager; each prefix selector keeps its own graph.
exec "$PYTHON_BIN" "$SPEC_MAIN/speculators/scripts/dflash_prefix_compiled/target_graph_server.py" \
    --spec-main "$SPEC_MAIN" --target "$TARGET_MODEL" --draft "$DRAFT" \
    --npu "$NPU" --port "$PORT" --target-execution "$TARGET_EXECUTION" --mode "$MODE" \
    --run-dir "$RUN_DIR" --served-model-name "${SERVED_MODEL_NAME:-qwen3-4b-dflash}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" --max-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" \
    --memory-utilization "${MEMORY_UTILIZATION:-0.96}" \
    --graph-metrics "${GRAPH_METRICS:-0}" --check-only "${CHECK_ONLY:-0}"
