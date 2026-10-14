#!/usr/bin/env bash
# All defaults can be edited or overridden; validate before benchmarking fast mode.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SPEC_MAIN="$(cd -- "$SPEC_MAIN" && pwd)"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-}}"
if [[ -z "$ENV_ROOT" ]]; then echo 'Activate the working A2 Conda environment or set ENV_ROOT.' >&2; exit 2; fi
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${DRAFT:-}"
if [[ -z "$DRAFT" ]]; then echo 'Set DRAFT to a compiled_prefix_v3 pilot checkpoint.' >&2; exit 2; fi
PORT="${PORT:-8213}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((10#$PORT < 1 || 10#$PORT > 65535)); then echo 'PORT must be in [1, 65535].' >&2; exit 2; fi
NPU="${NPU:-7}"
MODE="${MODE:-validate}"
if [[ "$MODE" != validate && "$MODE" != fast && "$MODE" != reference ]]; then echo 'MODE must be validate, fast, or reference.' >&2; exit 2; fi
unset PYTHONPATH ASCEND_LAUNCH_BLOCKING
source "${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}"
source "${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"; fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DFLASH_PREFIX_INFERENCE="$MODE" DFLASH_PREFIX_NPU_GRAPH=1
export DFLASH_PREFIX_TOKEN_CACHE=0 DFLASH_PREFIX_SCORE_CHECK=strict
export DFLASH_PREFIX_VALIDATE_STEPS="${VALIDATE_STEPS:-32}"
export DFLASH_PREFIX_PROFILE_STEPS=0
# Normally zero. The bounded diagnostic launcher sets this and PROFILE_DIR;
# regular serving never arms profiling or adds device synchronization.
export DFLASH_PREFIX_SELECTOR_PROFILE_STEPS="${SELECTOR_PROFILE_STEPS:-0}"
unset DFLASH_PREFIX_GRAPH_SCOPE DFLASH_PREFIX_HISTORY_GROUP_SIZE DFLASH_PREFIX_FUSION
# Target remains eager; the new selector owns its graph. Fixed serving batch=1.
for name in TARGET_MODEL DRAFT; do
    resolved_path="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${!name}")"
    printf -v "$name" '%s' "$resolved_path"
done
ARGS=(--draft "$DRAFT" --target "$TARGET_MODEL")
if [[ -n "${BLOCK_SIZE:-}" ]]; then ARGS+=(--block-size "$BLOCK_SIZE"); fi
SPEC_CONFIG="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" "$SPEC_MAIN/speculators/scripts/dflash_prefix_compiled/check_checkpoint.py" "${ARGS[@]}")"
cd "$SPEC_MAIN/speculators"
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$TARGET_MODEL" \
    --host 127.0.0.1 --port "$PORT" --served-model-name "${SERVED_MODEL_NAME:-qwen3-4b-dflash}" \
    --tensor-parallel-size 1 --data-parallel-size 1 --dtype bfloat16 --seed 42 \
    --generation-config vllm --max-num-seqs 1 --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" --block-size 128 \
    --gpu-memory-utilization "${MEMORY_UTILIZATION:-0.96}" --enforce-eager \
    --no-async-scheduling --no-enable-prefix-caching --no-enable-chunked-prefill \
    --api-server-count 1 --renderer-num-workers 1 \
    --additional-config '{"enable_reduce_sample": false}' --enable-per-request-metrics \
    --per-request-spec-decode-metrics summary --speculative-config "$SPEC_CONFIG" "$@"
