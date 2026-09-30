#!/usr/bin/env bash
set -eo pipefail

# Existing block-16 checkpoint. Validation first; MODE=fast for later timing.
export SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
export ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
export DRAFT="${DRAFT:-$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs16/tau_v2_scratch_20260924_044011_1685744/checkpoints/0}"
export NPU="${NPU:-11}"
export PORT="${PORT:-8211}"
export MODE="${MODE:-validate}"
export VALIDATE_STEPS="${VALIDATE_STEPS:-32}"

# The audit isolated the token-choice mismatch to the vocabulary projection cache.
# Keep per-block preparation and the first fast walk; graph that walk optionally.
export TOKEN_CACHE=0
export DFLASH_PREFIX_TOKEN_CACHE=0
export PROFILE_STEPS=0
export DFLASH_PREFIX_PROFILE_STEPS=0
export DFLASH_PREFIX_SCORE_CHECK=strict
export DFLASH_PREFIX_NPU_GRAPH="${DFLASH_PREFIX_NPU_GRAPH:-1}"

case "$MODE" in
    validate|fast) ;;
    *) echo "MODE must be validate or fast for this launcher" >&2; exit 1 ;;
esac
case "$DFLASH_PREFIX_NPU_GRAPH" in
    0|1) ;;
    *) echo "DFLASH_PREFIX_NPU_GRAPH must be 0 or 1" >&2; exit 1 ;;
esac

echo "Prefix walk: NPU_GRAPH=$DFLASH_PREFIX_NPU_GRAPH; TOKEN_CACHE=0; strict parity; MODE=$MODE"
echo "Draft: $DRAFT"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    exec bash "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_fast.sh"
fi

RUN_LOG="${RUN_LOG:-$SPEC_MAIN/output/prefix_graph_${MODE}_$(date +%Y%m%d_%H%M%S)_$$.log}"
mkdir -p "$(dirname "$RUN_LOG")"
if [[ -e "$RUN_LOG" ]]; then
    echo "Choose a fresh RUN_LOG: $RUN_LOG" >&2
    exit 1
fi
echo "Server log: $RUN_LOG"
bash "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_fast.sh" 2>&1 | tee "$RUN_LOG"
