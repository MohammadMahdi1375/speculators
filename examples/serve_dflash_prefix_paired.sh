#!/usr/bin/env bash
set -eo pipefail

# Existing checkpoint and serving flags come from the working combined launcher.
# Override its DRAFT_MODEL, NUM_SPECULATIVE_TOKENS, NPU, PORT, etc. as before.
export SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
export MODE="${MODE:-validate}"
export GRAPH_SCOPE=selector
export HISTORY_GROUP_SIZE="${HISTORY_GROUP_SIZE:-2}"  # 1: previous combined graph
export DFLASH_PREFIX_HISTORY_GROUP_SIZE="$HISTORY_GROUP_SIZE"
export PROFILE_STEPS="${PROFILE_STEPS:-0}"
export VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

case "$HISTORY_GROUP_SIZE" in
    1|2) ;;
    *) echo 'HISTORY_GROUP_SIZE must be 1 or 2' >&2; exit 1 ;;
esac
printf 'Prefix history group size: %s; mode: %s; profile samples: %s\n' \
    "$HISTORY_GROUP_SIZE" "$MODE" "$PROFILE_STEPS"
exec bash "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_combined.sh" "$@"
