#!/usr/bin/env bash
set -eo pipefail

SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
PORT="${PORT:-8209}"
FUSION="${FUSION:-local}"
NPU="${NPU:-9}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-$SPEC_MAIN/output/prefix_fusion_diagnostics}"

# This launcher is deliberately a strict diagnostic run, never a speed test.
# Keep the variant that failed so the same arithmetic is investigated.
export MODE=validate
export PORT FUSION NPU
export DFLASH_PREFIX_FAILURE_DIR="$DIAGNOSTIC_DIR"
mkdir -p "$DIAGNOSTIC_DIR"
printf 'Failure diagnostics: %s\nStrict checks retained; no fast-path fallback.\n' "$DIAGNOSTIC_DIR"
exec bash "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_fused.sh" "$@"
