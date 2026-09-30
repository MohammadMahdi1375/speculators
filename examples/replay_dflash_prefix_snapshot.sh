#!/usr/bin/env bash
set -eo pipefail

SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
NPU="${NPU:-9}"
SNAPSHOT="${SNAPSHOT:-$SPEC_MAIN/output/prefix_fusion_diagnostics/prefix_failure_1790651349132896730_37266.pt}"
CANDIDATE="${CANDIDATE:-local_unpaired}"

[[ -f "$SNAPSHOT" ]] || { echo "Snapshot not found: $SNAPSHOT" >&2; exit 1; }
unset PYTHONPATH
unset ASCEND_LAUNCH_BLOCKING
unset TRITON_INTERPRET
source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=false
printf 'Replay snapshot: %s\nNPU: %s; candidate: %s\nNo vLLM server or target model is loaded.\n' "$SNAPSHOT" "$NPU" "$CANDIDATE"
exec "$PYTHON_BIN" "$SPEC_MAIN/speculators/scripts/dflash_prefix_tau/replay_unpaired_snapshot.py" \
    --spec-main "$SPEC_MAIN" --snapshot "$SNAPSHOT" --candidate "$CANDIDATE" "$@"
