#!/usr/bin/env bash
set -eo pipefail

# Diagnostic run of the SAME checkpoint. Not a speed benchmark.
export SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
export ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
export DRAFT="${DRAFT:-$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs16/tau_v2_scratch_20260924_044011_1685744/checkpoints/0}"
export NPU="${NPU:-11}"
export PORT="${PORT:-8211}"
export MODE=validate
export TOKEN_CACHE="${TOKEN_CACHE:-1}"
export PROFILE_STEPS="${PROFILE_STEPS:-8}"
export VALIDATE_STEPS="${VALIDATE_STEPS:-32}"
export DFLASH_PREFIX_SCORE_CHECK=audit
export DFLASH_PREFIX_AUDIT_MAX_EVENTS="${AUDIT_MAX_EVENTS:-4}"
export DFLASH_PREFIX_AUDIT_DIR="${AUDIT_DIR:-$SPEC_MAIN/output/prefix_numeric_audit_$(date +%Y%m%d_%H%M%S)_$$}"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    echo "Audit directory for a real run: $DFLASH_PREFIX_AUDIT_DIR"
    exec bash "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_cached.sh"
fi

"$ENV_ROOT/bin/python" - "$DFLASH_PREFIX_AUDIT_DIR" "$DRAFT" "$PORT" "$NPU" "$TOKEN_CACHE" <<'PY'
import json, pathlib, sys
directory, draft, port, npu, token_cache = sys.argv[1:]
path = pathlib.Path(directory)
path.mkdir(parents=True, exist_ok=False)
(path / "launch.json").write_text(json.dumps({
    "draft": draft, "port": port, "npu": npu, "token_cache": token_cache,
    "inference": "validate", "score_check": "audit",
    "note": "Diagnostic run, not a throughput benchmark. Token disagreement or non-finite scores remains fatal."
}, indent=2) + "\n")
print(f"Numeric audit output: {path.resolve()}")
PY

# Keep the complete server traceback even if a worker exits.
set +e
bash "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_cached.sh" 2>&1 \
    | tee "$DFLASH_PREFIX_AUDIT_DIR/server.log"
server_status=${PIPESTATUS[0]}
exit "$server_status"
