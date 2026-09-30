#!/usr/bin/env bash
set -eo pipefail

# Existing checkpoint, with a token-projection lookup table built at warmup.
export SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
export ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
export TARGET="${TARGET:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DRAFT="${DRAFT:-$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs16/tau_v2_scratch_20260924_044011_1685744/checkpoints/0}"
export NPU="${NPU:-11}"
export PORT="${PORT:-8211}"
export MODE="${MODE:-fast}"
export VALIDATE_STEPS="${VALIDATE_STEPS:-32}"

TOKEN_CACHE="${TOKEN_CACHE:-1}"             # 0 = the previous measured fast path
PROFILE_STEPS="${PROFILE_STEPS:-8}"         # warmup only; 0 disables profiling
CACHE_CHUNK_TOKENS="${CACHE_CHUNK_TOKENS:-1024}"
CACHE_MAX_MIB="${CACHE_MAX_MIB:-768}"       # table budget; transient buffers are extra

export DFLASH_PREFIX_TOKEN_CACHE="$TOKEN_CACHE"
export DFLASH_PREFIX_PROFILE_STEPS="$PROFILE_STEPS"
export DFLASH_PREFIX_CACHE_CHUNK_TOKENS="$CACHE_CHUNK_TOKENS"
export DFLASH_PREFIX_CACHE_MAX_MIB="$CACHE_MAX_MIB"

# Read metadata only; no torch import or NPU allocation in CHECK_ONLY mode.
"$ENV_ROOT/bin/python" - "$TARGET" "$DRAFT" "$TOKEN_CACHE" "$PROFILE_STEPS" "$VALIDATE_STEPS" "$CACHE_CHUNK_TOKENS" "$CACHE_MAX_MIB" <<'PY'
import json, math, pathlib, sys
target, draft, enabled, profile, checks, chunk, budget = sys.argv[1:]
if enabled not in {"0", "1"}:
    raise SystemExit("TOKEN_CACHE must be 0 or 1")
profile, checks, chunk = int(profile), int(checks), int(chunk)
budget = float(budget)
if checks < 1 or not 0 <= profile <= checks or chunk < 1 or not math.isfinite(budget) or budget <= 0:
    raise SystemExit("Require VALIDATE_STEPS>0, 0<=PROFILE_STEPS<=VALIDATE_STEPS, and positive cache chunk/budget")
cfg = json.loads((pathlib.Path(draft)/"config.json").read_text())
target_cfg = json.loads((pathlib.Path(target)/"config.json").read_text())
rank, layers, vocab = cfg["prefix_rank"], cfg["prefix_attention_layers"], target_cfg["vocab_size"]
mib = vocab * rank * (4 + 2*layers) * 2 / 1024**2
print(f"Token cache={enabled}; estimated BF16 table={mib:.2f} MiB; table budget={budget:g} MiB; warmup profiling blocks={profile}")
if enabled == "1" and mib > budget:
    raise SystemExit("Cache exceeds budget. Set TOKEN_CACHE=0 for the existing fast path, or choose an explicit larger CACHE_MAX_MIB.")
PY

# Reuse the same target/draft validation and all measured serving settings.
exec bash "$SPEC_MAIN/speculators/examples/serve_dflash_prefix_fast.sh"
