#!/usr/bin/env bash
set -eo pipefail

# Calls your current evaluator without changing timing or acceptance formulas.
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
PORT="${PORT:-8209}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-4b-dflash}"
LABEL="${LABEL:-fused_validate_bs16}"
METHOD="${METHOD:-dflash_prefix_${LABEL}}"
DATASETS="${DATASETS:-all}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MANIFEST="${MANIFEST:-$SPEC_MAIN/Evaluator/retrace_eval_prompts_all_v2.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$SPEC_MAIN/Evaluator/results/prefix_${LABEL}_$(date +%Y%m%d_%H%M%S)_$$}"
SELECTOR_WARMUP_STEPS="${SELECTOR_WARMUP_STEPS:-32}"
SELECTOR_WARMUP_TOKENS="${SELECTOR_WARMUP_TOKENS:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
CONCURRENCY="${CONCURRENCY:-1}"
SEED="${SEED:-42}"

export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Prompt manifest not found: $MANIFEST. Set MANIFEST to the one used for your paired/vanilla results." >&2
    exit 1
fi
read -r -a dataset_args <<< "$DATASETS"
cd "$SPEC_MAIN/Evaluator"
exec "$PYTHON_BIN" evaluator.py \
    --base-url "$BASE_URL" --model "$SERVED_MODEL" --method "$METHOD" \
    --datasets "${dataset_args[@]}" --manifest "$MANIFEST" \
    --num-prompts "$NUM_PROMPTS" --temperature "$TEMPERATURE" \
    --top-p 1.0 --top-k -1 --max-new-tokens "$MAX_NEW_TOKENS" \
    --concurrency "$CONCURRENCY" --seed "$SEED" --prompt-seed "$SEED" \
    --warmup-requests 1 --selector-warmup-tokens "$SELECTOR_WARMUP_TOKENS" \
    --selector-warmup-steps "$SELECTOR_WARMUP_STEPS" --timeout-s 1800 \
    --no-thinking --output "$EVAL_OUTPUT" "$@"
