#!/usr/bin/env bash
set -eo pipefail

SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
LABEL="${LABEL:-fast}"
PORT="${PORT:-8211}"
MANIFEST="${MANIFEST:-$SPEC_MAIN/Evaluator/retrace_eval_prompts.json}"
RESULT_DIR="${RESULT_DIR:-$SPEC_MAIN/Evaluator/results/prefix_inference_${LABEL}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Set MANIFEST to your existing GSM8K prompt manifest: $MANIFEST was not found." >&2
    exit 1
fi
if [[ -e "$RESULT_DIR" ]]; then
    echo "Choose a fresh RESULT_DIR: $RESULT_DIR" >&2
    exit 1
fi
cd "$SPEC_MAIN/Evaluator"
exec "$ENV_ROOT/bin/python" evaluator.py \
    --base-url "http://127.0.0.1:$PORT" \
    --model qwen3-4b-dflash \
    --method "dflash_prefix_same_checkpoint_${LABEL}" \
    --datasets gsm8k \
    --manifest "$MANIFEST" \
    --num-prompts 128 \
    --temperature 0 \
    --top-p 1.0 \
    --top-k -1 \
    --max-new-tokens 2048 \
    --concurrency 1 \
    --seed 42 \
    --prompt-seed 42 \
    --warmup-requests 1 \
    --no-thinking \
    --output "$RESULT_DIR"
