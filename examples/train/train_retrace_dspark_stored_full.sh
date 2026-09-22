#!/usr/bin/env bash
# Full stored-response DSpark + ReTrace experiment, run the DSpark-specific NPU smoke first.
# This retains ReTrace conditioning but is NOT the paper's prompt-only/FSDP recipe.
# Start from the original local DSpark-block7; train 40K selected records for 8 epochs.
# Run: bash examples/train/train_retrace_dspark_stored_full.sh
# Print only: bash examples/train/train_retrace_dspark_stored_full.sh --print-command
# Resume an interrupted run using that run's generated resume_stored.sh instead.
set -euo pipefail

export RETRACE_ROOT="${RETRACE_ROOT:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
export RETRACE_PYTHON="${RETRACE_PYTHON:-/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python}"
export RETRACE_TARGET=/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export RETRACE_DSPARK_BASE=/home/n84449292/m84379596/Huggingface/Qwen3-4B-DSpark-block7
export RETRACE_ARROW=/home/n84449292/m84379596/Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072
export RETRACE_CANN_ROOT=/home/n84449292/m84379596/CANN/9.1.0

# A separate variable prevents an inherited smoke output path from being reused.
export RETRACE_DSPARK_STORED_OUTPUT="${RETRACE_DSPARK_FULL_OUTPUT:-$RETRACE_ROOT/output/retrace_dspark_stored_full_$(date +%Y%m%d_%H%M%S)}"
export RETRACE_DSPARK_STORED_TRACE_STEPS="${RETRACE_DSPARK_STORED_TRACE_STEPS:-0}"

trainer="$RETRACE_ROOT/speculators/examples/train/train_retrace_dspark_stored_vllm.sh"
command=(
  bash "$trainer"
  --target "$RETRACE_TARGET"
  --base-dspark "$RETRACE_DSPARK_BASE"
  --data "$RETRACE_ARROW"
  --output "$RETRACE_DSPARK_STORED_OUTPUT"
  --server-npus 8,9
  --trainer-npus 10,11,12,13,14,15
  --max-records 40000
  --epochs 8
  --lr 5e-5
  --seed 42
  --prompt-length 512
  --response-length 1024
  --tokens-per-worker 3072
  --pairs-per-batch 16
  --blocks-per-forward 4
  --request-concurrency 8
  --data-workers 2
  --port 8523
)

if (( $# > 1 )); then
  printf '%s\n' 'Usage: bash train_retrace_dspark_stored_full.sh [--print-command|--dry-run]' >&2
  exit 2
fi
case "${1:-}" in
  --print-command)
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
    ;;
  --dry-run)
    command+=(--dry-run)
    ;;
  '') ;;
  *)
    printf '%s\n' 'This launcher fixes 40K records and 8 epochs. For resume, use RUN/resume_stored.sh.' >&2
    exit 2
    ;;
esac

if [[ ! -f "$trainer" ]]; then
  printf 'Missing installed trainer: %s\n' "$trainer" >&2
  exit 1
fi
if [[ -e "$RETRACE_DSPARK_STORED_OUTPUT" ]]; then
  printf 'Output already exists: %s\nUse its resume_stored.sh, or set a new RETRACE_DSPARK_FULL_OUTPUT.\n' "$RETRACE_DSPARK_STORED_OUTPUT" >&2
  exit 1
fi
printf 'Pretrained DSpark: %s\nFull run: %s\n' "$RETRACE_DSPARK_BASE" "$RETRACE_DSPARK_STORED_OUTPUT"
if [[ "${1:-}" == --dry-run ]]; then
  exec "${command[@]}"
fi

# The installed launcher sources both CANN 9.1.0 scripts, preserves LD_PRELOAD,
# checks package origins, imports pretrained tensors strictly and starts vLLM.
# Keep the console log next to the new output directory: the supervisor must
# create the directory itself. Do not create the run directory before launch.
mkdir -p "$(dirname "$RETRACE_DSPARK_STORED_OUTPUT")"
"${command[@]}" 2>&1 | tee -a "${RETRACE_DSPARK_STORED_OUTPUT}.log"
printf 'Training finished. Final epoch checkpoint: %s/checkpoints/7\n' "$RETRACE_DSPARK_STORED_OUTPUT"
