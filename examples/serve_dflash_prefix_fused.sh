#!/usr/bin/env bash
set -eo pipefail

# Edit settings here, or pass their names before bash. No checkpoint is selected
# automatically. Default: the same block-16 checkpoint as your ~100 tokens/s run.
ENV_ROOT="${ENV_ROOT:-/home/n84449292/m84379596/conda/vllm-dflash2-main}"
SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"

TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT_MODEL="${DRAFT_MODEL:-$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs16/tau_v2_scratch_20260924_044011_1685744/checkpoints/0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"

# To use block 8, change BOTH DRAFT_MODEL and NUM_SPECULATIVE_TOKENS:
# DRAFT_MODEL="$SPEC_MAIN/output/dflash_prefix_qwen3_4b_openperfectblend_bs8/tau_v2_scratch_20260928_082808_3142334/checkpoints/0"
# NUM_SPECULATIVE_TOKENS=7

NPU="${NPU:-9}"
PORT="${PORT:-8209}"
MODE="${MODE:-validate}"                 # validate every block; fast checks first 32
GRAPH_SCOPE=selector
export DFLASH_PREFIX_HISTORY_GROUP_SIZE=2
PROFILE_STEPS=0  # legacy combined profiler; fusion has its own warmup timing
FUSION="${FUSION:-auto}"                   # auto | off | select | local | select_softmax | local_softmax
FUSION_TUNE_STEPS="${FUSION_TUNE_STEPS:-32}"
FUSION_PROFILE_STEPS="${FUSION_PROFILE_STEPS:-8}"
FUSION_CANDIDATES="${FUSION_CANDIDATES:-select,local,select_softmax,local_softmax}"
FUSION_MIN_GAIN="${FUSION_MIN_GAIN:-0.02}"
FUSION_REPORT="${FUSION_REPORT:-$SPEC_MAIN/output/prefix_fusion_${PORT}}"
export DFLASH_PREFIX_FUSION="$FUSION"
export DFLASH_PREFIX_FUSION_DRAFT="$DRAFT_MODEL"
export DFLASH_PREFIX_FUSION_TUNE_STEPS="$FUSION_TUNE_STEPS"
export DFLASH_PREFIX_FUSION_PROFILE_STEPS="$FUSION_PROFILE_STEPS"
export DFLASH_PREFIX_FUSION_CANDIDATES="$FUSION_CANDIDATES"
export DFLASH_PREFIX_FUSION_MIN_GAIN="$FUSION_MIN_GAIN"
export DFLASH_PREFIX_FUSION_REPORT="$FUSION_REPORT"
export VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
case "$MODE" in validate|fast) ;; *) echo 'MODE must be validate or fast' >&2; exit 1;; esac
case "$FUSION" in auto|off|select|local|select_softmax|local_softmax) ;; *) echo 'Invalid FUSION variant' >&2; exit 1;; esac
if [[ "$MODE" == fast && "$FUSION" == auto ]]; then
    echo 'For fast serving set FUSION to the selected variant from PREFIX_FUSION_RESULT (paired means off). This pins the variant you evaluated in validate mode.' >&2
    exit 1
fi
VALIDATE_STEPS="${VALIDATE_STEPS:-32}"

export DFLASH_PREFIX_INFERENCE="$MODE"
export DFLASH_PREFIX_NPU_GRAPH=1
export DFLASH_PREFIX_GRAPH_SCOPE="$GRAPH_SCOPE"
export DFLASH_PREFIX_TOKEN_CACHE=0
export DFLASH_PREFIX_SCORE_CHECK=strict
export DFLASH_PREFIX_PROFILE_STEPS=0
export DFLASH_PREFIX_SELECTOR_PROFILE_STEPS="$PROFILE_STEPS"
export DFLASH_PREFIX_VALIDATE_STEPS="$VALIDATE_STEPS"

# This check uses only Python's standard library. All vLLM options stay editable.
SPECULATIVE_CONFIG="$(TARGET_MODEL="$TARGET_MODEL" DRAFT_MODEL="$DRAFT_MODEL" \
    NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
target = Path(os.environ['TARGET_MODEL']).resolve(strict=True)
draft = Path(os.environ['DRAFT_MODEL']).resolve(strict=True)
target_cfg = json.loads((target / 'config.json').read_text())
cfg = json.loads((draft / 'config.json').read_text())
proposals = int(os.environ['NUM_SPECULATIVE_TOKENS'])
if target == draft or target_cfg.get('model_type') != 'qwen3' or target_cfg.get('speculators_model_type'):
    raise SystemExit('TARGET_MODEL must be the original Qwen3-4B target, not a drafter.')
if cfg.get('speculators_model_type') != 'dflash_prefix' or cfg.get('prefix_selector_kind') != 'local_prefix_v2':
    raise SystemExit('DRAFT_MODEL must be your trained local_prefix_v2 prefix checkpoint.')
if cfg.get('block_size') not in (8, 16) or proposals != cfg['block_size'] - 1:
    raise SystemExit('NUM_SPECULATIVE_TOKENS must equal the trained block_size minus one.')
if cfg.get('prefix_disable_selector', False) or cfg.get('prefix_inference_gate_scale', 1) != 1 or cfg.get('prefix_history_scale', 1) != 1:
    raise SystemExit('Use the unmodified checkpoint: selector enabled, both inference scales one.')
if cfg.get('prefix_walk_backend', 'torch') != 'torch':
    raise SystemExit('local_prefix_v2 needs prefix_walk_backend=torch.')
if os.environ['DFLASH_PREFIX_FUSION'] != 'off':
    if cfg.get('prefix_rank') not in (64, 256) or cfg.get('prefix_attention_heads') != 4 or cfg.get('prefix_attention_layers') != 2 or cfg.get('prefix_top_k') != 16:
        raise SystemExit('Fusion v1 supports K16, rank64/256, heads4, layers2; use FUSION=off otherwise.')
print(json.dumps({
    'method': 'dflash',
    'model': str(draft),
    'num_speculative_tokens': proposals,
    'draft_sample_method': 'greedy',
    'disable_padded_drafter_batch': False,
    'enable_adaptive_verification': False,
    'enforce_eager': True,
}))
PY
)"

printf 'Fusion variant: %s; tuning samples: %s; timing samples: %s\n' \
    "$FUSION" "$FUSION_TUNE_STEPS" "$FUSION_PROFILE_STEPS"
printf 'Target: %s\nDraft: %s\nProposals: %s; mode=%s; graph_scope=%s; NPU=%s; port=%s\n' \
    "$TARGET_MODEL" "$DRAFT_MODEL" "$NUM_SPECULATIVE_TOKENS" "$MODE" "$GRAPH_SCOPE" "$NPU" "$PORT"
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
    printf '%s\n' "$SPECULATIVE_CONFIG" 'Configuration printed; no NPU process started.'
    exit 0
fi

unset PYTHONPATH
unset ASCEND_LAUNCH_BLOCKING
source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
export PYTHONPATH="$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"
fi

export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

cd "$SPEC_MAIN/speculators"

# The target/draft backbones stay eager. This update captures only the selector.
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main \
    serve "$TARGET_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name qwen3-4b-dflash \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --seed 42 \
    --generation-config vllm \
    --max-num-seqs 1 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --gpu-memory-utilization 0.96 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --api-server-count 1 \
    --renderer-num-workers 1 \
    --additional-config '{"enable_reduce_sample": false}' \
    --enable-per-request-metrics \
    --per-request-spec-decode-metrics summary \
    --speculative-config "$SPECULATIVE_CONFIG" \
    "$@"
