#!/usr/bin/env bash
# Full-epoch compiled-prefix scratch training. Edit defaults or override via env.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"

# BEGIN JOINT_RESUME_ARGS_V3
# EPOCHS is the total epoch target for both fresh and resumed training.
# Example: EPOCHS=8 bash train_dflash_prefix_joint.sh --resume-training /run/checkpoints/4
_JOINT_RESUME_CHECKPOINT=""
while (($#)); do
    case "$1" in
        --resume-training|--resume-traning)
            if [[ -n "$_JOINT_RESUME_CHECKPOINT" || $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                echo 'Use --resume-training CHECKPOINT exactly once.' >&2
                exit 2
            fi
            _JOINT_RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --resume-training=*|--resume-traning=*)
            if [[ -n "$_JOINT_RESUME_CHECKPOINT" || -z "${1#*=}" ]]; then
                echo 'Use --resume-training CHECKPOINT exactly once.' >&2
                exit 2
            fi
            _JOINT_RESUME_CHECKPOINT="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: EPOCHS=8 bash ${BASH_SOURCE[0]} [--resume-training CHECKPOINT]"
            echo 'EPOCHS is the total epoch target. Checkpoint 4 completed epoch 5; EPOCHS=8 runs epochs 6-8.'
            echo 'Without --resume-training, the existing scratch-training settings apply.'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1. Use --help." >&2
            exit 2
            ;;
    esac
done
# END JOINT_RESUME_ARGS_V3

# ============================================================
# Paths (no trained draft/teacher checkpoint is required)
# ============================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
SPEC_MAIN="$(cd -- "$SPEC_MAIN" && pwd)"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-}}"
if [[ -z "$ENV_ROOT" ]]; then echo 'Activate your A2 Conda environment or set ENV_ROOT.' >&2; exit 2; fi
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then echo "Python not found: $PYTHON_BIN" >&2; exit 2; fi
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
HELPER_DIR="$SPEC_MAIN/speculators/scripts/dflash_prefix_joint"
export MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATASET="${DATASET:-/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
# Read architecture/optimizer/data-split settings only. No weights are loaded.
export BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/run.yaml}"
if [[ -z "$_JOINT_RESUME_CHECKPOINT" && ! -r "$BASE_TRAIN_CONFIG" ]]; then
    echo "Missing recipe: $BASE_TRAIN_CONFIG. Set BASE_TRAIN_CONFIG to your vanilla scratch run.yaml." >&2
    exit 2
fi

# ============================================================
# Model, training budget and learning rates
# ============================================================
export BLOCK_SIZE="${BLOCK_SIZE:-16}"       # anchor + 15 proposals; also supports 8
export PREFIX_TOP_K="${PREFIX_TOP_K:-16}"
export PREFIX_RANK="${PREFIX_RANK:-256}"
export PREFIX_HEADS="${PREFIX_HEADS:-4}"
export PREFIX_LAYERS="${PREFIX_LAYERS:-1}"   # compiled_prefix_v3 has one compiled layer
export COMPILED_LOGIT_CAP="${COMPILED_LOGIT_CAP:-8}"
export EPOCHS="${EPOCHS:-5}"                 # real complete training-split passes
export SEED="${SEED:-42}"
export HEAD_LR="${HEAD_LR:-0.0003}"
export HEAD_WEIGHT_DECAY="${HEAD_WEIGHT_DECAY:-0.01}"
# Optional independent overrides; empty means preserve the vanilla recipe.
export BACKBONE_LR="${BACKBONE_LR:-}" MUON_LR="${MUON_LR:-}"
export BACKBONE_WARMUP_RATIO="${BACKBONE_WARMUP_RATIO:-0.02}"
export SELECTOR_DELAY_STEPS="${SELECTOR_DELAY_STEPS:-2000}"
export SELECTOR_WARMUP_STEPS="${SELECTOR_WARMUP_STEPS:-2000}"
export PREFIX_LOSS_ALPHA="${PREFIX_LOSS_ALPHA:-0.25}"
export PREFIX_FOCUS_START_STEPS="${PREFIX_FOCUS_START_STEPS:-4000}"
export PREFIX_FOCUS_WARMUP_STEPS="${PREFIX_FOCUS_WARMUP_STEPS:-4000}"
export PREFIX_DENSE_FLOOR="${PREFIX_DENSE_FLOOR:-0.25}"
export MAX_ANCHORS="${MAX_ANCHORS:-512}"
export ANCHOR_CHUNK_SIZE="${ANCHOR_CHUNK_SIZE:-8}"
export VAL_BATCHES="${VAL_BATCHES:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LOG_FREQ="${LOG_FREQ:-20}"
export DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-sdpa}"

# Execution controls: 0 disables recomputation and uses more activation memory.
# Original behavior remains the default; start a memory-headroom trial with both 0.
export BACKBONE_CHECKPOINTING="${BACKBONE_CHECKPOINTING:-1}"
export SELECTOR_CHECKPOINTING="${SELECTOR_CHECKPOINTING:-1}"
export MEMORY_LOG_EVERY="${MEMORY_LOG_EVERY:-100}" # per-rank peak GiB; 0 disables
# Empty preserves the source recipe. This is a packed token batch budget,
# not the speculative BLOCK_SIZE and not a conventional sample-count batch size.
export TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-}"

# ============================================================
# Optional training-only constraint; GUARD_MODE=off is the ablation
# Applies to selector updates on replay features, not backbone updates.
# ============================================================
export GUARD_MODE="${GUARD_MODE:-project}"   # project | off
export GUARD_START_STEPS="${GUARD_START_STEPS:-4000}"
export GUARD_EVERY="${GUARD_EVERY:-1}"
export GUARD_REPLAY_ANCHORS="${GUARD_REPLAY_ANCHORS:-32}"  # per trainer rank
export GUARD_ADD_ANCHORS="${GUARD_ADD_ANCHORS:-4}"
export GUARD_CHUNK_SIZE="${GUARD_CHUNK_SIZE:-4}"
export GUARD_BACKTRACKS="${GUARD_BACKTRACKS:-3}"
export CHECK_ONLY="${CHECK_ONLY:-0}"
export PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" # config + single/distributed NPU checks; no full training
if [[ "$CHECK_ONLY" != 0 && "$CHECK_ONLY" != 1 ]]; then echo 'CHECK_ONLY must be 0 or 1.' >&2; exit 2; fi
if [[ "$PREFLIGHT_ONLY" != 0 && "$PREFLIGHT_ONLY" != 1 ]]; then echo 'PREFLIGHT_ONLY must be 0 or 1.' >&2; exit 2; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs${BLOCK_SIZE}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/joint_${GUARD_MODE}_$(date +%Y%m%d_%H%M%S)_$$}"

# ============================================================
# A2 NPU allocation (NPU 7 is left free by default)
# ============================================================
export VLLM_NPUS="${VLLM_NPUS:-0,1}" VLLM_DP="${VLLM_DP:-2}"
export TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}" NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-6}"
export VLLM_PORT="${VLLM_PORT:-8092}" VLLM_RPC_PORT="${VLLM_RPC_PORT:-29692}"
VLLM_MEMORY_UTILIZATION="${VLLM_MEMORY_UTILIZATION:-0.85}"

# ============================================================
# Environment
# ============================================================
# BEGIN JOINT_RESUME_DISPATCH_V3
# Read EPOCHS and the normal path/NPU/port/output settings above before dispatch.
if [[ -n "$_JOINT_RESUME_CHECKPOINT" ]]; then
    export ENV_ROOT PYTHON_BIN CANN_ROOT VLLM_MEMORY_UTILIZATION
    RESUME_CHECKPOINT="$_JOINT_RESUME_CHECKPOINT" \
        exec bash "$SCRIPT_DIR/continue_dflash_prefix_joint.sh"
fi
# END JOINT_RESUME_DISPATCH_V3

unset PYTHONPATH ASCEND_LAUNCH_BLOCKING
for env_file in "${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}" "${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"; do
    if [[ ! -r "$env_file" ]]; then echo "Cannot read Ascend environment: $env_file" >&2; exit 2; fi
    source "$env_file"
done
# SOC_VERSION is inherited from the A2 installation; no old A3 chip is forced.
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"; fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY DFLASH_TP_GATHER
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 OMP_PROC_BIND=false
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
set -u
echo "Using ENV_ROOT=$ENV_ROOT"
echo "Using PYTHON_BIN=$PYTHON_BIN"
for name in MODEL DATASET BASE_TRAIN_CONFIG OUTPUT_DIR; do
    resolved_path="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${!name}")"
    printf -v "$name" '%s' "$resolved_path"
    export "$name"
done

# ============================================================
# Configuration, data checks and provenance
# ============================================================
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" "$HELPER_DIR/prepare_experiment.py"
read -r -a TARGET_LAYERS < "$OUTPUT_DIR/target_layers.txt"
MASK_TOKEN_ID="$(cat "$OUTPUT_DIR/mask_token_id.txt")"
SERVER_MAX_MODEL_LEN="$(cat "$OUTPUT_DIR/server_max_model_len.txt")"
cd "$SPEC_MAIN/speculators"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" scripts/preflight_dflash_prefix.py \
    --model "$MODEL" --dataset "$DATASET" --output "$OUTPUT_DIR" \
    --target-layers "${TARGET_LAYERS[@]}" --mask-token-id "$MASK_TOKEN_ID"
mkdir -p "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/hidden_states"
cp -a -- "$HELPER_DIR" "$OUTPUT_DIR/provenance/training_helpers"
cp -- "$SCRIPT_DIR/train_dflash_prefix_joint.sh" "$OUTPUT_DIR/provenance/"
cp -a -- src/speculators/models/dflash_prefix "$OUTPUT_DIR/provenance/model_code"
for repo in speculators vllm vllm-ascend; do
    git -C "$SPEC_MAIN/$repo" rev-parse HEAD > "$OUTPUT_DIR/provenance/${repo}_commit.txt"
    git -C "$SPEC_MAIN/$repo" diff > "$OUTPUT_DIR/provenance/${repo}.patch"
done
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Configuration/data checks passed: $OUTPUT_DIR"
    echo 'No NPU smoke test, hidden-state server or training process was started.'
    echo 'Use a fresh OUTPUT_DIR for the actual run.'
    exit 0
fi

if [[ "$PREFLIGHT_ONLY" == 0 ]]; then
"$PYTHON_BIN" - <<'PY_PORTS'
import os, socket
for key in ('VLLM_PORT', 'VLLM_RPC_PORT'):
    with socket.socket() as sock:
        try:
            sock.bind(('0.0.0.0', int(os.environ[key])))
        except OSError as exc:
            raise SystemExit(f'{key} is unavailable: {exc}')
PY_PORTS
fi
# Exercise full-model backward, the guard and export before starting the target.
ASCEND_RT_VISIBLE_DEVICES="${TRAIN_NPUS%%,*}" "$PYTHON_BIN" "$HELPER_DIR/check_training.py" \
    --device npu:0 --settings "$OUTPUT_DIR/joint_settings.json" \
    2>&1 | tee "$OUTPUT_DIR/logs/joint_smoke.log"
# Check actual HCCL/DDP + explicit guard reductions on the selected ranks.
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    "$HELPER_DIR/check_distributed.py" --settings "$OUTPUT_DIR/joint_settings.json" \
    --output "$OUTPUT_DIR/provenance/distributed_smoke" \
    2>&1 | tee "$OUTPUT_DIR/logs/joint_distributed_smoke.log"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    echo "JOINT_PREFLIGHT_PASS: $OUTPUT_DIR"
    echo 'Single-NPU and distributed backward/export checks passed. No target server or full training was started.'
    echo 'Use PREFLIGHT_ONLY=0 and a fresh OUTPUT_DIR for the actual run.'
    exit 0
fi

# ============================================================
# Target hidden-state server and distributed training
# ============================================================
setsid env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON_BIN" scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids "${TARGET_LAYERS[@]}" --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --provenance-dir "$OUTPUT_DIR/provenance" -- \
    --served-model-name "$MODEL" --data-parallel-size "$VLLM_DP" --tensor-parallel-size 1 \
    --data-parallel-rpc-port "$VLLM_RPC_PORT" --port "$VLLM_PORT" \
    --max-model-len "$SERVER_MAX_MODEL_LEN" --gpu-memory-utilization "$VLLM_MEMORY_UTILIZATION" \
    --api-server-count 1 --renderer-num-workers 1 \
    > "$OUTPUT_DIR/logs/hidden_states_server.log" 2>&1 &
VLLM_PID=$!
cleanup() {
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
    for ((i=0; i<15; i++)); do
        if ! kill -0 -- "-$VLLM_PID" 2>/dev/null; then break; fi
        sleep 1
    done
    kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + 1800))
until curl --noproxy '*' --connect-timeout 3 --max-time 5 -fsS \
    "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null || ((SECONDS >= deadline)); then
        tail -n 120 "$OUTPUT_DIR/logs/hidden_states_server.log"
        exit 1
    fi
    sleep 3
done
"$PYTHON_BIN" - <<'PY_ID'
import json, os, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f"http://127.0.0.1:{os.environ['VLLM_PORT']}/v1/models", timeout=15) as response:
    ids = [item['id'] for item in json.load(response)['data']]
if os.environ['MODEL'] not in ids:
    raise SystemExit(f'Target server model ID mismatch: {ids}')
PY_ID
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    "$OUTPUT_DIR/provenance/training_helpers/train_entry.py" --config "$OUTPUT_DIR/train_config.yaml" \
    2>&1 | tee "$OUTPUT_DIR/logs/train.log"
echo "Full-epoch training finished: $OUTPUT_DIR/checkpoints"
echo "Saved-weight validation history: $OUTPUT_DIR/checkpoints/joint_history.jsonl"
echo "Best offline reference EAL checkpoint: $OUTPUT_DIR/checkpoints/checkpoint_best"
printf 'Serve: DRAFT=%q TARGET_MODEL=%q NPU=7 PORT=8213 TARGET_EXECUTION=graph MODE=fast GRAPH_METRICS=0 bash %q\n' \
    "$OUTPUT_DIR/checkpoints/checkpoint_best" "$MODEL" "$SPEC_MAIN/speculators/examples/serve_dflash_target_graph.sh"
