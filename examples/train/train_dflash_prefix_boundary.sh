#!/usr/bin/env bash
# Verification-boundary v2: support + selection; serving unchanged; settings editable.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
SPEC_MAIN="$(cd -- "$SPEC_MAIN" && pwd)"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-}}"
if [[ -z "$ENV_ROOT" ]]; then echo 'Activate your A2 Conda environment or set ENV_ROOT.' >&2; exit 2; fi
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
HELPER_DIR="$SPEC_MAIN/speculators/scripts/dflash_prefix_boundary"
# Empty means reuse the actual paths in the completed compiled run's recipe.
# Set MODEL and DATASET explicitly if files have moved since that run.
export MODEL="${MODEL:-}" DATASET="${DATASET:-}"
# Set your current compiled checkpoint here or pass INIT_CHECKPOINT=... .
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
if [[ -z "$INIT_CHECKPOINT" ]]; then echo 'Set INIT_CHECKPOINT to your trained compiled_prefix_v3 checkpoint.' >&2; exit 2; fi
# BASE_TRAIN_CONFIG is optional: inferred beside the teacher checkpoint.
export BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-}"
# BLOCK_SIZE is inferred from the teacher (8 or 16); an override must match.
export BLOCK_SIZE="${BLOCK_SIZE:-}"
export PILOT_STEPS="${PILOT_STEPS:-2000}" EVAL_EVERY="${EVAL_EVERY:-500}"
export SEED="${SEED:-42}"
export HEAD_LR="${HEAD_LR:-0.00003}" VAL_BATCHES="${VAL_BATCHES:-32}"
export DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-sdpa}" # sdpa | eager; explicitly restored when loading teacher
export MAX_ANCHORS="${MAX_ANCHORS:-32}" NUM_WORKERS="${NUM_WORKERS:-4}"
# Matched control: identical joint training, without boundary distillation.
export OBJECTIVE="${OBJECTIVE:-dual_boundary}" # dual_boundary | boundary | control
export TRAIN_LAST_LAYERS="${TRAIN_LAST_LAYERS:-all}" # all includes fc and hidden_norm; or 1 | 2
export BACKBONE_LR="${BACKBONE_LR:-0.000003}"
export FULL_CE_WEIGHT="${FULL_CE_WEIGHT:-1.0}"
export SELECTOR_CE_WEIGHT="${SELECTOR_CE_WEIGHT:-1.0}"
export BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.5}"
export SUPPORT_WEIGHT="${SUPPORT_WEIGHT:-0.25}"
export PRESERVE_WEIGHT="${PRESERVE_WEIGHT:-0.1}"
export TEACHER_CE_IMPROVEMENT="${TEACHER_CE_IMPROVEMENT:-0.05}"
export ANCHOR_CHUNK_SIZE="${ANCHOR_CHUNK_SIZE:-4}"
export CHECK_ONLY="${CHECK_ONLY:-0}"
if [[ "$CHECK_ONLY" != 0 && "$CHECK_ONLY" != 1 ]]; then echo 'CHECK_ONLY must be 0 or 1.' >&2; exit 2; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$SPEC_MAIN/output/dflash_prefix_boundary}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${OBJECTIVE}_$(date +%Y%m%d_%H%M%S)_$$}"
export VLLM_NPUS="${VLLM_NPUS:-0,1}" VLLM_DP="${VLLM_DP:-2}"
export VLLM_PORT="${VLLM_PORT:-8092}" VLLM_RPC_PORT="${VLLM_RPC_PORT:-29692}"
export TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6}" NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-5}"
unset PYTHONPATH ASCEND_LAUNCH_BLOCKING
source "${CANN_ENV_SH:-$CANN_ROOT/ascend-toolkit/set_env.sh}"
source "${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
# SOC_VERSION is deliberately inherited from your A2 installation.
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6"; fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY DFLASH_TP_GATHER
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ASCEND_BALANCE_SCHEDULING=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 OMP_PROC_BIND=false
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export HCCL_CONNECT_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
set -u
for name in MODEL DATASET INIT_CHECKPOINT OUTPUT_DIR; do
    if [[ -z "${!name}" ]]; then continue; fi
    resolved_path="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${!name}")"
    printf -v "$name" '%s' "$resolved_path"
    export "$name"
done
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" "$HELPER_DIR/prepare_experiment.py"
export MODEL="$(cat "$OUTPUT_DIR/model_path.txt")"
export DATASET="$(cat "$OUTPUT_DIR/dataset_path.txt")"
mapfile -t TARGET_LAYERS < "$OUTPUT_DIR/target_layers.txt"
BLOCK_SIZE="$(cat "$OUTPUT_DIR/block_size.txt")"
MASK_TOKEN_ID="$(cat "$OUTPUT_DIR/mask_token_id.txt")"
SERVER_MAX_MODEL_LEN="$(cat "$OUTPUT_DIR/server_max_model_len.txt")"
cd "$SPEC_MAIN/speculators"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" scripts/preflight_dflash_prefix.py \
    --model "$MODEL" --dataset "$DATASET" --output "$OUTPUT_DIR" \
    --target-layers "${TARGET_LAYERS[@]}" --mask-token-id "$MASK_TOKEN_ID"
cp -a -- "$HELPER_DIR" "$OUTPUT_DIR/provenance/training_helpers"
cp -- "$SCRIPT_DIR/train_dflash_prefix_boundary.sh" "$OUTPUT_DIR/provenance/"
cp -a -- src/speculators/models/dflash_prefix "$OUTPUT_DIR/provenance/model_code"
for repo in speculators vllm vllm-ascend; do
    git -C "$SPEC_MAIN/$repo" rev-parse HEAD > "$OUTPUT_DIR/provenance/${repo}_commit.txt"
    git -C "$SPEC_MAIN/$repo" diff > "$OUTPUT_DIR/provenance/${repo}.patch"
done
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Configuration checks passed: $OUTPUT_DIR. No server, NPU test, or training started."
    echo 'Use a fresh OUTPUT_DIR for the actual run.'
    exit 0
fi
# Test training operators on one assigned trainer NPU before starting the target server.
ASCEND_RT_VISIBLE_DEVICES="${TRAIN_NPUS%%,*}" "$PYTHON_BIN" "$HELPER_DIR/check_selector.py" \
    --device npu:0 --block-size "$BLOCK_SIZE" --teacher "$INIT_CHECKPOINT" \
    2>&1 | tee "$OUTPUT_DIR/logs/selector_check.log"
"$PYTHON_BIN" - <<'PY_PORTS'
import os, socket
for key in ('VLLM_PORT', 'VLLM_RPC_PORT'):
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', int(os.environ[key])))
PY_PORTS
setsid env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON_BIN" scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids "${TARGET_LAYERS[@]}" --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --provenance-dir "$OUTPUT_DIR/provenance" -- \
    --served-model-name "$MODEL" --data-parallel-size "$VLLM_DP" --tensor-parallel-size 1 \
    --data-parallel-rpc-port "$VLLM_RPC_PORT" --port "$VLLM_PORT" \
    --max-model-len "$SERVER_MAX_MODEL_LEN" --gpu-memory-utilization 0.85 --api-server-count 1 --renderer-num-workers 1 \
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
        tail -n 100 "$OUTPUT_DIR/logs/hidden_states_server.log"
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
echo "Pilot finished: $OUTPUT_DIR/checkpoints"
echo "Existing serving pipeline is unchanged. Each evaluation window has a step_<completed_steps> checkpoint link; the final link is step_${PILOT_STEPS}."
echo 'Use serve_dflash_target_graph.sh for every arm, with identical settings; proxy loss is not a speed measurement.'
