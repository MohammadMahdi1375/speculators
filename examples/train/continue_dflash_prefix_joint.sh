#!/usr/bin/env bash
# Resume an explicitly selected joint checkpoint in a separate output directory.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
SPEC_MAIN="$(cd -- "$SPEC_MAIN" && pwd)"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-}}"
if [[ -z "$ENV_ROOT" ]]; then echo 'Activate Conda or set ENV_ROOT.' >&2; exit 2; fi
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then echo "Python not found: $PYTHON_BIN" >&2; exit 2; fi
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
HELPER_DIR="$SPEC_MAIN/speculators/scripts/dflash_prefix_joint"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:?Use train_dflash_prefix_joint.sh --resume-training CHECKPOINT}"
export RESUME_RUN_DIR="${RESUME_RUN_DIR:-}" # only needed if the checkpoint was copied outside its run
export OUTPUT_DIR="${OUTPUT_DIR:-$SPEC_MAIN/output/joint_resume_$(date +%Y%m%d_%H%M%S)_$$}"

# EPOCHS has its usual meaning: stop when this many total epochs are complete.
export EPOCHS="${EPOCHS:?Set EPOCHS to the total training epoch target}"
# auto: restore an unfinished LR plan; restart when extending past its horizon.
# restart: new cosine cycle at 0.1x source peak LRs, with 1% warmup by default.
export RESUME_LR_POLICY="${RESUME_LR_POLICY:-auto}" # auto | restore | restart
export RESUME_LR_SCALE="${RESUME_LR_SCALE:-${CONTINUE_LR_SCALE:-0.1}}"
export RESUME_WARMUP_RATIO="${RESUME_WARMUP_RATIO:-${CONTINUE_WARMUP_RATIO:-0.01}}"
# Optional independent peak overrides; empty uses the scale above.
export RESUME_BACKBONE_LR="${RESUME_BACKBONE_LR:-${CONTINUE_BACKBONE_LR:-}}"
export RESUME_MUON_LR="${RESUME_MUON_LR:-${CONTINUE_MUON_LR:-}}"
export RESUME_HEAD_LR="${RESUME_HEAD_LR:-${CONTINUE_HEAD_LR:-}}"
export CHECK_ONLY="${CHECK_ONLY:-0}"
export PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
if [[ "$CHECK_ONLY" != 0 && "$CHECK_ONLY" != 1 ]]; then echo 'CHECK_ONLY must be 0 or 1.' >&2; exit 2; fi
if [[ "$PREFLIGHT_ONLY" != 0 && "$PREFLIGHT_ONLY" != 1 ]]; then echo 'PREFLIGHT_ONLY must be 0 or 1.' >&2; exit 2; fi

# Keep the source run's number of training cards; the preparation checks it.
# Default: six training cards and two hidden-state-server cards on A2.
export VLLM_NPUS="${VLLM_NPUS:-0,1}" VLLM_DP="${VLLM_DP:-2}"
export TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}"
IFS=',' read -r -a TRAIN_DEVICE_LIST <<< "$TRAIN_NPUS"
export NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-${#TRAIN_DEVICE_LIST[@]}}"
export VLLM_PORT="${VLLM_PORT:-8092}" VLLM_RPC_PORT="${VLLM_RPC_PORT:-29692}"
VLLM_MEMORY_UTILIZATION="${VLLM_MEMORY_UTILIZATION:-0.85}"

# ============================================================
# Environment
# ============================================================
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
for name in RESUME_CHECKPOINT OUTPUT_DIR; do
    resolved_path="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${!name}")"
    printf -v "$name" '%s' "$resolved_path"
    export "$name"
done
# Model/data and the training recipe come from the selected checkpoint's run.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" "$HELPER_DIR/prepare_continuation.py"
export MODEL="$(cat "$OUTPUT_DIR/model_path.txt")"
export DATASET="$(cat "$OUTPUT_DIR/dataset_path.txt")"
read -r -a TARGET_LAYERS < "$OUTPUT_DIR/target_layers.txt"
MASK_TOKEN_ID="$(cat "$OUTPUT_DIR/mask_token_id.txt")"
SERVER_MAX_MODEL_LEN="$(cat "$OUTPUT_DIR/server_max_model_len.txt")"
cd "$SPEC_MAIN/speculators"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" scripts/preflight_dflash_prefix.py \
    --model "$MODEL" --dataset "$DATASET" --output "$OUTPUT_DIR" \
    --target-layers "${TARGET_LAYERS[@]}" --mask-token-id "$MASK_TOKEN_ID"
mkdir -p "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/hidden_states"
cp -a -- "$HELPER_DIR" "$OUTPUT_DIR/provenance/training_helpers"
cp -- "$SCRIPT_DIR/continue_dflash_prefix_joint.sh" "$OUTPUT_DIR/provenance/"
cp -a -- src/speculators/models/dflash_prefix "$OUTPUT_DIR/provenance/model_code"
for repo in speculators vllm vllm-ascend; do
    git -C "$SPEC_MAIN/$repo" rev-parse HEAD > "$OUTPUT_DIR/provenance/${repo}_commit.txt"
    git -C "$SPEC_MAIN/$repo" diff > "$OUTPUT_DIR/provenance/${repo}.patch"
done
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Resume configuration/data checks passed: $OUTPUT_DIR"
    echo 'No checkpoint weights copied; no NPU process started. Use a fresh OUTPUT_DIR for training.'
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
    "$OUTPUT_DIR/provenance/training_helpers/continue_entry.py" --config "$OUTPUT_DIR/train_config.yaml" \
    2>&1 | tee "$OUTPUT_DIR/logs/train.log"
echo "Resume finished: $OUTPUT_DIR/checkpoints"
echo "Saved-weight validation history: $OUTPUT_DIR/checkpoints/joint_history.jsonl"
echo "Best offline reference EAL checkpoint: $OUTPUT_DIR/checkpoints/checkpoint_best"
printf 'Serve: DRAFT=%q TARGET_MODEL=%q NPU=7 PORT=8213 TARGET_EXECUTION=graph MODE=fast GRAPH_METRICS=0 bash %q\n' \
    "$OUTPUT_DIR/checkpoints/checkpoint_best" "$MODEL" "$SPEC_MAIN/speculators/examples/serve_dflash_target_graph.sh"
