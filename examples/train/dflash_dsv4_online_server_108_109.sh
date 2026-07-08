#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DeepSeek-V4 DFlash online hidden-state server
#
# Run on:
#   node 108: bash dflash_dsv4_online_server_108_109.sh 0
#   node 109: bash dflash_dsv4_online_server_108_109.sh 1
#
# This script uses 108 + 109 for the vLLM target/verifier only.
# Trainer runs separately on 115.
# =============================================================================

NODE_RANK="${1:?usage: bash dflash_dsv4_online_server_108_109.sh <node_rank 0|1>}"

# ===================== paths/config to modify if needed =====================
SPEC_MAIN="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main"
CONDA_ENV="/home/n84449292/m84379596/conda/vllm-ascend-0202"

MODEL="/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16"

# Must be visible/read-writable from 108, 109, and 115.
ONLINE_HS_PATH="${ONLINE_HS_PATH:-/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/dflash_online_hidden_states/dsv4_284b}"

MASTER_ADDR="${MASTER_ADDR:-80.5.5.108}"
MASTER_PORT="${MASTER_PORT:-29610}"

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-30000}"

NNODES=2
NPROC_PER_NODE=8
TARGET_TP_SIZE=16
LOCAL_NPUS="${LOCAL_NPUS:-0,1,2,3,4,5,6,7}"

TARGET_MAX_MODEL_LEN="${DFLASH_TARGET_MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

# Must match trainer TARGET_LAYER_IDS.
TARGET_LAYER_IDS_JSON='[2,20,40]'
# ============================================================================

# CANN env scripts are not safe under `set -u` because they reference
# variables such as PYTHONPATH/ZSH_VERSION before defining them.
unset PYTHONPATH
export PYTHONPATH=""
export ZSH_VERSION="${ZSH_VERSION:-}"

set +u
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh
set -u

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

export VLLM_USE_V1=1
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export DSV4_VLLM_SERVE_PATCH=1
export DFLASH_DISABLE_QLI=1

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"

if [ "$NODE_RANK" = "0" ]; then
    PEER_IP="80.5.5.109"
else
    PEER_IP="80.5.5.108"
fi

NET_IFACE="$(
python - <<PY
import subprocess
peer = "$PEER_IP"
out = subprocess.check_output(["bash", "-lc", f"ip route get {peer}"], text=True)
parts = out.split()
print(parts[parts.index("dev") + 1] if "dev" in parts else "")
PY
)"

if [ -z "$NET_IFACE" ]; then
    echo "ERROR: could not detect network interface to $PEER_IP"
    exit 1
fi

export GLOO_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"
export VLLM_HOST_IP="$(
python - <<PY
import subprocess
iface = "$NET_IFACE"
cmd = "ip -4 addr show " + iface + " | awk '/inet /{print \$2}' | cut -d/ -f1 | head -1"
print(subprocess.check_output(["bash", "-lc", cmd], text=True).strip())
PY
)"

mkdir -p "$ONLINE_HS_PATH"
mkdir -p "$SPEC_MAIN/speculators/logs"

SPEC_JSON="$(
python - <<PY
import json
target_layer_ids = $TARGET_LAYER_IDS_JSON
print(json.dumps({
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,
    "draft_model_config": {
        "hf_config": {
            "eagle_aux_hidden_state_layer_ids": target_layer_ids
        }
    }
}))
PY
)"

KV_JSON="$(
ONLINE_HS_PATH="$ONLINE_HS_PATH" python - <<'PY'
import json
import os
print(json.dumps({
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_buffer_size": 134217728,
    "kv_connector_extra_config": {
        "shared_storage_path": os.environ["ONLINE_HS_PATH"]
    }
}))
PY
)"

# Detect exact CLI option names in your local vLLM checkout.
HELP_TEXT="$(python -m vllm.entrypoints.openai.api_server --help 2>&1 || true)"

if grep -q -- "--speculative_config" <<<"$HELP_TEXT"; then
    SPEC_ARG="--speculative_config"
else
    SPEC_ARG="--speculative-config"
fi

if grep -q -- "--kv-transfer-config" <<<"$HELP_TEXT"; then
    KV_ARG="--kv-transfer-config"
else
    KV_ARG="--kv_transfer_config"
fi

echo "=================================================================="
echo "[server node $NODE_RANK] DeepSeek-V4 hidden-state vLLM server"
echo "MODEL=$MODEL"
echo "ONLINE_HS_PATH=$ONLINE_HS_PATH"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SERVER_HOST=$SERVER_HOST"
echo "SERVER_PORT=$SERVER_PORT"
echo "TARGET_TP_SIZE=$TARGET_TP_SIZE"
echo "TARGET_MAX_MODEL_LEN=$TARGET_MAX_MODEL_LEN"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "NET_IFACE=$NET_IFACE"
echo "VLLM_HOST_IP=$VLLM_HOST_IP"
echo "SPEC_ARG=$SPEC_ARG"
echo "KV_ARG=$KV_ARG"
echo "SPEC_JSON=$SPEC_JSON"
echo "KV_JSON=$KV_JSON"
echo "=================================================================="

cd "$SPEC_MAIN/speculators"

ASCEND_RT_VISIBLE_DEVICES="$LOCAL_NPUS" torchrun \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --nproc_per_node "$NPROC_PER_NODE" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --served-model-name "$MODEL" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --tokenizer-mode deepseek_v4 \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --tensor-parallel-size "$TARGET_TP_SIZE" \
    --distributed-executor-backend external_launcher \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$TARGET_MAX_MODEL_LEN" \
    --max-num-seqs 1 \
    --no-enable-chunked-prefill \
    "$SPEC_ARG" "$SPEC_JSON" \
    "$KV_ARG" "$KV_JSON"
