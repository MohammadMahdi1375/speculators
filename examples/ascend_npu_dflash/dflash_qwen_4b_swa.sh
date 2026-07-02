#!/bin/bash
# ============================================================================
# DFlash SEPARATE (online) trainer for Qwen3-4B — vLLM hidden-states server +
# FSDP trainer on disjoint NPUs (server NPU 0, DP=1; trainer NPUs 1-7, DP=7).
#
# Adapted from the Qwen3-8B version. Intended differences vs that script:
# bash examples/ascend_npu_dflash/dflash_qwen3_4b.sh 2>&1 | tee logs/qwen4b_dflash.log
#   - verifier model  : Qwen3-8B -> Qwen3-4B
#   - dataset         : 10k subset -> open_perfectblend_full.jsonl
#   - NPU split       : 4 server / 4 trainer -> 1 server / 7 trainer
#
# DRAFTER DESIGN (matches z-lab/Qwen3-4B-DFlash-b16/config.json):
# The DFlash trainer derives the WHOLE draft architecture from the verifier
# config in create_transformer_layer_config() (scripts/train.py); there are NO
# CLI flags for hidden_size / intermediate_size / heads / head_dim.
#
# NOTE: the curl health URL and --vllm-endpoint are PLAIN URLs.
# ============================================================================
set -eo pipefail

export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main

# Start clean first, before sourcing CANN.
unset PYTHONPATH

# Source CANN/ATB after clearing old paths, so CANN can add acl/tbe/te paths.
source /home/n84449292/m84379596/CANN/CANN9.0.0/ascend-toolkit/set_env.sh
source /home/n84449292/m84379596/CANN/CANN9.0.0/nnal/atb/set_env.sh

# Prepend our source repos, but KEEP the CANN PYTHONPATH entries.
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"

export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
unset DFLASH_TP_GATHER
set -u

cd "$SPEC_MAIN/speculators"

echo "=== Step 0: import sanity check ==="
python - <<'PY'
import inspect
import speculators
from speculators.data_generation.preprocessing import load_and_preprocess_dataset

expected = "/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators/src/speculators"
actual = speculators.__file__
sig = inspect.signature(load_and_preprocess_dataset)

print("speculators:", actual)
print("load_and_preprocess_dataset:", sig)

if not actual.startswith(expected):
    raise SystemExit(f"ERROR: wrong speculators import path: {actual}")

if "trust_remote_code" not in str(sig):
    raise SystemExit("ERROR: wrong speculators version: missing trust_remote_code argument")
PY

# ============ Configuration ============
MODEL="/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"
DATASET="/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend_full.jsonl"
OUTPUT_DIR="./output/dflash_qwen4b_swa_new_dataset"
VLLM_PORT=8000
MAX_SAMPLES=1420905
# MAX_SAMPLES=1420

# ---- Sequence lengths -------------------------------------------------------
# PREP_SEQ_LEN caps each sample in prepare_data. The open-perfectblend convs run
# up to ~4.3k tokens, so 3072 was TRUNCATING the assistant turn (cause of the
# "No assistant response spans found" warnings). 8192 keeps them intact with
# headroom. (Qwen3-4B supports up to 40960 positions.)
PREP_SEQ_LEN=3072

# vLLM must be able to serve hidden states for the longest sample, so its
# --max-model-len MUST be >= PREP_SEQ_LEN in online mode.
VLLM_MAX_MODEL_LEN=$((PREP_SEQ_LEN + 256))

# ---- "Batch size" lever (token-budget multipack) ----------------------------
# This trainer has NO --batch-size flag. Each rank fills a batch up to
# TOTAL_SEQ_LEN tokens (MultipackDistributedBatchSamplerV2), so sequences/rank =
# TOTAL_SEQ_LEN / AVG_sample_len -- a VARIABLE count, not a fixed number. Your
# samples are mostly far shorter than PREP_SEQ_LEN, so you'll typically pack
# several per rank at the value below. Raise it for bigger batches; must be
# >= PREP_SEQ_LEN so a max-length sample still fits in a batch.
TOTAL_SEQ_LEN=3072
EPOCHS=1
LR=6e-4
SEED=42

SPECULATOR_TYPE="dflash"
BLOCK_SIZE=16
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"
SLIDING_WINDOW=1024
SLIDING_WINDOW_INDICES="0 1 2 3 4"

# VOCAB: Qwen3-4B verifier full vocab is 151936. The z-lab Qwen3-4B-DFlash-b16
# checkpoint uses the FULL vocab, so omit --draft-vocab-size when equal.
VERIFIER_VOCAB=151936
DRAFT_VOCAB_SIZE=151936
VOCAB_FLAG=""
if [ "$DRAFT_VOCAB_SIZE" -lt "$VERIFIER_VOCAB" ]; then
    VOCAB_FLAG="--draft-vocab-size $DRAFT_VOCAB_SIZE"
fi

VLLM_NPUS="0,1"               # 1 NPU serves hidden states, DP=1
TRAIN_NPUS="2,3,4,5,6,7"  # 7 NPUs train, draft DP=7
NUM_TRAIN_NPUS=6
VLLM_DP=2
# =======================================================================

echo "=== Step 0a: drafter config preview (what the trainer injects) ==="
DRAFT_VOCAB_FOR_PREVIEW="$DRAFT_VOCAB_SIZE"
MODEL="$MODEL" NUM_LAYERS="$NUM_LAYERS" BLOCK_SIZE="$BLOCK_SIZE" \
MAX_ANCHORS="$MAX_ANCHORS" TARGET_LAYER_IDS="$TARGET_LAYER_IDS" \
DRAFT_VOCAB="$DRAFT_VOCAB_FOR_PREVIEW" python - <<'PY' || echo "[preview] skipped (non-fatal)"
import json, os, sys

try:
    model = os.environ["MODEL"]
    v = json.load(open(os.path.join(model, "config.json")))
    v = v.get("text_config", v)

    hidden = v["hidden_size"]
    n_heads = v["num_attention_heads"]
    n_kv = v["num_key_value_heads"]
    head_dim = v.get("head_dim")

    if head_dim and hidden % n_heads != 0 and hidden % head_dim == 0:
        n_heads = hidden // head_dim
        if n_heads % n_kv != 0:
            n_kv = n_heads

    n_layers = int(os.environ["NUM_LAYERS"])
    target_ids = [int(x) for x in os.environ["TARGET_LAYER_IDS"].split()]
    draft_vocab = int(os.environ["DRAFT_VOCAB"])
    full_vocab = v["vocab_size"]
    block_size = int(os.environ["BLOCK_SIZE"])

    drafter = {
        "speculators_model_type": "dflash",
        "architectures": ["DFlashSpeculator"],
        "block_size": block_size,
        "max_anchors": int(os.environ["MAX_ANCHORS"]),
        "draft_vocab_size": full_vocab if draft_vocab >= full_vocab else draft_vocab,
        "mask_token_id": 151669,
        "aux_hidden_state_layer_ids": target_ids,
        "sliding_window_non_causal": False,
        "proposal_speculative_tokens": block_size - 1,
        "transformer_layer_config": {
            "model_type": v.get("model_type", "qwen3"),
            "vocab_size": full_vocab,
            "hidden_size": hidden,
            "intermediate_size": v["intermediate_size"],
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv,
            "head_dim": head_dim,
            "hidden_act": "silu",
            "max_position_embeddings": v.get("max_position_embeddings"),
            "rms_norm_eps": v.get("rms_norm_eps"),
            "rope_theta": v.get("rope_theta"),
            "tie_word_embeddings": False,
            "layer_types": ["full_attention"] * n_layers,
        },
        "_info": {
            "vocab_mapping_used": draft_vocab < full_vocab,
            "verifier_num_layers": v["num_hidden_layers"],
            "note": "z-lab config.json flattens transformer_layer_config and adds "
                    "num_target_layers / dflash_config / auto_map; dims are identical.",
        },
    }

    print(json.dumps(drafter, indent=2))
except Exception as e:
    print(f"[preview] could not derive drafter config: {e}", file=sys.stderr)
PY

echo "=== Step 1: prepare_data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$PREP_SEQ_LEN" \
    --overwrite

# Drop stale vocab mappings from any prior run so they regenerate.
rm -f "$OUTPUT_DIR"/d2t.npy "$OUTPUT_DIR"/t2d.npy

echo "=== Step 2: launch vLLM server (NPUs $VLLM_NPUS, DP=$VLLM_DP) ==="
mkdir -p "$OUTPUT_DIR"
VLLM_LOG="$OUTPUT_DIR/vllm_server.log"
echo "vLLM output -> $VLLM_LOG"

ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
       --max-model-len "$VLLM_MAX_MODEL_LEN" \
       --gpu-memory-utilization 0.85 2>&1 | tee "$VLLM_LOG" &

VLLM_PROCS="launch_vllm.py|vllm.entrypoints|EngineCore|APIServer|vllm serve|from_engine_args"
cleanup() {
    echo "Stopping vLLM..."
    pkill -f "$VLLM_PROCS" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server health at http://localhost:${VLLM_PORT}/health ..."
WAITED=0
BOOT_TIMEOUT=${VLLM_BOOT_TIMEOUT:-1800}
GRACE=120

until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if [ "$WAITED" -ge "$GRACE" ] && ! pgrep -f "$VLLM_PROCS" >/dev/null 2>&1; then
        echo "ERROR: no vLLM process alive and port still down after ${WAITED}s. Last 40 lines:"
        tail -n 40 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi

    if [ "$WAITED" -ge "$BOOT_TIMEOUT" ]; then
        echo "ERROR: vLLM not healthy after ${BOOT_TIMEOUT}s. Last 40 lines:"
        tail -n 40 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi

    sleep 5
    WAITED=$(( WAITED + 5 ))
    [ $(( WAITED % 30 )) -eq 0 ] && echo "  ...still waiting (${WAITED}s)" || true
done

echo "Server ready after ${WAITED}s."

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

echo "=== Step 3: train (online, NPUs $TRAIN_NPUS, draft DP=$NUM_TRAIN_NPUS) ==="
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    $VOCAB_FLAG \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$TOTAL_SEQ_LEN" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --draft-attn-impl sdpa \
    --sliding-window "$SLIDING_WINDOW" \
    --sliding-window-indices $SLIDING_WINDOW_INDICES \
    --mask-token-id 151669 \
    --scheduler-type cosine \
    --logger tensorboard \
    --run-name dflash_separate_qwen3_4b \
    --log-dir ./logs/separate_qwen3_4b \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 180 \
    --max-retries 8 \
    --log-freq 10 \
    --no-resume-from-checkpoint \
    --seed "$SEED"

# --loss-fn ce \
echo "=== Step 4: final drafter config (authoritative, from saved checkpoint) ==="
FINAL_CFG=$(ls -t "$OUTPUT_DIR"/checkpoints/*/config.json 2>/dev/null | head -1)

if [ -n "${FINAL_CFG:-}" ]; then
    echo "--- $FINAL_CFG ---"
    python -m json.tool "$FINAL_CFG" 2>/dev/null || cat "$FINAL_CFG"
else
    echo "[final] no saved config.json found under $OUTPUT_DIR/checkpoints/"
fi

echo "Done. Checkpoints: $OUTPUT_DIR/checkpoints/  |  TB logs: ./logs/separate_qwen3_4b"