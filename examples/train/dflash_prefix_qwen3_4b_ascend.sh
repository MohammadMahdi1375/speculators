#!/usr/bin/env bash
# ============================================================================
# Online DFlash-Prefix training on Ascend NPU
# Qwen3-4B + prepared Open-PerfectBlend; five epochs from scratch.
# Requires the installed DFlash-Prefix bundle and review patch v2.
# Includes model-ID normalization and a live server identity check.
#
# Run:   bash dflash_prefix_qwen3_4b_ascend.sh
# Check: CHECK_ONLY=1 bash dflash_prefix_qwen3_4b_ascend.sh
# Edit the configuration below; no long environment-variable command is needed.
# ============================================================================
set -eo pipefail

# ============================================================================
# Environment
# ============================================================================
export SPEC_MAIN="${SPEC_MAIN:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main}"
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
export PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:?Activate vllm-dflash2-main first}/bin/python}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"

# ============================================================================
# Model, prepared dataset, and output
# ============================================================================
export MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATASET="${DATASET:-/home/n84449292/m84379596/Huggingface/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072/}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
export OUTPUT_DIR="/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_prefix_qwen3_4b_openperfectblend_bs16"
export CHECK_ONLY="${CHECK_ONLY:-0}"  # 0: train; 1: configuration/data checks only.

# ============================================================================
# Training: inherit the vanilla DFlash configuration for a matched comparison
# ============================================================================
# This reads configuration only; no vanilla or smoke checkpoint is loaded.
# Recipe: EPOCHS=5, LR=6e-4, SEQ_LENGTH=3072, SEED=42, BLOCK_SIZE=16,
# MAX_ANCHORS=512, NUM_LAYERS=5, TARGET_LAYER_IDS="1 9 17 25 33",
# DRAFT_VOCAB_SIZE=151936, MASK_TOKEN_ID=151669, checkpoint every epoch.
# Optimizer, attention, scheduler, and data split come from this run.yaml.
# If the default run.yaml is absent, resolve your original DFlash CLI defaults.
export BASELINE_CONFIG="${BASELINE_CONFIG:-$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/run.yaml}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
# Optional: set NUM_WORKERS=0 here if necessary; otherwise inherit the baseline.

# ============================================================================
# vLLM hidden-state server
# ============================================================================
export VLLM_NPUS="${VLLM_NPUS:-8,9}"
export VLLM_DP="${VLLM_DP:-2}"
export VLLM_PORT="${VLLM_PORT:-8092}"
export VLLM_RPC_PORT="${VLLM_RPC_PORT:-29692}"

# ============================================================================
# Trainer NPU assignments
# ============================================================================
export TRAIN_NPUS="${TRAIN_NPUS:-10,11,12,13,14,15}"
export NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-6}"

# ============================================================================
# Prefix selector
# ============================================================================
export PREFIX_TOP_K="${PREFIX_TOP_K:-16}"
export PREFIX_RANK="${PREFIX_RANK:-64}"
export PREFIX_LOSS_ALPHA="${PREFIX_LOSS_ALPHA:-1.0}"
export PREFIX_GATE_INIT="${PREFIX_GATE_INIT:-0.1}"
export PREFIX_LOSS_KIND="${PREFIX_LOSS_KIND:-target_ce}"
export PREFIX_DETACH_BACKBONE="${PREFIX_DETACH_BACKBONE:-1}"
export PREFIX_WALK_BACKEND="${PREFIX_WALK_BACKEND:-torch}"
# Detach blocks selector-loss gradients into backbone features. The backbone
# still trains with its DFlash loss. Both backbone and selector start fresh.

# ============================================================================
# Step 0: Set up CANN and local communication
# ============================================================================
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
unset DFLASH_TP_GATHER ASCEND_LAUNCH_BLOCKING
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
set -u
LAUNCHER_PATH="$("$PYTHON_BIN" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$0")"
# The hidden-state client compares model names as exact strings. Resolve once
# for both server and trainer, including trailing slashes and local symlinks.
MODEL="$("$PYTHON_BIN" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$MODEL")"
export MODEL
cd "$SPEC_MAIN/speculators"

# ============================================================================
# Step 1: Resolve the experiment and check model, data, devices, and ports
# ============================================================================
echo "Preparing full five-epoch scratch experiment..."
echo "Output: $OUTPUT_DIR"
echo "Shared server/trainer model ID: $MODEL"
"$PYTHON_BIN" - <<'PY_CONFIG'
import json
import math
import os
from pathlib import Path

import yaml
from speculators.train.config import TrainConfig
from speculators.train.config.schema import CONFIG_DESTS, nest_flat


def fail(message):
    raise SystemExit(message)


out = Path(os.environ["OUTPUT_DIR"]).expanduser().resolve()
model = Path(os.environ["MODEL"]).expanduser().resolve()
data = Path(os.environ["DATASET"]).expanduser().resolve()
base_path = Path(os.environ["BASELINE_CONFIG"]).expanduser().resolve()
default_base = (Path(os.environ["SPEC_MAIN"]) / "output" /
                "dflash_qwen3_4b_openperfectblend_bs16/checkpoints/run.yaml").resolve()
for name in ("CHECK_ONLY", "GRADIENT_CHECKPOINTING", "PREFIX_DETACH_BACKBONE"):
    if os.environ[name] not in {"0", "1"}:
        fail(f"{name} must be 0 or 1")
if out.exists() and any(out.iterdir()):
    fail(f"OUTPUT_DIR is not empty; choose a fresh directory: {out}")
for protected in (model, data, base_path.parent):
    if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
        fail(f"OUTPUT_DIR must be separate from {protected}")
if not (model / "config.json").is_file() or not data.is_dir():
    fail("MODEL/config.json or the prepared DATASET is missing")
if "dflash_prefix" not in TrainConfig.model_fields:
    fail("The installed training schema lacks dflash_prefix. Install the original bundle first.")
if "prefix_loss_kind" not in TrainConfig().flatten():
    fail("Install dflash_prefix_review_patch before using this reviewed launcher.")

# This duplicates the supplied vanilla command, including its omitted defaults.
baseline_argv = [
    "--verifier-name-or-path", str(model), "--data-path", str(data),
    "--vllm-endpoint", "http://127.0.0.1:8080/v1",
    "--save-path", str(default_base.parent), "--draft-vocab-size", "151936",
    "--epochs", "5", "--checkpoint-freq", "1", "--lr", "6e-4",
    "--total-seq-len", "3072", "--speculator-type", "dflash",
    "--block-size", "16", "--max-anchors", "512", "--num-layers", "5",
    "--target-layer-ids", "1", "9", "17", "25", "33",
    "--draft-arch", "qwen3", "--draft-hidden-act", "silu",
    "--draft-attn-impl", "sdpa", "--mask-token-id", "151669",
    "--scheduler-type", "cosine", "--logger", "tensorboard",
    "--run-name", "dflash_qwen3_4b_full_vocab_openperfectblend_npu",
    "--log-dir", "./logs/dflash_qwen3_4b_full_vocab_openperfectblend_npu",
    "--on-missing", "generate", "--on-generate", "delete",
    "--request-timeout", "180", "--max-retries", "8", "--log-freq", "10",
    "--seed", "42",
]
if base_path.is_file():
    base_text = base_path.read_text()
    base = TrainConfig.resolve(["--config", str(base_path)])
    source = str(base_path)
else:
    if base_path != default_base:
        fail(f"Explicit BASELINE_CONFIG does not exist: {base_path}")
    base = TrainConfig.resolve(baseline_argv)
    base_text = base.dump_yaml()
    source = "Supplied vanilla command, resolved with the currently installed trainer defaults"
b = base.flatten()
expected = {
    "speculator_type": "dflash", "epochs": 5, "lr": 6e-4,
    "total_seq_len": 3072, "block_size": 16, "max_anchors": 512,
    "num_layers": 5, "draft_arch": "qwen3", "draft_attn_impl": "sdpa",
    "draft_vocab_size": 151936, "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33], "seed": 42,
}
for key, value in expected.items():
    if b[key] != value:
        fail(f"Baseline {key}={b[key]!r} differs from the supplied recipe ({value!r}). Check BASELINE_CONFIG.")
for key, path in (("verifier_name_or_path", model), ("data_path", data)):
    if Path(b[key]).expanduser().resolve() != path:
        fail(f"Baseline {key} does not match this experiment: {b[key]}")
if b["from_pretrained"] or b["draft_config"] or b["prefix_backbone_init"]:
    fail("Expected a baseline trained from scratch using the decoder-shaping flags")
if b["sample_from_anchor"] or b["max_steps"] is not None:
    fail("Expected sample_from_anchor=False and no baseline step cap")
if b["full_attention_indices"] and len(set(b["full_attention_indices"])) != b["num_layers"]:
    fail("Mixed full/sliding draft layers require v2 serving, which this prefix adapter does not support")
if not b["full_attention_indices"] and b["sliding_window_non_causal"]:
    fail("Noncausal sliding attention needs additional Ascend serving parity work; use the supplied causal-SWA baseline")
if b["hidden_states_backend"] != "file":
    fail("This Ascend launcher expects the working file hidden-state backend")
if b["on_missing"] != "generate" or b["on_generate"] != "delete":
    fail("Expected the online generate/delete policy from the supplied vanilla script")

f = dict(b)
f.update(
    speculator_type="dflash_prefix", verifier_name_or_path=str(model),
    data_path=str(data), save_path=str(out / "checkpoints"),
    vllm_endpoint=f"http://127.0.0.1:{int(os.environ['VLLM_PORT'])}/v1",
    hidden_states_path=str(out / "hidden_states"),
    run_name=f"dflash_prefix_scratch_5ep_{os.environ['RUN_TAG']}",
    log_dir=str(out / "logs/tensorboard"),
    epochs=5, max_steps=None, checkpoint_freq=1.0, save_best=False,
    no_resume_from_checkpoint=True, dry_run=False,
    from_pretrained="", draft_config="", prefix_backbone_init=None,
    prefix_freeze_backbone=False,
    prefix_detach_backbone=os.environ["PREFIX_DETACH_BACKBONE"] == "1",
    prefix_loss_kind=os.environ["PREFIX_LOSS_KIND"],
    prefix_walk_backend=os.environ["PREFIX_WALK_BACKEND"],
    gradient_checkpointing=os.environ["GRADIENT_CHECKPOINTING"] == "1",
    prefix_top_k=int(os.environ["PREFIX_TOP_K"]),
    prefix_rank=int(os.environ["PREFIX_RANK"]),
    prefix_loss_alpha=float(os.environ["PREFIX_LOSS_ALPHA"]),
    prefix_gate_init=float(os.environ["PREFIX_GATE_INIT"]),
)
if "NUM_WORKERS" in os.environ:
    f["num_workers"] = int(os.environ["NUM_WORKERS"])
if f["num_workers"] < 0 or f["prefix_top_k"] > f["draft_vocab_size"]:
    fail("Invalid NUM_WORKERS or PREFIX_TOP_K")
if not all(math.isfinite(f[k]) for k in ("prefix_gate_init", "prefix_loss_alpha")):
    fail("Prefix hyperparameters must be finite")
cfg = TrainConfig.from_flat(f)
resolved = cfg.flatten()
for key in ("from_pretrained", "draft_config", "prefix_backbone_init", "prefix_freeze_backbone"):
    if resolved[key]:
        fail(f"Scratch initialization violated: {key}={resolved[key]}")

# Persist materialized settings, including defaults. Exclude unused algorithm
# groups to avoid misleading warnings, but retain a complete flat JSON snapshot.
nested = nest_flat({k: v for k, v in resolved.items() if k in CONFIG_DESTS})
for unused in ("dflash2", "dspark", "peagle", "mtp"):
    nested.pop(unused, None)
nested["backend"] = dict(cfg.backend_args)
provenance = out / "provenance"
provenance.mkdir(parents=True)
(out / "logs").mkdir()
config_path = out / "train_config.yaml"
config_path.write_text(yaml.safe_dump({"train": nested}, sort_keys=False))
reloaded = TrainConfig.resolve(["--config", str(config_path)]).flatten()
if reloaded != resolved:
    fail("Generated config failed to round-trip; no NPU processes were started")
changes = {k: {"baseline": b.get(k), "prefix": v}
           for k, v in resolved.items() if b.get(k) != v}
(provenance / "baseline_input.yaml").write_text(base_text)
(provenance / "baseline_resolved.json").write_text(json.dumps(b, indent=2) + "\n")
(provenance / "prefix_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
(provenance / "experiment.json").write_text(json.dumps({
    "baseline_config_source": source,
    "baseline_defaults": "Resolved using the installed trainer; omitted historical defaults cannot be recovered from run.yaml alone.",
    "initialization": "Random draft backbone and prefix head; normal target-owned DFlash weights",
    "dataset": "Full prepared dataset with the inherited train/validation split; no subset or retokenization",
    "changes_from_baseline": changes,
    "optimizer_adjustment": "With Muon, prefix_head.token_codes.weight belongs to AdamW (embedding)",
    "devices": {k: os.environ[k] for k in ("VLLM_NPUS", "VLLM_DP", "TRAIN_NPUS", "NUM_TRAIN_NPUS")},
    "ports": {k: int(os.environ[k]) for k in ("VLLM_PORT", "VLLM_RPC_PORT")},
}, indent=2) + "\n")
print(f"Baseline configuration: {source}")
for key in ("epochs", "max_steps", "lr", "optimizer", "loss_fn", "loss_implementation",
            "num_layers", "sliding_window", "full_attention_indices", "sliding_window_non_causal",
            "scheduler_type", "scheduler_warmup_steps", "scheduler_warmup_ratio",
            "train_data_ratio", "total_seq_len", "max_anchors", "num_workers", "gradient_checkpointing"):
    print(f"  {key}: {resolved[key]}")
print(f"Prefix: top-K={resolved['prefix_top_k']}, rank={resolved['prefix_rank']}, "
      f"loss alpha={resolved['prefix_loss_alpha']}, gate={resolved['prefix_gate_init']}")
print(f"Selector loss: {resolved['prefix_loss_kind']}; detach backbone inputs: {resolved['prefix_detach_backbone']}")
print(f"Serving walk backend: {resolved['prefix_walk_backend']}")
print("Trainable: draft backbone and prefix head; no draft checkpoint loaded")
print(f"Resolved config and comparison: {config_path}, {provenance / 'experiment.json'}")
PY_CONFIG

"$PYTHON_BIN" scripts/preflight_dflash_prefix.py \
    --model "$MODEL" --dataset "$DATASET" --output "$OUTPUT_DIR" \
    --target-layers 1 9 17 25 33 --mask-token-id 151669
"$PYTHON_BIN" - <<'PY_PORTS'
import os
import socket

ports = [int(os.environ[k]) for k in ("VLLM_PORT", "VLLM_RPC_PORT")]
if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
    raise SystemExit("VLLM_PORT and VLLM_RPC_PORT must be distinct valid ports")
for port in ports:
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            raise SystemExit(f"Port {port} is unavailable; choose a different port: {exc}")
PY_PORTS

# ============================================================================
# Step 2: Prepare NPU training and save the resolved experiment
# ============================================================================
# Per-run entry point: register torch_npu and classify the selector's vocabulary
# embedding like the other embeddings/codebooks. Does not edit shared source.
cat > "$OUTPUT_DIR/provenance/train_entry.py" <<'PY_TRAIN'
import os


def configure_prefix_optimizer():
    from speculators.train import optimizers

    hints = optimizers._ADAMW_NAME_HINTS
    if "token_codes" not in hints:
        optimizers._ADAMW_NAME_HINTS = (*hints, "token_codes")


if __name__ == "__main__":
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    configure_prefix_optimizer()
    from speculators.train.cli import main
    from speculators.train.config import TrainConfig

    config = TrainConfig.resolve()
    flat = config.flatten()
    if (flat["speculator_type"] != "dflash_prefix"
            or any(flat[k] for k in ("from_pretrained", "draft_config", "prefix_backbone_init", "prefix_freeze_backbone"))
            or flat["max_steps"] is not None or flat["epochs"] != 5
            or not flat["no_resume_from_checkpoint"]):
        raise SystemExit("This entry point is for the full five-epoch scratch experiment")
    main(config)
PY_TRAIN

cp -- "$LAUNCHER_PATH" "$OUTPUT_DIR/provenance/launcher.sh"
for repo in speculators vllm vllm-ascend; do
    git -C "$SPEC_MAIN/$repo" rev-parse HEAD > "$OUTPUT_DIR/provenance/${repo}_commit.txt"
    git -C "$SPEC_MAIN/$repo" diff > "$OUTPUT_DIR/provenance/${repo}.patch"
done
"$PYTHON_BIN" - > "$OUTPUT_DIR/provenance/packages.txt" <<'PY_PACKAGES'
from importlib.metadata import distributions

print("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})))
PY_PACKAGES
"$PYTHON_BIN" scripts/snapshot_dflash_prefix.py \
    --spec-main "$SPEC_MAIN" --output "$OUTPUT_DIR/provenance/prefix_source"
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Configuration/data checks passed. No server or training process was started."
    echo "Inspection files: $OUTPUT_DIR (choose a fresh OUTPUT_DIR for the actual run)."
    exit 0
fi

# ============================================================================
# Step 3: Check the prefix selector on NPU
# ============================================================================
# Validate the revised head in FP32 and BF16 on the first trainer NPU before
# starting either the hidden-state server or the six-rank training process.
echo "Checking selector arithmetic on physical NPU ${TRAIN_NPUS%%,*}..."
ASCEND_RT_VISIBLE_DEVICES="${TRAIN_NPUS%%,*}" "$PYTHON_BIN" \
    scripts/check_dflash_prefix.py --device npu:0 \
    --rank "$PREFIX_RANK" --top-k "$PREFIX_TOP_K" --steps 15 \
    --backend "$PREFIX_WALK_BACKEND" \
    2>&1 | tee "$OUTPUT_DIR/logs/selector_check.log"

# ============================================================================
# Step 4: Start the hidden-state server and wait until ready
# ============================================================================
mkdir -p "$OUTPUT_DIR/hidden_states"
VLLM_LOG="$OUTPUT_DIR/logs/hidden_states_server.log"
echo "Starting hidden-state server on NPUs $VLLM_NPUS, HTTP=$VLLM_PORT, DP RPC=$VLLM_RPC_PORT"
echo "Server log: $VLLM_LOG"
setsid env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON_BIN" scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids 1 9 17 25 33 \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --provenance-dir "$OUTPUT_DIR/provenance" -- \
    --served-model-name "$MODEL" \
    --data-parallel-size "$VLLM_DP" --tensor-parallel-size 1 \
    --data-parallel-rpc-port "$VLLM_RPC_PORT" --port "$VLLM_PORT" \
    --max-model-len 3328 --gpu-memory-utilization 0.85 \
    > "$VLLM_LOG" 2>&1 &
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
deadline=$((SECONDS + ${VLLM_BOOT_TIMEOUT:-1800}))
next_notice=$((SECONDS + 30))
until curl --noproxy '*' --connect-timeout 3 --max-time 5 -fsS \
    "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null || ((SECONDS >= deadline)); then
        tail -n 100 "$VLLM_LOG"; exit 1
    fi
    if ((SECONDS >= next_notice)); then
        echo "Waiting for hidden-state server; see $VLLM_LOG"
        next_notice=$((SECONDS + 30))
    fi
    sleep 3
done
# Match the exact first advertised ID, just as ArrowDataset._setup_client does.
# Abort before torchrun if the server and resolved training config disagree.
"$PYTHON_BIN" - <<'PY_SERVER_MODEL'
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

out = Path(os.environ["OUTPUT_DIR"])
resolved = json.loads((out / "provenance/prefix_resolved.json").read_text())
expected = resolved["verifier_name_or_path"]
endpoint = resolved["vllm_endpoint"].rstrip("/") + "/models"
check = {"endpoint": endpoint, "trainer_model_id": expected,
         "server_launch_model_id": os.environ["MODEL"], "matched": False}
try:
    if os.environ["MODEL"] != expected:
        raise ValueError("Server launch model ID differs from the resolved training config")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(endpoint, headers={"Authorization": "Bearer EMPTY"})
    with opener.open(request, timeout=15) as response:
        models = json.load(response)
    entries = models.get("data", []) if isinstance(models, dict) else []
    if not isinstance(entries, list) or not entries or not all(
        isinstance(entry, dict) and isinstance(entry.get("id"), str) for entry in entries
    ):
        raise ValueError("Server /v1/models returned no valid model IDs")
    check["advertised_model_ids"] = [entry["id"] for entry in entries]
    if entries[0]["id"] != expected:
        raise ValueError(
            f"Trainer expects {expected!r}, but the server's first model ID is "
            f"{entries[0]['id']!r}. Training has not been started."
        )
    check["matched"] = True
except (OSError, ValueError, urllib.error.URLError) as exc:
    check["error"] = str(exc)
    raise SystemExit(f"Hidden-state server model identity check failed: {exc}") from exc
finally:
    (out / "provenance/server_model_check.json").write_text(json.dumps(check, indent=2) + "\n")
print(f"Hidden-state server model identity passed: {expected}")
PY_SERVER_MODEL
# ============================================================================
# Step 5: Train DFlash-Prefix
# ============================================================================
echo "Starting five full epochs on NPUs $TRAIN_NPUS ($NUM_TRAIN_NPUS ranks)."
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    "$OUTPUT_DIR/provenance/train_entry.py" --config "$OUTPUT_DIR/train_config.yaml" \
    2>&1 | tee "$OUTPUT_DIR/logs/train.log"
echo "Completed. Checkpoints: $OUTPUT_DIR/checkpoints"
