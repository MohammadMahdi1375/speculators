#!/usr/bin/env python3
"""Prepare an isolated, frozen-backbone compiled-prefix distillation pilot."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path


def load_teacher(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    cfg = json.loads((checkpoint / "config.json").read_text())
    if cfg.get("speculators_model_type") != "dflash_prefix" or cfg.get("prefix_selector_kind") != "local_prefix_v2":
        raise ValueError("TEACHER_CHECKPOINT must be your trained local_prefix_v2 checkpoint")
    if cfg.get("block_size") not in (8, 16) or cfg.get("prefix_top_k") != 16:
        raise ValueError("This pilot supports block size 8 or 16 and top-K=16")
    if cfg.get("prefix_history_scale", 1) != 1 or cfg.get("prefix_inference_gate_scale", 1) != 1:
        raise ValueError("Use the original teacher with history and inference scales equal to one")
    if cfg.get("prefix_disable_selector", False):
        raise ValueError("The teacher selector must be enabled")
    layers = cfg.get("aux_hidden_state_layer_ids")
    if not layers or not all(type(x) is int and x >= 0 for x in layers):
        raise ValueError("Teacher config must record aux_hidden_state_layer_ids")
    if type(cfg.get("mask_token_id")) is not int:
        raise ValueError("Teacher config must record mask_token_id")
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Teacher checkpoint contains no safetensors weights")
    return checkpoint, cfg


def find_base_config(checkpoint, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"BASE_TRAIN_CONFIG is not a file: {path}")
        return path
    checkpoint = Path(checkpoint).resolve(strict=True)
    # Includes checkpoints/run.yaml and its enclosing run's train_config.yaml.
    for parent in (checkpoint, checkpoint.parent, checkpoint.parent.parent):
        for name in ("run.yaml", "train_config.yaml"):
            path = parent / name
            if path.is_file():
                return path
    raise ValueError("Cannot find the teacher training recipe. Set BASE_TRAIN_CONFIG to its run.yaml or train_config.yaml")


def allocation(server, trainer, dp, count):
    def parse(value):
        tokens = value.split(",")
        if any(not token.strip().isdigit() for token in tokens):
            raise ValueError("NPU lists must be comma-separated nonnegative integers")
        ids = [int(token) for token in tokens]
        if len(set(ids)) != len(ids):
            raise ValueError("NPU lists must not repeat devices")
        return ids
    a, b = parse(server), parse(trainer)
    if len(a) != dp or len(b) != count or dp < 1 or count < 1 or set(a) & set(b):
        raise ValueError("Server/trainer NPUs must be disjoint and match VLLM_DP / NUM_TRAIN_NPUS")
    return a, b


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(env):
    import yaml
    from speculators.train.config import TrainConfig
    from speculators.train.config.schema import CONFIG_DESTS, nest_flat

    checkpoint, teacher = load_teacher(env["TEACHER_CHECKPOINT"])
    base_path = find_base_config(checkpoint, env.get("BASE_TRAIN_CONFIG"))
    model = Path(env["MODEL"]).expanduser().resolve(strict=True)
    dataset = Path(env["DATASET"]).expanduser().resolve(strict=True)
    output = Path(env["OUTPUT_DIR"]).expanduser().resolve()
    if output.exists():
        raise ValueError(f"OUTPUT_DIR already exists; choose a fresh path: {output}")
    block = teacher["block_size"]
    if env.get("BLOCK_SIZE") and int(env["BLOCK_SIZE"]) != block:
        raise ValueError("BLOCK_SIZE must equal the teacher block size; do not edit checkpoint metadata")
    steps, interval = int(env.get("PILOT_STEPS", "2000")), int(env.get("EVAL_EVERY", "500"))
    val_batches = int(env.get("VAL_BATCHES", "32"))
    head_lr = float(env.get("HEAD_LR", "0.0003"))
    if min(steps, interval, val_batches) < 1 or not math.isfinite(head_lr) or head_lr <= 0:
        raise ValueError("Positive step counts, validation batches, and finite head LR are required")
    if interval > steps:
        raise ValueError("EVAL_EVERY must not exceed PILOT_STEPS")
    server, trainer = allocation(env["VLLM_NPUS"], env["TRAIN_NPUS"], int(env["VLLM_DP"]), int(env["NUM_TRAIN_NPUS"]))
    ports = [int(env[name]) for name in ("VLLM_PORT", "VLLM_RPC_PORT")]
    if any(not 1 <= port <= 65535 for port in ports) or ports[0] == ports[1]:
        raise ValueError("VLLM_PORT and VLLM_RPC_PORT must be distinct ports in [1, 65535]")
    target_cfg = json.loads((model / "config.json").read_text())
    target_cfg = target_cfg.get("text_config", target_cfg)
    transformer = teacher["transformer_layer_config"]
    for key in ("vocab_size", "hidden_size"):
        if target_cfg.get(key) != transformer.get(key):
            raise ValueError(f"Teacher/target {key} mismatch")
    if max(teacher["aux_hidden_state_layer_ids"]) > target_cfg["num_hidden_layers"]:
        raise ValueError("Teacher target-layer IDs exceed the selected target model")
    base = TrainConfig.resolve(["--config", str(base_path), "--verifier-name-or-path", str(model), "--data-path", str(dataset)]).flatten()
    if base["hidden_states_backend"] != "file":
        raise ValueError("The pilot requires the existing file hidden-state backend")
    # Checkpoint serialization does not retain the private attention-backend
    # setting. Resolve it explicitly for this A2 pilot and pass it to the builder.
    draft_attn_impl = env.get("DRAFT_ATTN_IMPL", "sdpa")
    if draft_attn_impl not in ("sdpa", "eager"):
        raise ValueError("DRAFT_ATTN_IMPL must be sdpa or eager for the A2 pilot")
    cfg = copy.deepcopy(base)
    cfg.update(speculator_type="dflash_prefix", verifier_name_or_path=str(model), data_path=str(dataset),
               from_pretrained="", draft_config="", prefix_backbone_init=None,
               draft_attn_impl=draft_attn_impl,
               prefix_freeze_backbone=True, prefix_detach_backbone=True,
               prefix_loss_kind="target_ce", prefix_walk_backend="torch", prefix_loss_alpha=1.0,
               block_size=block, prefix_top_k=teacher["prefix_top_k"], prefix_rank=teacher["prefix_rank"],
               target_layer_ids=teacher["aux_hidden_state_layer_ids"], mask_token_id=teacher["mask_token_id"],
               sample_from_anchor=False, save_path=str(output / "checkpoints"),
               hidden_states_path=str(output / "hidden_states"), vllm_endpoint=f"http://127.0.0.1:{env['VLLM_PORT']}/v1",
               epochs=math.ceil(steps / interval), max_steps=None, checkpoint_freq=1.0,
               save_best=False, no_resume_from_checkpoint=True, fsdp_shard=False, dry_run=False,
               optimizer="adamw", lr=head_lr, weight_decay=.01,
               scheduler_type="cosine", scheduler_total_steps=steps,
               scheduler_warmup_steps=min(100, max(1, steps // 10)), scheduler_warmup_ratio=None,
               gradient_checkpointing=False, max_anchors=int(env.get("MAX_ANCHORS", "128")),
               num_workers=int(env.get("NUM_WORKERS", "4")), log_freq=20,
               run_name=f"compiled_prefix_pilot_bs{block}", log_dir=str(output / "logs/tensorboard"))
    settings = dict(teacher_checkpoint=str(checkpoint), target_path=str(model),
                    block_size=block, top_k=teacher["prefix_top_k"],
                    head_lr=head_lr, head_weight_decay=.01, pilot_steps=steps, eval_every=interval,
                    val_batches=val_batches, seed=cfg["seed"], draft_attn_impl=draft_attn_impl,
                    cap=float(env.get("COMPILED_LOGIT_CAP", "8")),
                    kd_weight=float(env.get("KD_WEIGHT", "0.5")),
                    history_weight=float(env.get("HISTORY_WEIGHT", "0.1")),
                    intervention_fraction=float(env.get("INTERVENTION_FRACTION", "0.25")),
                    anchor_chunk_size=int(env.get("ANCHOR_CHUNK_SIZE", "64")))
    for key in ("cap", "kd_weight", "history_weight", "intervention_fraction"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < settings["cap"] <= 8 or not 0 <= settings["intervention_fraction"] <= 1 or not 1 <= settings["anchor_chunk_size"] <= 64:
        raise ValueError("Invalid compiled cap, intervention fraction, or anchor chunk size")
    resolved = TrainConfig.from_flat(cfg)
    nested = nest_flat({k: v for k, v in resolved.flatten().items() if k in CONFIG_DESTS})
    nested["backend"] = dict(resolved.backend_args)
    output.mkdir(parents=True)
    for directory in ("logs", "provenance", "hidden_states"):
        (output / directory).mkdir()
    path = output / "train_config.yaml"
    path.write_text(yaml.safe_dump({"train": nested}, sort_keys=False))
    reloaded = TrainConfig.resolve(["--config", str(path)]).flatten()
    if resolved.flatten() != reloaded:
        raise ValueError("Prepared training config did not round-trip")
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "target_layers.txt").write_text("\n".join(map(str, teacher["aux_hidden_state_layer_ids"])) + "\n")
    (output / "mask_token_id.txt").write_text(str(teacher["mask_token_id"]) + "\n")
    (output / "server_max_model_len.txt").write_text(str(int(reloaded["total_seq_len"]) + 256) + "\n")
    (output / "block_size.txt").write_text(str(block) + "\n")
    (output / "experiment.json").write_text(json.dumps({
        "initialization": "Trained Prefix teacher backbone and local term copied and frozen; new compiled attention trained from scratch",
        "teacher_checkpoint": str(checkpoint), "teacher_config_sha256": file_hash(checkpoint / "config.json"),
        "teacher_weight_sha256": {p.name: file_hash(p) for p in sorted(checkpoint.glob("*.safetensors"))},
        "base_training_config": str(base_path), "base_config_sha256": file_hash(base_path),
        "training_windows": "Trainer epoch is an evaluation window, not a complete dataset epoch",
        "expected_checkpoint_steps": {str(i): min((i + 1) * interval, steps) for i in range(math.ceil(steps / interval))},
        "server_npus": server, "trainer_npus": trainer, "settings": settings, "train_config": reloaded,
        "note": "Research pilot. No measured A2 speedup or acceptance guarantee."}, indent=2) + "\n")
    print(f"Prepared compiled-prefix PILOT: {path}")
    print(f"Teacher: {checkpoint}; block={block}; proposals={block - 1}; target layers={teacher['aux_hidden_state_layer_ids']}")
    print(f"Frozen backbone/local term; {steps} head-training steps; evaluate/save every {interval} steps")
    return path


if __name__ == "__main__":
    prepare(os.environ)
