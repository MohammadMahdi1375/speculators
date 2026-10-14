#!/usr/bin/env python3
"""Prepare full-epoch compiled scratch training; no draft checkpoint is loaded."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import yaml
from speculators.train.config import TrainConfig
from speculators.train.config.schema import CONFIG_DESTS, nest_flat


def integer(env, name, default, minimum=0):
    value = int(env.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def flag(env, name, default):
    value = str(env.get(name, int(default)))
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def number(env, name, default, *, low=0., high=float("inf"), positive=False):
    value = float(env.get(name, default))
    if not math.isfinite(value) or value < low or value > high or (positive and value <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def allocation(env):
    def devices(name):
        values = env[name].split(",")
        if not values or any(not x.isdecimal() for x in values) or len(set(values)) != len(values):
            raise ValueError(f"{name} must be comma-separated distinct numeric NPU IDs")
        return values
    server, train = devices("VLLM_NPUS"), devices("TRAIN_NPUS")
    if set(server) & set(train):
        raise ValueError("Server/training NPU assignments overlap")
    if len(server) != int(env["VLLM_DP"]) or len(train) != int(env["NUM_TRAIN_NPUS"]):
        raise ValueError("NPU counts differ from VLLM_DP/NUM_TRAIN_NPUS")
    ports = [int(env[k]) for k in ("VLLM_PORT", "VLLM_RPC_PORT")]
    if any(not 1 <= p <= 65535 for p in ports) or len(set(ports)) != 2:
        raise ValueError("Choose distinct valid server and RPC ports")


def prepare(env):
    allocation(env)
    output = Path(env["OUTPUT_DIR"]).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"OUTPUT_DIR must be fresh: {output}")
    recipe = Path(env["BASE_TRAIN_CONFIG"]).expanduser().resolve(strict=True)
    target = Path(env["MODEL"]).expanduser().resolve(strict=True)
    data = Path(env["DATASET"]).expanduser().resolve(strict=True)
    target_cfg = json.loads((target / "config.json").read_text())
    target_cfg = target_cfg.get("text_config", target_cfg)
    # Explicit overrides handle relocation; the old recipe need not name an
    # existing dataset/model on this server.
    base = TrainConfig.resolve(["--config", str(recipe), "--verifier-name-or-path", str(target),
                                "--data-path", str(data)]).flatten()
    if base["speculator_type"] not in {"dflash", "dflash_prefix"}:
        raise ValueError("BASE_TRAIN_CONFIG must be a DFlash-family scratch recipe")
    if any(base.get(k) for k in ("from_pretrained", "draft_config", "prefix_backbone_init")):
        raise ValueError("BASE_TRAIN_CONFIG must describe a scratch architecture, without draft weight/config paths")
    if base["hidden_states_backend"] != "file" or base["optimizer"] not in {"muon", "adamw"}:
        raise ValueError("Use the existing file-backed Muon/AdamW scratch recipe")
    if base.get("sample_from_anchor") or any(base.get(k) for k in ("token_freq_path", "d2t_path", "t2d_path")):
        raise ValueError("Compiled selector requires an anchor and full target vocabulary")
    if base.get("draft_vocab_size") not in (None, 0, target_cfg["vocab_size"]):
        raise ValueError("Reduced-vocabulary recipes are unsupported")
    block = integer(env, "BLOCK_SIZE", 16, 2)
    packed_tokens = integer(env, "TOTAL_SEQ_LEN", base["total_seq_len"], block) if env.get("TOTAL_SEQ_LEN") else base["total_seq_len"]
    if block not in (8, 16) or block > packed_tokens:
        raise ValueError("Use BLOCK_SIZE=8 or 16, matching the installed serving path")
    if integer(env, "PREFIX_TOP_K", 16, 2) != 16:
        raise ValueError("This matched serving experiment requires PREFIX_TOP_K=16")
    if integer(env, "PREFIX_LAYERS", 1, 1) != 1:
        raise ValueError("compiled_prefix_v3 implements one compiled history layer")
    settings = dict(
        version=2, output_dir=str(output), selector_kind="compiled_prefix_v3", block_size=block,
        num_speculative_tokens=block-1, top_k=16,
        epochs=integer(env, "EPOCHS", 5, 1), rank=integer(env, "PREFIX_RANK", 256, 8),
        heads=integer(env, "PREFIX_HEADS", 4, 1), layers=1,
        compiled_logit_cap=number(env, "COMPILED_LOGIT_CAP", 8., high=8., positive=True),
        head_lr=number(env, "HEAD_LR", .0003, positive=True),
        head_weight_decay=number(env, "HEAD_WEIGHT_DECAY", .01),
        loss_alpha=number(env, "PREFIX_LOSS_ALPHA", .25, positive=True),
        delay_steps=integer(env, "SELECTOR_DELAY_STEPS", 2000),
        head_warmup_steps=integer(env, "SELECTOR_WARMUP_STEPS", 2000, 1),
        focus_start_steps=integer(env, "PREFIX_FOCUS_START_STEPS", 4000),
        focus_warmup_steps=integer(env, "PREFIX_FOCUS_WARMUP_STEPS", 4000, 1),
        dense_floor=number(env, "PREFIX_DENSE_FLOOR", .25, high=1., positive=True),
        anchor_chunk_size=integer(env, "ANCHOR_CHUNK_SIZE", 8, 1),
        backbone_checkpointing=flag(env, "BACKBONE_CHECKPOINTING", True),
        selector_checkpointing=flag(env, "SELECTOR_CHECKPOINTING", True),
        packed_tokens_per_rank=packed_tokens,
        memory_log_every=integer(env, "MEMORY_LOG_EVERY", 100),
        val_batches=integer(env, "VAL_BATCHES", 32, 1),
        guard_mode=env.get("GUARD_MODE", "project"),
        guard_start_steps=integer(env, "GUARD_START_STEPS", 4000),
        guard_every=integer(env, "GUARD_EVERY", 1, 1),
        guard_replay_anchors=integer(env, "GUARD_REPLAY_ANCHORS", 32, 1),
        guard_add_anchors=integer(env, "GUARD_ADD_ANCHORS", 4, 1),
        guard_chunk_size=integer(env, "GUARD_CHUNK_SIZE", 4, 1),
        guard_backtracks=integer(env, "GUARD_BACKTRACKS", 3),
    )
    if settings["rank"] % settings["heads"]:
        raise ValueError("PREFIX_RANK must be divisible by PREFIX_HEADS")
    if settings["guard_mode"] not in {"off", "project"}:
        raise ValueError("GUARD_MODE must be off or project")
    if settings["guard_add_anchors"] > settings["guard_replay_anchors"]:
        raise ValueError("GUARD_ADD_ANCHORS exceeds replay capacity")
    if settings["guard_mode"] == "project" and settings["guard_start_steps"] < settings["delay_steps"] + settings["head_warmup_steps"]:
        raise ValueError("Activate the guard after selector delay and warmup")
    layers = base["target_layer_ids"] or [1, 9, 17, 25, 33]
    if any(i < 0 or i > target_cfg["num_hidden_layers"] for i in layers):
        raise ValueError("Recipe target layers are outside the selected target")
    cfg = copy.deepcopy(base)
    cfg.update(speculator_type="dflash_prefix", from_pretrained="", draft_config="",
        prefix_backbone_init=None, prefix_freeze_backbone=False, prefix_detach_backbone=False,
        draft_vocab_size=None, embed_requires_grad=False,
        sample_from_anchor=False, prefix_rank=settings["rank"], prefix_top_k=16,
        prefix_loss_kind="target_ce", prefix_loss_alpha=settings["loss_alpha"], prefix_walk_backend="torch",
        block_size=block, verifier_name_or_path=str(target), data_path=str(data), target_layer_ids=layers,
        save_path=str(output / "checkpoints"), hidden_states_path=str(output / "hidden_states"),
        vllm_endpoint=f"http://127.0.0.1:{env['VLLM_PORT']}/v1", draft_attn_impl=env.get("DRAFT_ATTN_IMPL", "sdpa"),
        epochs=settings["epochs"], max_steps=None, checkpoint_freq=1., save_best=False,
        no_resume_from_checkpoint=True, dry_run=False, fsdp_shard=False,
        gradient_checkpointing=settings["backbone_checkpointing"], total_seq_len=packed_tokens,
        num_workers=integer(env, "NUM_WORKERS", 4), max_anchors=integer(env, "MAX_ANCHORS", 512, 1),
        prefetch_factor=integer(env, "PREFETCH_FACTOR", base["prefetch_factor"], 1) if env.get("PREFETCH_FACTOR") else base["prefetch_factor"],
        scheduler_type="cosine", scheduler_total_steps=None, scheduler_warmup_steps=None,
        scheduler_warmup_ratio=number(env, "BACKBONE_WARMUP_RATIO", .02, high=1),
        hidden_states_dtype="bfloat16", seed=integer(env, "SEED", base["seed"]),
        log_freq=integer(env, "LOG_FREQ", 20, 1), run_name=f"compiled_joint_bs{block}_{settings['guard_mode']}",
        log_dir=str(output / "logs/tensorboard"))
    if cfg["draft_attn_impl"] not in {"sdpa", "eager"}:
        raise ValueError("A2 scratch training requires DRAFT_ATTN_IMPL=sdpa or eager")
    if env.get("BACKBONE_LR"):
        cfg["lr"] = number(env, "BACKBONE_LR", 0, positive=True)
    if env.get("MUON_LR"):
        cfg["muon_lr"] = number(env, "MUON_LR", 0, positive=True)
    # Resolve the Muon learning rate explicitly, independently from AdamW.
    if cfg["optimizer"] == "muon" and cfg.get("muon_lr") is None:
        cfg["muon_lr"] = 10 * cfg["lr"]
    resolved = TrainConfig.from_flat(cfg)
    nested = nest_flat({k: v for k, v in resolved.flatten().items() if k in CONFIG_DESTS})
    for name in ("dflash2", "dspark", "peagle", "mtp"):
        nested.pop(name, None)
    nested["backend"] = dict(resolved.backend_args)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "train_config.yaml"
    path.write_text(yaml.safe_dump({"train": nested}, sort_keys=False))
    roundtrip = TrainConfig.resolve(["--config", str(path)]).flatten()
    if roundtrip != resolved.flatten():
        raise ValueError("Training config did not round-trip")
    (output / "joint_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "target_layers.txt").write_text(" ".join(map(str, layers)) + "\n")
    mask_token = cfg.get("mask_token_id")
    (output / "mask_token_id.txt").write_text(str(151669 if mask_token is None else mask_token) + "\n")
    (output / "server_max_model_len.txt").write_text(str(cfg["total_seq_len"] + 256) + "\n")
    (output / "experiment.json").write_text(json.dumps(dict(
        initialization="Random draft and complete compiled head; no trained draft/teacher checkpoint",
        frozen_target=str(target), source_recipe=str(recipe), source_recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        train_config=roundtrip, settings=settings, epochs_are="Full training split passes, without a batch cap",
        constraint_scope="Selector updates on detached rank-local training replay; no backbone/population guarantee",
        selection_metric="Saved-weight offline reference EAL; serving tau/throughput measured separately"), indent=2) + "\n")
    print(f"JOINT_SCRATCH_CONFIG {path}")
    print(f"Full epochs={settings['epochs']}; block={block}; proposals={block-1}; top-K=16; rank={settings['rank']}; heads={settings['heads']}")
    print(f"Backbone={cfg['optimizer']}; AdamW LR={cfg['lr']}; Muon LR={cfg['muon_lr']}; selector AdamW LR={settings['head_lr']}")
    print(f"Guard={settings['guard_mode']}; start={settings['guard_start_steps']}; every={settings['guard_every']}; draft checkpoint=NONE")
    print(f"Recomputation: backbone={settings['backbone_checkpointing']}; selector={settings['selector_checkpointing']}; selector chunk={settings['anchor_chunk_size']}; guard chunk={settings['guard_chunk_size']}")
    print(f"Packed token budget/rank={packed_tokens}; global budget={packed_tokens * int(env['NUM_TRAIN_NPUS'])}; max anchors/rank={cfg['max_anchors']}; workers/rank={cfg['num_workers']}; prefetch={cfg['prefetch_factor']}")
    if packed_tokens != base["total_seq_len"]:
        print("PACKED_BATCH_CHANGED: optimizer steps/epoch and data order change; revisit step-based warmup/guard schedules and validate model quality.")
    return path


if __name__ == "__main__":
    prepare(os.environ)
