#!/usr/bin/env python3
"""Prepare a fresh five-epoch run; never open a trained draft checkpoint."""
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml
from speculators.train.config import TrainConfig
from speculators.train.config.schema import CONFIG_DESTS, nest_flat


def write_config(base_path, output, *, model, dataset, port=8092, epochs=5,
                 anchors=512, workers=4, rank=256, top_k=16, heads=4, layers=2,
                 head_lr=3e-4, loss_alpha=.25, delay_steps=2000, head_warmup_steps=2000,
                 block_size=8):
    base_path, output = Path(base_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a fresh OUTPUT_DIR: {output}")
    base = TrainConfig.resolve(["--config", str(base_path)]).flatten()
    if base["hidden_states_backend"] != "file":
        raise ValueError("Expected the existing file-backed hidden-state setup")
    if base["from_pretrained"] or base["draft_config"]:
        raise ValueError("BASE_TRAIN_CONFIG must describe scratch architecture directly; use the original vanilla run.yaml")
    for key, value in (("verifier_name_or_path", model), ("data_path", dataset)):
        if Path(base[key]).resolve() != Path(value).resolve():
            raise ValueError(f"Original run's {key} differs from the requested experiment")
    if base["total_seq_len"] != 3072:
        raise ValueError("Expected the existing sequence-3072 training recipe")
    if type(block_size) is not int or not 2 <= block_size <= base["total_seq_len"]:
        raise ValueError("BLOCK_SIZE must be an integer from 2 through total_seq_len (anchor included)")
    if base["optimizer"] not in {"adamw", "muon"}:
        raise ValueError("Supported backbone optimizers: AdamW or Muon")
    if epochs < 1 or anchors < 1 or workers < 0:
        raise ValueError("Invalid epochs, anchors, or workers")
    if rank < 8 or heads < 1 or rank % heads or layers < 1 or top_k < 2:
        raise ValueError("Invalid selector dimensions")
    if head_lr <= 0 or loss_alpha <= 0 or delay_steps < 0 or head_warmup_steps < 1:
        raise ValueError("Invalid selector optimization settings")
    cfg = copy.deepcopy(base)
    cfg.update(
        speculator_type="dflash_prefix", from_pretrained="", draft_config="",
        block_size=block_size,
        prefix_backbone_init=None, prefix_freeze_backbone=False, prefix_detach_backbone=False,
        prefix_rank=rank, prefix_top_k=top_k, prefix_walk_backend="torch",
        prefix_loss_kind="target_ce", prefix_loss_alpha=loss_alpha,
        verifier_name_or_path=str(Path(model).resolve()), data_path=str(Path(dataset).resolve()),
        vllm_endpoint=f"http://127.0.0.1:{port}/v1",
        save_path=str(output / "checkpoints"), hidden_states_path=str(output / "hidden_states"),
        epochs=epochs, max_steps=None, checkpoint_freq=1.0, save_best=False,
        scheduler_type="cosine", scheduler_total_steps=None,
        scheduler_warmup_steps=None, scheduler_warmup_ratio=.02,
        no_resume_from_checkpoint=True, dry_run=False, fsdp_shard=False,
        gradient_checkpointing=True, num_workers=workers, max_anchors=anchors,
        log_freq=20, run_name=f"dflash_prefix_tau_v2_scratch_bs{block_size}",
        log_dir=str(output / "logs/tensorboard"),
    )
    resolved = TrainConfig.from_flat(cfg)
    nested = nest_flat({k: v for k, v in resolved.flatten().items() if k in CONFIG_DESTS})
    for name in ("dflash2", "dspark", "peagle", "mtp"):
        nested.pop(name, None)
    nested["backend"] = dict(resolved.backend_args)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    config_path = output / "train_config.yaml"
    config_path.write_text(yaml.safe_dump({"train": nested}, sort_keys=False))
    reloaded = TrainConfig.resolve(["--config", str(config_path)]).flatten()
    for key, value in resolved.flatten().items():
        if value != reloaded[key]:
            raise ValueError(f"Training config did not round-trip: {key}")
    settings = {
        "block_size": block_size, "num_speculative_tokens": block_size - 1,
        "selector_kind": "local_prefix_v2", "rank": rank, "top_k": top_k,
        "heads": heads, "layers": layers, "head_lr": head_lr,
        "head_weight_decay": .01, "loss_alpha": loss_alpha,
        "delay_steps": delay_steps, "head_warmup_steps": head_warmup_steps,
    }
    (output / "selector_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "experiment.json").write_text(json.dumps({
        "initialization": "Random draft backbone and new selector; no trained draft loaded",
        "base_training_config": str(base_path),
        "base_training_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "base_block_size": base["block_size"],
        "block_size": block_size, "num_speculative_tokens": block_size - 1,
        "epochs_are": "Full passes through the original training split; no batch limit",
        "pretrained_weights_used": "Frozen Qwen3 target embedding, output head and normalization only",
        "train_config": reloaded, "selector": settings,
    }, indent=2) + "\n")
    print(f"Scratch configuration: {config_path}", flush=True)
    print(f"BLOCK_SIZE={block_size}: 1 anchor + {block_size - 1} proposals; "
          f"shortlist top-K={top_k}; base recipe block_size={base['block_size']}", flush=True)
    print(f"Full epochs: {epochs}; backbone optimizer: {cfg['optimizer']}; "
          f"backbone AdamW LR: {cfg['lr']}; Muon LR: {cfg['muon_lr']}", flush=True)
    print("Trained draft checkpoint: NONE; resume: disabled", flush=True)
    return config_path


if __name__ == "__main__":
    write_config(
        os.environ["BASE_TRAIN_CONFIG"], os.environ["OUTPUT_DIR"],
        model=os.environ["MODEL"], dataset=os.environ["DATASET"],
        port=int(os.environ["VLLM_PORT"]), epochs=int(os.environ["EPOCHS"]),
        anchors=int(os.environ["MAX_ANCHORS"]), workers=int(os.environ["NUM_WORKERS"]),
        rank=int(os.environ["PREFIX_RANK"]), top_k=int(os.environ["PREFIX_TOP_K"]),
        heads=int(os.environ["PREFIX_HEADS"]), layers=int(os.environ["PREFIX_LAYERS"]),
        head_lr=float(os.environ["HEAD_LR"]), loss_alpha=float(os.environ["PREFIX_LOSS_ALPHA"]),
        delay_steps=int(os.environ["SELECTOR_DELAY_STEPS"]),
        head_warmup_steps=int(os.environ["SELECTOR_WARMUP_STEPS"]),
        block_size=int(os.environ.get("BLOCK_SIZE", "8")),
    )
