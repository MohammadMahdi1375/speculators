#!/usr/bin/env python3
"""Resume an explicitly selected joint checkpoint in a new run directory."""
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil

import yaml
from speculators.train.config import TrainConfig
from speculators.train.config.schema import CONFIG_DESTS, nest_flat


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def locate_source(checkpoint, explicit=None):
    required = ("joint_settings.json", "train_config.yaml", "resolved_schedule.json")
    candidates = [Path(explicit).expanduser().resolve(strict=True)] if explicit else checkpoint.parents
    for directory in candidates:
        if all((directory / name).is_file() for name in required):
            return directory
    raise ValueError("Cannot find this checkpoint's joint run metadata. Set RESUME_RUN_DIR to its original run directory.")


def scheduler_spec(source, checkpoint, flat, settings, schedule, seen=None):
    context = checkpoint / "resume_context.json"
    if context.is_file():
        return json_read(context)["scheduler_spec"]
    seen = set() if seen is None else seen
    if source in seen:
        raise ValueError("Cyclic continuation provenance")
    seen.add(source)
    old = source / "continuation.json"
    if old.is_file():
        prior = json_read(old)
        state = json_read(checkpoint / "training_state.json")
        # A retained input checkpoint still belongs to its original LR schedule.
        if state["global_step"] == prior["start_global_step"]:
            origin = Path(prior["source_checkpoint"]).resolve(strict=True)
            root = locate_source(origin, prior["source_run"])
            return scheduler_spec(root, origin,
                TrainConfig.resolve(["--config", str(root / "train_config.yaml")]).flatten(),
                json_read(root / "joint_settings.json"), json_read(root / "resolved_schedule.json"), seen)
        if "scheduler_spec" in schedule:
            return schedule["scheduler_spec"]
        # Backwards compatibility with the previously installed v1 continuation.
        return dict(kind="restart_cosine", origin_step=prior["start_global_step"],
                    total_steps=schedule["additional_steps"], warmup_steps=schedule["warmup_steps"],
                    planned_epochs=settings["epochs"])
    total = flat["scheduler_total_steps"] or schedule["planned_optimizer_steps"]
    warmup = flat["scheduler_warmup_steps"]
    if warmup is None:
        warmup = int(total * flat["scheduler_warmup_ratio"]) if flat["scheduler_warmup_ratio"] is not None else total // 100
    if flat["scheduler_type"] not in ("linear", "cosine"):
        raise ValueError("The joint trainer requires a linear or cosine backbone scheduler")
    return dict(kind="joint_scratch", origin_step=0, planned_epochs=settings["epochs"],
        backbone_kind=flat["scheduler_type"], total_steps=total, warmup_steps=warmup,
        num_cycles=flat["scheduler_num_cosine_cycles"], head_total_steps=schedule["planned_optimizer_steps"],
        head_delay=settings["delay_steps"], head_warmup=settings["head_warmup_steps"])


def source_world(source, rank_dirs):
    command = source / "checkpoints/train_command.txt"
    if command.is_file():
        match = re.search(r"^# World size: (\d+)\s*$", command.read_text(), re.M)
        if match:
            return int(match[1])
    schedule = json_read(source / "resolved_schedule.json")
    if schedule.get("trainer_world_size"):
        return int(schedule["trainer_world_size"])
    if rank_dirs and {p.name for p in rank_dirs} == {f"rank_{i}" for i in range(len(rank_dirs))}:
        return len(rank_dirs)
    raise ValueError("Cannot verify original trainer world size: missing train_command.txt and cached validation ranks")


def prepare(env):
    from prepare_experiment import allocation
    allocation(env)
    if not env.get("RESUME_CHECKPOINT"):
        raise ValueError("Use --resume-training CHECKPOINT; no checkpoint is chosen automatically")
    checkpoint = Path(env["RESUME_CHECKPOINT"]).expanduser().resolve(strict=True)
    source = locate_source(checkpoint, env.get("RESUME_RUN_DIR"))
    output = Path(env["OUTPUT_DIR"]).expanduser().resolve()
    if any(output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in (source, checkpoint)):
        raise ValueError("Use a separate OUTPUT_DIR outside the source run and checkpoint")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"OUTPUT_DIR must be fresh: {output}")
    check = env.get("CHECK_ONLY", "0")
    if check not in ("0", "1"):
        raise ValueError("CHECK_ONLY must be 0 or 1")
    settings = json_read(source / "joint_settings.json")
    schedule = json_read(source / "resolved_schedule.json")
    flat = TrainConfig.resolve(["--config", str(source / "train_config.yaml")]).flatten()
    if (settings.get("selector_kind") != "compiled_prefix_v3"
            or flat["speculator_type"] != "dflash_prefix" or flat["fsdp_shard"]
            or flat["max_steps"] is not None or flat["hidden_states_backend"] != "file"):
        raise ValueError("Expected a file-backed joint compiled-prefix DDP training checkpoint")
    if any(flat.get(k) for k in ("from_pretrained", "draft_config", "prefix_backbone_init")):
        raise ValueError("Source architecture has additional checkpoint initialization paths")
    required = ("config.json", "model.safetensors", "optimizer_state_dict.pt",
                "scheduler_state_dict.pt", "training_state.json")
    for name in required:
        if not (checkpoint / name).is_file():
            raise ValueError(f"Full training checkpoint missing: {checkpoint / name}. "
                "Weights-only exports and legacy interrupted/ checkpoints without training_state.json cannot be resumed; select a saved epoch/step checkpoint.")
    state = json_read(checkpoint / "training_state.json")
    for key in ("epoch", "local_step", "global_step"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError(f"Invalid checkpoint training progress: {key}")
    last, skip = state["epoch"], state["local_step"]
    if last >= settings["epochs"] or state["global_step"] < skip or state["global_step"] < 1:
        raise ValueError("Checkpoint progress is inconsistent with the source run")
    # For a partial epoch, finishing it counts as the first requested epoch.
    completed = last + (skip == 0)
    steps_per_epoch = int(schedule["steps_per_epoch"])
    cfg = json_read(checkpoint / "config.json")
    for key, value in dict(prefix_selector_kind="compiled_prefix_v3", block_size=settings["block_size"],
                           prefix_rank=settings["rank"], prefix_top_k=settings["top_k"],
                           prefix_attention_heads=settings["heads"]).items():
        if cfg.get(key) != value:
            raise ValueError(f"Source checkpoint/settings mismatch: {key}")
    if cfg.get("architectures") != ["DFlashPrefixDraftModel"]:
        raise ValueError("Source checkpoint is not an exported DFlashPrefixDraftModel")
    rank_dirs = sorted((source / "validation_inputs").glob("rank_*"))
    world = int(env["NUM_TRAIN_NPUS"])
    original_world = source_world(source, rank_dirs)
    if original_world != world:
        raise ValueError(f"Keep the original trainer world size ({original_world}); requested {world}")
    manifests = {}
    expected_ranks = {f"rank_{i}" for i in range(world)}
    if {p.name for p in rank_dirs} == expected_ranks and all((p / "manifest.json").is_file() for p in rank_dirs):
        for directory in rank_dirs:
            manifest = json_read(directory / "manifest.json")
            if not manifest:
                raise ValueError(f"Empty validation cohort: {directory}")
            for item in manifest:
                path = directory / item["file"]
                if Path(item["file"]).name != item["file"] or not path.is_file() or path.stat().st_size != item["bytes"]:
                    raise ValueError(f"Incomplete validation cohort: {directory}")
            manifests[directory.name] = digest(directory / "manifest.json")
    validation_mode = "reuse_fixed" if manifests else "create_fixed"
    try:
        total_epochs = int(env["EPOCHS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EPOCHS must be a positive integer total epoch target") from exc
    if total_epochs < 1:
        raise ValueError("EPOCHS must be a positive integer total epoch target")
    if total_epochs <= completed:
        raise ValueError(f"EPOCHS={total_epochs} leaves no training to run: the checkpoint has completed {completed} epochs. "
                         f"Set EPOCHS above {completed}; for three more full epochs use EPOCHS={completed + 3}.")
    remaining = total_epochs - completed
    prior_spec = scheduler_spec(source, checkpoint, flat, settings, schedule)
    policy = env.get("RESUME_LR_POLICY", "auto")
    if policy not in ("auto", "restore", "restart"):
        raise ValueError("RESUME_LR_POLICY must be auto, restore, or restart")
    within_plan = total_epochs <= prior_spec["planned_epochs"] and state["global_step"] < prior_spec["origin_step"] + prior_spec["total_steps"]
    selected_policy = ("restore" if within_plan else "restart") if policy == "auto" else policy
    if selected_policy == "restore" and not within_plan:
        raise ValueError("Requested epochs extend beyond the saved LR plan; use RESUME_LR_POLICY=restart or auto")
    scale = positive(env.get("RESUME_LR_SCALE", env.get("CONTINUE_LR_SCALE", ".1")), "RESUME_LR_SCALE")
    warmup_ratio = float(env.get("RESUME_WARMUP_RATIO", env.get("CONTINUE_WARMUP_RATIO", ".01")))
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError("RESUME_WARMUP_RATIO must be in [0, 1)")
    overrides = {key: env.get("RESUME_" + key) or env.get("CONTINUE_" + key) for key in ("BACKBONE_LR", "MUON_LR", "HEAD_LR")}
    if selected_policy == "restore" and any(overrides.values()):
        raise ValueError("LR overrides require RESUME_LR_POLICY=restart")
    # Model/data identity is inherited, not silently replaced by shell defaults.
    target = Path(flat["verifier_name_or_path"]).resolve(strict=True)
    data = Path(flat["data_path"]).resolve(strict=True)
    new_flat = copy.deepcopy(flat)
    new_settings = copy.deepcopy(settings)
    new_flat.update(epochs=total_epochs, no_resume_from_checkpoint=False,
        save_path=str(output / "checkpoints"), hidden_states_path=str(output / "hidden_states"),
        vllm_endpoint=f"http://127.0.0.1:{env['VLLM_PORT']}/v1", checkpoint_freq=1., save_best=False,
        dry_run=False,
        run_name=f"joint_continue_epoch{completed + 1}_to_{total_epochs}",
        log_dir=str(output / "logs/tensorboard"))
    new_settings.update(output_dir=str(output), epochs=total_epochs)
    if selected_policy == "restart":
        new_flat.update(scheduler_type="cosine", scheduler_total_steps=remaining * steps_per_epoch,
            scheduler_warmup_steps=int(remaining * steps_per_epoch * warmup_ratio), scheduler_warmup_ratio=None)
        new_flat["lr"] = positive(overrides["BACKBONE_LR"] or flat["lr"] * scale, "RESUME_BACKBONE_LR")
        if flat["optimizer"] == "muon":
            original_muon = flat.get("muon_lr") or 10 * flat["lr"]
            new_flat["muon_lr"] = positive(overrides["MUON_LR"] or original_muon * scale, "RESUME_MUON_LR")
        new_settings["head_lr"] = positive(overrides["HEAD_LR"] or settings["head_lr"] * scale, "RESUME_HEAD_LR")
    record = dict(version=3, source_run=str(source), source_checkpoint=str(checkpoint),
        requested_checkpoint=env["RESUME_CHECKPOINT"], source_epoch_index=last,
        completed_epochs=completed, start_epoch_index=completed, resume_local_step=skip,
        remaining_epoch_completions=remaining, total_epochs=total_epochs, start_global_step=state["global_step"],
        steps_per_epoch=steps_per_epoch, trainer_world_size=world, lr_scale=scale, warmup_ratio=warmup_ratio,
        lr_policy=selected_policy, requested_lr_policy=policy, source_scheduler_spec=prior_spec,
        epoch_semantics="EPOCHS is the total epoch target, identical to scratch training",
        peak_lrs=dict(backbone_adamw=new_flat["lr"], muon=new_flat["muon_lr"], selector_adamw=new_settings["head_lr"]),
        validation_mode=validation_mode, validation_manifests=manifests, staged=check == "0",
        baseline_checkpoint="resume_source" if skip else last,
        limitations=["Source exports round weights and floating optimizer state to BF16",
                     "Source per-rank RNG and guard replay were not saved; replay is repopulated",
                     "Resume is not bit-exact to uninterrupted training"])
    # Store source metadata; confirm the installed training model matches the saved architecture at startup.
    record["source_metadata_sha256"] = {name: digest(source / name) for name in
        ("train_config.yaml", "joint_settings.json", "resolved_schedule.json")}
    resolved = TrainConfig.from_flat(new_flat)
    nested = nest_flat({k: v for k, v in resolved.flatten().items() if k in CONFIG_DESTS})
    for name in ("dflash2", "dspark", "peagle", "mtp"):
        nested.pop(name, None)
    nested["backend"] = dict(resolved.backend_args)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "train_config.yaml"
    path.write_text(yaml.safe_dump({"train": nested}, sort_keys=False))
    if TrainConfig.resolve(["--config", str(path)]).flatten() != resolved.flatten():
        raise ValueError("Continuation configuration did not round-trip")
    (output / "joint_settings.json").write_text(json.dumps(new_settings, indent=2) + "\n")
    for name, value in {"model_path.txt": target, "dataset_path.txt": data,
        "target_layers.txt": " ".join(map(str, flat["target_layer_ids"])),
        "mask_token_id.txt": 151669 if flat["mask_token_id"] is None else flat["mask_token_id"],
        "server_max_model_len.txt": flat["total_seq_len"] + 256}.items():
        (output / name).write_text(str(value) + "\n")
    if check == "0":
        destination = output / "checkpoints" / str(last)
        destination.mkdir(parents=True)
        record["checkpoint_sha256"] = {}
        copied_names = (*required, *(("resume_context.json",) if (checkpoint / "resume_context.json").is_file() else ()))
        initial_stats = {name: (checkpoint / name).stat() for name in copied_names}
        for name in copied_names:
            print(f"Copying source checkpoint file: {name}", flush=True)
            src, dst = checkpoint / name, destination / name
            before = src.stat()
            shutil.copy2(src, dst)
            # Both hashes are compared; mutation during staging is a hard error.
            source_digest, copied_digest = digest(src), digest(dst)
            if (before.st_size != src.stat().st_size or before.st_mtime_ns != src.stat().st_mtime_ns
                    or source_digest != copied_digest):
                raise ValueError(f"Source checkpoint changed while staging: {src}")
            record["checkpoint_sha256"][name] = copied_digest
        for name, before in initial_stats.items():
            after = (checkpoint / name).stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f"Source checkpoint changed during staging: {checkpoint / name}")
        if json_read(destination / "training_state.json") != state or json_read(destination / "config.json") != cfg:
            raise ValueError("Source checkpoint changed after preparation began")
        (destination / "resume_context.json").write_text(json.dumps(dict(version=1,
            scheduler_spec=prior_spec, trainer_world_size=world), indent=2) + "\n")
        if skip:
            # The numeric directory is overwritten when this partial epoch finishes.
            # Retain a full resumable baseline for checkpoint_best selection.
            baseline = output / "checkpoints/resume_source"
            baseline.mkdir()
            for name in (*required, "resume_context.json"):
                shutil.copy2(destination / name, baseline / name)
        if manifests:
            # Large fixed target batches are read-only; do not duplicate them.
            (output / "validation_inputs").symlink_to(source / "validation_inputs", target_is_directory=True)
    (output / "continuation.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "experiment.json").write_text(json.dumps(dict(initialization="Explicit joint checkpoint + saved optimizer/training state",
        train_config=resolved.flatten(), settings=new_settings, continuation=record), indent=2) + "\n")
    print(f"JOINT_RESUME_CONFIG {path}")
    print(f"Selected checkpoint: {checkpoint}")
    print(f"Epoch indices to run: {completed} through {total_epochs - 1}; total EPOCHS={total_epochs}; skipped batches={skip}")
    print(f"Restore global_step={state['global_step']} and optimizer state; LR policy={selected_policy}")
    if selected_policy == "restart":
        print(f"New cosine peak LRs: {record['peak_lrs']}; warmup ratio={warmup_ratio}")
    else:
        print("Restore saved schedulers, LR values, selector delay and original LR horizon")
    print("Source model/data/batch/recomputation/guard settings retained. Source checkpoint is unchanged.")
    print("Old BF16 exports do not contain original FP32 precision, RNG or guard replay; resume is not bit-exact.")
    return path


if __name__ == "__main__":
    prepare(os.environ)
