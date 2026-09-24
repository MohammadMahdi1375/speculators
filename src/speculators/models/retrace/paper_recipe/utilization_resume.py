"""Shared resume identity and read-only checkpoint checks before server startup."""

import json
import math
from pathlib import Path

from .pretrained import sha256


def training_signature(args, world_size, total_steps, target_backend):
    defaults = {
        "lr_warmup_ratio": 0.05, "weight_decay": 0.01, "gamma": 4.0,
        "seed": 42, "dtype": "bfloat16",
    }
    fields = (
        "target", "prompt_length", "response_length", "global_prompt_batch",
        "epochs", "lr", "lr_warmup_ratio", "weight_decay", "gamma", "seed",
        "disable_conditioning", "dtype", "blocks_per_forward", "performance_mode",
        "trace_storage", "rollout_batch_size", "attention_backend", "target_norm",
        "work_distribution",
        "rollout_scheduler",
    )
    return {
        **{key: getattr(args, key, defaults.get(key)) for key in fields},
        "pool_sha256": sha256(args.pool), "world_size": world_size,
        "total_steps": total_steps, "target_backend": target_backend,
    }


def resolve_run(run):
    """Only use a completed checkpoint pointer, never a partial directory."""
    run = Path(run).resolve()
    pointer = run / "checkpoints/latest.txt"
    if not pointer.is_file() or not pointer.read_text().strip():
        raise ValueError(
            f"No completed checkpoint pointer at {pointer}. Unsaved updates cannot "
            "be recovered by this launcher; leave the old trainer running until it saves."
        )
    checkpoint = Path(pointer.read_text().strip())
    if not checkpoint.is_absolute():
        checkpoint = pointer.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(pointer.parent.resolve()):
        raise ValueError("The checkpoint pointer must refer inside its run's checkpoints directory")
    command_path = run / "checkpoints/train_command.json"
    if not command_path.is_file():
        raise ValueError(f"Missing saved training arguments: {command_path}")
    command = json.loads(command_path.read_text())
    pool = Path(command["args"]["pool"])
    if not pool.is_absolute():
        raise ValueError("The saved prompt pool must have an absolute path")
    return checkpoint, pool


def check_checkpoint(checkpoint, workers):
    checkpoint = Path(checkpoint)
    if ".partial" in checkpoint.name:
        raise ValueError("A partial checkpoint cannot be resumed")
    names = ["config.json", "training_state.json", "master_parameters.pt", "optimizer.pt"]
    names += [f"rng_rank_{rank}.pt" for rank in range(workers)]
    missing = [name for name in names if not (checkpoint / name).is_file()
               or (checkpoint / name).stat().st_size == 0]
    weights = checkpoint / "model.safetensors"
    index_path = checkpoint / "model.safetensors.index.json"
    if not weights.is_file() and not index_path.is_file():
        missing.append("model.safetensors or model.safetensors.index.json")
    elif index_path.is_file():
        index = json.loads(index_path.read_text())
        for shard in set(index["weight_map"].values()):
            path = (checkpoint / shard).resolve()
            if not path.is_relative_to(checkpoint.resolve()) or not path.is_file():
                missing.append(shard)
    if missing:
        raise ValueError(f"Incomplete checkpoint {checkpoint}: {missing}")
    state = json.loads((checkpoint / "training_state.json").read_text())
    if type(state.get("step")) is not int or state["step"] < 0:
        raise ValueError("Invalid checkpoint step")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("training_objective") != "clean_block_ce":
        raise ValueError("Resume requires a recurrent clean-CE paper_recipe checkpoint")
    if (not state.get("pretrained_fingerprint")
            or state["pretrained_fingerprint"] != config.get("pretrained_fingerprint")):
        raise ValueError("Checkpoint pretrained identity differs from its training state")
    return state


def preflight_resume(args, pool):
    state = check_checkpoint(args.resume, len(args.trainer_npus))
    total = math.ceil(args.max_prompts / args.global_prompt_batch) * args.epochs
    total = min(total, args.max_steps) if args.max_steps else total
    # Use the resolved path without mutating args until validation has passed.
    from argparse import Namespace
    request = Namespace(**(vars(args) | {"pool": str(pool)}))
    signature = training_signature(request, len(args.trainer_npus), total, args.target_backend)
    changes = validate_resume_signature(
        state["signature"], signature, args.allow_performance_change
    )
    return state, changes, total


def validate_resume_signature(saved, requested, allow_performance_change=False):
    from .performance import validate_resume_signature as validate_base

    before, after = dict(saved), dict(requested)
    old_scheduler = before.pop("rollout_scheduler", "cohort")
    new_scheduler = after.pop("rollout_scheduler", "cohort")
    if old_scheduler not in {"cohort", "continuous"} or new_scheduler not in {"cohort", "continuous"}:
        raise ValueError("Unknown rollout scheduler in checkpoint signature")
    if before.get("target_backend") != "local" or after.get("target_backend") != "local":
        raise ValueError("This continuation requires a local-target checkpoint")
    changes = validate_base(before, after, allow_performance_change)
    if old_scheduler != new_scheduler:
        if not allow_performance_change:
            raise ValueError("Changing rollout scheduler requires --allow-performance-change")
        changes["rollout_scheduler"] = {"before": old_scheduler, "after": new_scheduler}
    return changes
