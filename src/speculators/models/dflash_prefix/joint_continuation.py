"""Strict state loading and immutable validation for selected-checkpoint resume."""
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from speculators.train.checkpointer import SingleGPUCheckpointer, convert_float_dtype
from .joint_validation import FixedSampler


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


class FixedContinuationValidation:
    def __init__(self, directory, expected_manifest):
        self.directory = Path(directory)
        if sha256(self.directory / "manifest.json") != expected_manifest:
            raise ValueError("Source validation manifest changed")
        self.inventory = json.loads((self.directory / "manifest.json").read_text())
        if not self.inventory:
            raise ValueError("Empty source validation cohort")
        for item in self.inventory:
            name = item["file"]
            path = self.directory / name
            if Path(name).name != name or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"Corrupt or changed validation batch: {path}")
        self.batch_sampler = FixedSampler()

    def __len__(self):
        return len(self.inventory)

    def __iter__(self):
        for item in self.inventory:
            yield torch.load(self.directory / item["file"], weights_only=True, map_location="cpu")


def lr_multiplier(step, total, warmup):
    if total < 1 or not 0 <= warmup < total:
        raise ValueError("Invalid continuation scheduler horizon")
    if warmup and step < warmup:
        return max(0., step / warmup)
    progress = min(1., max(0., (step - warmup) / (total - warmup)))
    return .5 * (1. + math.cos(math.pi * progress))


def restore_schedulers(optimizers, directory, spec, global_step):
    """Reconstruct the original functions before loading LambdaLR state.

    Loading a LambdaLR state dict alone does not restore its lambda function.
    Also restore the exact saved group LRs after the constructor's initial step.
    """
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    states = torch.load(Path(directory) / "scheduler_state_dict.pt", map_location="cpu", weights_only=True)
    states = states if isinstance(states, list) else [states]
    if len(states) != len(optimizers):
        raise ValueError("Saved scheduler count differs from current optimizer count")
    schedulers = []
    clock = global_step - spec["origin_step"]
    for index, (opt, state) in enumerate(zip(optimizers, states, strict=True)):
        if state.get("last_epoch") != clock:
            raise ValueError("Saved scheduler clock differs from checkpoint global_step")
        rates = [group["lr"] for group in opt.param_groups]
        if (rates != state.get("_last_lr") or len(state.get("base_lrs", [])) != len(rates)):
            raise ValueError("Saved optimizer LRs and scheduler state disagree")
        for group, rate in zip(opt.param_groups, state["base_lrs"], strict=True):
            group["lr"] = group["initial_lr"] = rate
        if spec["kind"] == "restart_cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt,
                lambda step: lr_multiplier(step, spec["total_steps"], spec["warmup_steps"]))
        elif spec["kind"] == "joint_scratch":
            if index == len(optimizers) - 1:
                def head(step):
                    active = step - spec["head_delay"]
                    if active <= 0:
                        return 0.
                    if active < spec["head_warmup"]:
                        return active / spec["head_warmup"]
                    progress = min(1., (active - spec["head_warmup"]) /
                                   max(1, spec["head_total_steps"] - spec["head_delay"] - spec["head_warmup"]))
                    return .5 * (1. + math.cos(math.pi * progress))
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, head)
            elif spec["backbone_kind"] == "linear":
                scheduler = get_linear_schedule_with_warmup(opt,
                    num_warmup_steps=spec["warmup_steps"], num_training_steps=spec["total_steps"])
            elif spec["backbone_kind"] == "cosine":
                scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=spec["warmup_steps"],
                    num_training_steps=spec["total_steps"], num_cycles=spec["num_cycles"])
            else:
                raise ValueError("Unknown saved backbone scheduler kind")
        else:
            raise ValueError("Unknown saved joint scheduler kind")
        scheduler.load_state_dict(state)
        for group, rate in zip(opt.param_groups, rates, strict=True):
            group["lr"] = rate
        schedulers.append(scheduler)
    return schedulers


class JointContinuationCheckpointer(SingleGPUCheckpointer):
    def load_model_state_dict(self, model, float_dtype=None):
        directory = self.path / str(self.previous_epoch)
        cfg = json.loads((directory / "config.json").read_text())
        for key in ("block_size", "prefix_selector_kind", "prefix_rank", "prefix_top_k", "prefix_attention_heads",
                    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size",
                    "sliding_window", "target_layer_ids", "prefix_compiled_logit_cap"):
            if key in cfg and cfg[key] != getattr(model.config, key, None):
                raise ValueError(f"Constructed architecture differs from checkpoint: {key}")
        state = load_file(str(self.model_path(self.previous_epoch)), device="cpu")
        ignored = set(getattr(model, "_keys_to_ignore_on_save", []))
        expected = set(model.state_dict()) - ignored
        if set(state) != expected:
            raise ValueError(f"Checkpoint tensor keys differ: missing={sorted(expected - set(state))[:8]}, extra={sorted(set(state) - expected)[:8]}")
        # copy_ preserves the FP32 trainable masters and FP32 rotary buffers.
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) - ignored:
            raise ValueError(f"Incomplete checkpoint load: {result}")

    def load_optimizer_state_dict(self, model, optimizer, float_dtype=None):
        optimizers = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
        payload = torch.load(self.optimizer_path(self.previous_epoch), weights_only=True, map_location="cpu")
        saved = payload if isinstance(payload, list) else [payload]
        if len(saved) != len(optimizers):
            raise ValueError("Saved optimizer count differs from current training recipe")
        for opt, state in zip(optimizers, saved, strict=True):
            if len(opt.param_groups) != len(state["param_groups"]):
                raise ValueError("Saved optimizer groups differ from current recipe")
            known = set()
            for group, old in zip(opt.param_groups, state["param_groups"], strict=True):
                if len(group["params"]) != len(old["params"]):
                    raise ValueError("Saved optimizer parameter count differs")
                if "param_names" in old:
                    names = lambda xs: [name.removeprefix("module.") for name in xs]
                    if names(group.get("param_names", [])) != names(old["param_names"]):
                        raise ValueError("Saved optimizer parameter ordering differs")
                for parameter, index in zip(group["params"], old["params"], strict=True):
                    known.add(index)
                    for name, value in state["state"].get(index, {}).items():
                        if isinstance(value, torch.Tensor) and name != "step" and value.shape != parameter.shape:
                            raise ValueError(f"Saved optimizer state shape mismatch: {name}")
            if set(state["state"]) - known:
                raise ValueError("Orphaned saved optimizer state")
            opt.load_state_dict(convert_float_dtype(state, torch.float32))
            # Do not infer exact pre-export moments/counters; the source rounded them to BF16.
