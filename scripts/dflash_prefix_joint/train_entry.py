#!/usr/bin/env python3
"""Full dataset epochs; joint scratch model; optional selector update guard."""
import json
import math
import os
from pathlib import Path


def ensure_scratch(flat):
    if any(flat.get(k) for k in ("from_pretrained", "draft_config", "prefix_backbone_init")):
        raise ValueError("Joint scratch must not load trained draft weights or configs")
    if not flat["no_resume_from_checkpoint"] or flat["fsdp_shard"] or flat["max_steps"] is not None:
        raise ValueError("Use fresh full-epoch DDP training; resume/FSDP/max_steps are unsupported")


def build_model(builder, args, t2d, d2t, draft_vocab_size, settings):
    from speculators.models.dflash_prefix.joint_training import JointCompiledScratch
    ensure_scratch(vars(args))
    if t2d is not None or d2t is not None:
        raise ValueError("Joint compiled training requires the full target vocabulary")
    args.prefix_selector_kind = "compiled_prefix_v3"
    args.prefix_attention_heads, args.prefix_attention_layers = settings["heads"], 1
    args.prefix_compiled_logit_cap = settings["compiled_logit_cap"]
    args.prefix_history_scale = 1.
    model = builder(args, JointCompiledScratch, t2d, d2t, draft_vocab_size)
    model.configure_joint(settings)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if not all(p.requires_grad for p in model.prefix_head.parameters()):
        raise ValueError("All compiled selector parameters must learn from scratch")
    if not any(p.requires_grad for p in model.layers.parameters()):
        raise ValueError("Draft backbone was accidentally frozen")
    if model.block_size != settings["block_size"] or model.prefix_head.top_k != settings["top_k"]:
        raise ValueError("Training/selector dimensions differ")
    return model


def build_optimizers(model, config, settings, backbone_builder):
    import torch

    class BackboneParameters:
        def named_parameters(self):
            return ((n, p) for n, p in model.named_parameters()
                    if p.requires_grad and not n.removeprefix("module.").startswith("prefix_head."))

    class SelectorAdamW(torch.optim.AdamW):
        pass

    optimizers = backbone_builder(BackboneParameters(), config)
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name.removeprefix("module.").startswith("prefix_head."):
            groups.setdefault(p.ndim >= 2, []).append(p)
    optimizers.append(SelectorAdamW([
        dict(params=ps, weight_decay=settings["head_weight_decay"] if decay else 0.)
        for decay, ps in groups.items()], lr=settings["head_lr"]))
    return optimizers


def head_lr_multiplier(step, total, delay, warmup):
    active = step - delay
    if active <= 0:
        return 0.
    if active < warmup:
        return active / warmup
    return .5 * (1 + math.cos(math.pi * min(1., (active - warmup) / max(1, total - delay - warmup))))


def training_memory_record(model, rank, step):
    """Allocator counters only; no extra accelerator synchronization."""
    import torch
    device = model.prefix_head.head_weights.device
    if device.type not in {"npu", "cuda"}:
        return None
    backend = getattr(torch, device.type)
    fields = {"allocated_gib": "memory_allocated", "reserved_gib": "memory_reserved",
              "peak_allocated_gib": "max_memory_allocated", "peak_reserved_gib": "max_memory_reserved"}
    return dict(rank=rank, completed_steps=step, device=str(device),
                peak_scope="process lifetime, includes validation", **{
        name: getattr(backend, method)(device) / 2**30 if hasattr(backend, method) else None
        for name, method in fields.items()})


def install_hooks(cli, trainer_module, settings):
    import torch
    from speculators.models.dflash_prefix.joint_validation import CachedValidationLoader, checkpoint_parameters
    Trainer = trainer_module.Trainer
    old_build, old_loaders = cli.build_draft_model, cli.create_train_val_loaders
    old_setup, old_run, old_val = Trainer.setup_optimizer, Trainer.run_training, Trainer.val_epoch
    backbone_builder = trainer_module.build_optimizers

    def build(args, model_class, t2d, d2t, draft_vocab_size):
        return build_model(old_build, args, t2d, d2t, draft_vocab_size, settings)

    def loaders(**kwargs):
        train, val = old_loaders(**kwargs)
        if not len(train) or val is None or not len(val):
            raise ValueError("Nonempty training and held-out validation splits are required")
        return train, CachedValidationLoader(val, settings["val_batches"],
            Path(settings["output_dir"]) / "validation_inputs" / f"rank_{os.environ.get('RANK', '0')}")

    def setup(self):
        total = self.config.num_epochs * len(self.train_loader)
        if settings["delay_steps"] + settings["head_warmup_steps"] >= total:
            raise ValueError("Selector delay plus warmup must be shorter than the full run")
        if settings["guard_mode"] == "project" and settings["guard_start_steps"] >= total:
            raise ValueError("Guard would never activate; reduce GUARD_START_STEPS or use GUARD_MODE=off")
        old_setup(self)
        head = self.optimizers[-1]
        for group in head.param_groups:
            group["lr"] = group["initial_lr"] = settings["head_lr"]
        self.schedulers[-1] = torch.optim.lr_scheduler.LambdaLR(head,
            lambda step: head_lr_multiplier(step, total, settings["delay_steps"], settings["head_warmup_steps"]))
        if self.rank == 0:
            record = dict(steps_per_epoch=len(self.train_loader), full_epochs=self.config.num_epochs,
                          planned_optimizer_steps=total, guard_mode=settings["guard_mode"],
                          inference_architecture="compiled_prefix_v3", pretrained_draft=None,
                          backbone_checkpointing=self.config.gradient_checkpointing,
                          selector_checkpointing=settings.get("selector_checkpointing", True),
                          anchor_chunk_size=settings["anchor_chunk_size"], guard_chunk_size=settings["guard_chunk_size"],
                          packed_tokens_per_rank=settings.get("packed_tokens_per_rank"))
            (Path(settings["output_dir"]) / "resolved_schedule.json").write_text(json.dumps(record, indent=2) + "\n")
            print("JOINT_SCRATCH_SCHEDULE " + json.dumps(record), flush=True)

    def step(self):
        model = getattr(self.model, "module", self.model)
        for optimizer in self.optimizers[:-1]:
            optimizer.step()
        head_optimizer = self.optimizers[-1]
        report = dict(active=0., protected_positions=0., projected=0., rejected=0., accepted_scale=1.,
                      proposed_lost_tokens=0., accepted_lost_tokens=0., backtracks=0.)
        if self.global_step < settings["delay_steps"]:
            for group in head_optimizer.param_groups:
                for p in group["params"]:
                    p.grad = None
        elif (model.selector_guard is not None and self.global_step >= settings["guard_start_steps"]
              and (self.global_step - settings["guard_start_steps"]) % settings["guard_every"] == 0):
            report = model.selector_guard.step(model.prefix_head, head_optimizer, model.embed_tokens.weight)
        else:
            head_optimizer.step()
        if model.joint_metrics is not None:
            device = model.prefix_head.head_weights.device
            for key, value in report.items():
                model.joint_metrics["guard_" + key + "_sum"] = torch.tensor(value, device=device)
                model.joint_metrics["guard_" + key + "_total"] = torch.ones((), device=device)
        interval = settings.get("memory_log_every", 0)
        if interval and (self.global_step == 0 or (self.global_step + 1) % interval == 0):
            memory = training_memory_record(model, self.rank, self.global_step + 1)
            if memory is not None:
                memory["guard_active"] = bool(report["active"])
                print("JOINT_TRAIN_MEMORY " + json.dumps(memory), flush=True)

    def validate(self, epoch):
        raw = getattr(self.model, "module", self.model)
        devices = [] if self.device_type == "cpu" else [torch.get_device_module(self.device_type).current_device()]
        directory = Path(self.checkpointer.path) / str(epoch)
        with torch.random.fork_rng(devices=devices, device_type=self.device_type), checkpoint_parameters(raw, directory) as (replay_model, reload_info):
            torch.random.default_generator.manual_seed(4242)
            if self.device_type != "cpu":
                torch.get_device_module(self.device_type).manual_seed(4242)
            # This separate saved-weight model never changes the live DDP
            # Parameter objects, dtype, storage, gradient accumulators or hooks.
            wrapped = self.model
            try:
                self.model = replay_model
                metrics = old_val(self, epoch)
            finally:
                self.model = wrapped
        del replay_model
        if self.rank == 0:
            record = dict(epoch=epoch, step=self.global_step, metrics=metrics, reload=reload_info,
                          note="Actual saved weights and fixed cached inputs; offline reference proxy, not serving tau")
            with (Path(self.checkpointer.path) / "joint_history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return metrics

    def best(self, epoch, metrics):
        value = (metrics or {}).get("rollout_reference_eal_epoch")
        if value is None or not math.isfinite(value):
            raise ValueError("Missing finite reference EAL; cannot choose a checkpoint")
        if value <= getattr(self, "_joint_best", -float("inf")):
            return
        self._joint_best = value
        self.checkpointer.save_val_metrics(epoch, metrics)
        self.checkpointer.update_best_symlink(epoch)
        if self.rank == 0:
            print(f"JOINT_BEST epoch={epoch} saved_weight_reference_eal={value:.6f}; serving evaluation still required", flush=True)

    def run(self):
        model = getattr(self.model, "module", self.model)
        def schedule(module, inputs):
            module.joint_step = self.global_step
        handle = model.register_forward_pre_hook(schedule)
        try:
            return old_run(self)
        finally:
            handle.remove()

    cli.build_draft_model, cli.create_train_val_loaders = build, loaders
    trainer_module.build_optimizers = lambda model, cfg: build_optimizers(model, cfg, settings, backbone_builder)
    Trainer.setup_optimizer, Trainer._optimizers_step = setup, step
    Trainer.val_epoch, Trainer.maybe_update_best, Trainer.run_training = validate, best, run


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    import speculators.train.cli as cli
    import speculators.train.trainer as trainer_module
    from speculators.train.config import TrainConfig
    config = TrainConfig.resolve()
    flat = config.flatten()
    ensure_scratch(flat)
    settings = json.loads((Path(flat["save_path"]).resolve().parent / "joint_settings.json").read_text())
    if settings["epochs"] != flat["epochs"] or settings["block_size"] != flat["block_size"]:
        raise ValueError("Prepared settings and training config differ")
    install_hooks(cli, trainer_module, settings)
    cli.main(config)


if __name__ == "__main__":
    main()
