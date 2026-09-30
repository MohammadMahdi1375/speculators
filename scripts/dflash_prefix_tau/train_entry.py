"""Full scratch training: original backbone optimizer, separate selector optimizer."""
import json
import math
import os
from itertools import islice
from pathlib import Path


class FixedValidationSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(0)


class ValidationLoader:
    def __init__(self, loader, batches):
        self.loader, self.batches = loader, batches
        self.batch_sampler = FixedValidationSampler(loader.batch_sampler)

    def __len__(self):
        return min(self.batches, len(self.loader))

    def __iter__(self):
        return islice(iter(self.loader), len(self))


def ensure_scratch(flat):
    if (flat["from_pretrained"] or flat["draft_config"] or flat["prefix_backbone_init"]
            or not flat["no_resume_from_checkpoint"] or flat["fsdp_shard"]):
        raise ValueError("Scratch training requires empty draft weight/config paths, no resume and no FSDP")
    if flat["max_steps"] is not None:
        raise ValueError("This experiment uses full epochs; remove max_steps")


def prepare_trainable_parameters(model):
    import torch
    if model.config.prefix_selector_kind != "local_prefix_v2" or model.config.prefix_freeze_backbone:
        raise ValueError("Install the tau update; scratch training needs its unfrozen new model")
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if any(p.requires_grad and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Trainable master parameters must be FP32")
    return model


def build_scratch_model(builder, args, model_class, t2d, d2t, draft_vocab_size, settings):
    ensure_scratch(vars(args))
    if args.block_size != settings["block_size"]:
        raise ValueError("Training config and selector settings disagree about block_size")
    # Additional architecture fields go directly to from_training_args.
    args.prefix_selector_kind = settings["selector_kind"]
    args.prefix_attention_heads = settings["heads"]
    args.prefix_attention_layers = settings["layers"]
    args.prefix_history_scale = 1.0
    model = builder(args, model_class, t2d, d2t, draft_vocab_size)
    model.config.prefix_loss_scope = "reachable_greedy_prefix"
    model.config.prefix_damage_weight = 1.0
    if model.config.prefix_rank != settings["rank"] or model.config.prefix_top_k != settings["top_k"]:
        raise ValueError("CLI/selector settings disagree")
    proposals = settings["block_size"] - 1
    if (model.config.block_size != settings["block_size"]
            or model.prefix_head.max_proposals != proposals
            or settings["num_speculative_tokens"] != proposals):
        raise ValueError("Model/head proposal count must equal training block_size - 1")
    if any(layer.distance_bias.shape[-1] != settings["block_size"]
           for layer in model.prefix_head.prefix_layers):
        raise ValueError("Selector distance-bias width does not match block_size")
    return prepare_trainable_parameters(model)


def build_scratch_optimizers(model, config, settings, backbone_builder):
    import torch

    def is_head(name):
        return name.removeprefix("module.").startswith("prefix_head.")

    class BackboneParameters:
        def named_parameters(self):
            return ((n, p) for n, p in model.named_parameters() if p.requires_grad and not is_head(n))

    class SelectorAdamW(torch.optim.AdamW):
        """Distinct name makes the original trainer log its LR separately."""

    optimizers = backbone_builder(BackboneParameters(), config)
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and is_head(name):
            decay = parameter.ndim >= 2 and "distance_bias" not in name
            groups.setdefault(decay, []).append(parameter)
    head = SelectorAdamW([
        {"params": parameters, "weight_decay": settings["head_weight_decay"] if decay else 0.0}
        for decay, parameters in groups.items()
    ], lr=settings["head_lr"])
    return optimizers + [head]


def selector_lr_multiplier(step, total_steps, delay_steps, warmup_steps):
    active = step - delay_steps
    if active <= 0:
        return 0.0
    if active < warmup_steps:
        return active / warmup_steps
    remaining = max(1, total_steps - delay_steps - warmup_steps)
    progress = min(1.0, (active - warmup_steps) / remaining)
    return .5 * (1 + math.cos(math.pi * progress))


def update_best(trainer, epoch, metrics):
    tau = metrics.get("rollout_reference_eal_epoch") if metrics else None
    if tau is None or not math.isfinite(tau):
        raise ValueError("Missing/non-finite rollout_reference_eal; cannot select checkpoint")
    record = {"epoch": epoch, "global_step": trainer.global_step, "metrics": metrics,
              "selection_metric": "rollout_reference_eal_epoch",
              "note": "Held-out reference-prefix proxy; measure serving tau separately."}
    root = Path(trainer.checkpointer.path)
    improved = tau > getattr(trainer, "_tau_best", -float("inf"))
    if improved:
        trainer._tau_best = tau
    if trainer.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "tau_history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if improved:
            link = root / "checkpoint_best"
            if link.exists() and not link.is_symlink():
                raise ValueError("checkpoint_best is an ordinary directory")
            link.unlink(missing_ok=True)
            link.symlink_to(Path(str(epoch)), target_is_directory=True)
            (root / "best_tau.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"TAU_SELECTION: epoch={epoch}, tau={tau:.6f}, "
              f"best={trainer._tau_best:.6f}, improved={improved}", flush=True)


def install_training_hooks(cli, trainer_module, config, settings, *, val_batches, device_type="npu"):
    import torch
    from speculators.train.trainer import Trainer
    flat = config.flatten()
    ensure_scratch(flat)
    if val_batches < 1:
        raise ValueError("VAL_BATCHES must be positive")
    old_loaders, old_build = cli.create_train_val_loaders, cli.build_draft_model
    old_val, old_run = Trainer.val_epoch, Trainer.run_training
    old_setup, old_step = Trainer.setup_optimizer, Trainer._optimizers_step
    backbone_builder = trainer_module.build_optimizers

    def loaders(**kwargs):
        train, val = old_loaders(**kwargs)
        if len(train) == 0 or val is None or len(val) == 0:
            raise ValueError("Nonempty training and held-out validation splits are required")
        # Return the training loader unchanged: every epoch is complete.
        return train, ValidationLoader(val, val_batches)

    def build(args, model_class, t2d, d2t, draft_vocab_size):
        return build_scratch_model(old_build, args, model_class, t2d, d2t,
                                   draft_vocab_size, settings)

    def setup(self):
        total = self.config.num_epochs * len(self.train_loader)
        if settings["delay_steps"] + settings["head_warmup_steps"] >= total:
            raise ValueError("Selector delay plus warm-up must be shorter than total training")
        old_setup(self)
        if len(self.schedulers) != len(self.optimizers):
            raise ValueError("Use the prepared cosine scheduler config")
        head = self.optimizers[-1]
        for group in head.param_groups:
            group["lr"] = group["initial_lr"] = settings["head_lr"]
        self.schedulers[-1] = torch.optim.lr_scheduler.LambdaLR(
            head, lambda step: selector_lr_multiplier(step, total, settings["delay_steps"],
                                                       settings["head_warmup_steps"]))

    def step(self):
        if self.global_step < settings["delay_steps"]:
            # DDP still sees every parameter; the dormant head accumulates no Adam state.
            for group in self.optimizers[-1].param_groups:
                for parameter in group["params"]:
                    parameter.grad = None
        return old_step(self)

    def validate(self, epoch):
        device = torch.get_device_module(device_type)
        devices = [] if device_type == "cpu" else [device.current_device()]
        with torch.random.fork_rng(devices=devices, device_type=device_type):
            torch.random.default_generator.manual_seed(4242)
            if device_type != "cpu":
                device.manual_seed(4242)
            return old_val(self, epoch)

    def run(self):
        model = getattr(self.model, "module", self.model)

        def loss_schedule(module, inputs):
            active = self.global_step >= settings["delay_steps"]
            module.config.prefix_loss_alpha = settings["loss_alpha"] if active else 0.0

        handle = model.register_forward_pre_hook(loss_schedule)
        try:
            return old_run(self)
        finally:
            handle.remove()
            model.config.prefix_loss_alpha = settings["loss_alpha"]

    cli.create_train_val_loaders, cli.build_draft_model = loaders, build
    trainer_module.build_optimizers = lambda model, cfg: build_scratch_optimizers(
        model, cfg, settings, backbone_builder)
    Trainer.setup_optimizer, Trainer._optimizers_step = setup, step
    Trainer.val_epoch, Trainer.run_training = validate, run
    Trainer.maybe_update_best = lambda self, epoch, metrics: update_best(self, epoch, metrics)


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
    run_root = Path(flat["save_path"]).resolve().parent
    settings = json.loads((run_root / "selector_settings.json").read_text())
    install_training_hooks(cli, trainer_module, config, settings,
                           val_batches=int(os.environ["VAL_BATCHES"]))
    cli.main(config)


if __name__ == "__main__":
    main()
