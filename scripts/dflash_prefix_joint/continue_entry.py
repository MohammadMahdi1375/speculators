#!/usr/bin/env python3
"""Resume a selected joint checkpoint with explicit epoch and LR semantics."""
import copy
import json
import os
from pathlib import Path


def install_hooks(cli, trainer_module, settings, continuation):
    import torch
    import torch.distributed as dist
    import train_entry as joint_entry
    from speculators.models.dflash_prefix.joint_continuation import (
        FixedContinuationValidation, JointContinuationCheckpointer, lr_multiplier, restore_schedulers)
    from speculators.models.dflash_prefix.joint_validation import CachedValidationLoader
    old_build, old_loaders = cli.build_draft_model, cli.create_train_val_loaders
    joint_entry.install_hooks(cli, trainer_module, settings)
    Trainer = trainer_module.Trainer
    joint_run = Trainer.run_training
    old_save = Trainer.maybe_save_checkpoint

    def build(args, model_class, t2d, d2t, draft_vocab_size):
        # Construct the original architecture; Trainer loads ALL saved draft parameters before training.
        architecture_args = copy.copy(args)
        architecture_args.no_resume_from_checkpoint = True
        return joint_entry.build_model(old_build, architecture_args, t2d, d2t, draft_vocab_size, settings)

    def loaders(**kwargs):
        train, val = old_loaders(**kwargs)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world != continuation["trainer_world_size"]:
            raise ValueError("Continuation trainer world size changed")
        directory = Path(settings["output_dir"]) / "validation_inputs" / f"rank_{rank}"
        if continuation["validation_mode"] == "create_fixed":
            if val is None or not len(val):
                raise ValueError("Resume requires a nonempty validation split")
            return train, CachedValidationLoader(val, settings["val_batches"], directory)
        return train, FixedContinuationValidation(directory, continuation["validation_manifests"][f"rank_{rank}"])

    def setup(self):
        if (not self.resume_from_checkpoint or self.current_epoch != continuation["start_epoch_index"]
                or self.global_step != continuation["start_global_step"]
                or self._resume_local_step != continuation["resume_local_step"]
                or len(self.train_loader) != continuation["steps_per_epoch"]
                or self.config.num_epochs != continuation["total_epochs"]):
            raise ValueError("Checkpoint, epoch count or data-loader length differs from resume plan")
        self.optimizers = trainer_module.build_optimizers(self.model, self.config)
        # Preserve each freshly constructed optimizer's requested peak LR before loading zero/end LRs.
        peaks = [[group["lr"] for group in opt.param_groups] for opt in self.optimizers]
        self.checkpointer.load_optimizer_state_dict(self.model, self.optimizers, float_dtype=torch.float32)
        # Packing can produce a different number of batches for each epoch.
        sampler = self.train_loader.batch_sampler
        generator = getattr(sampler, "_generate_batches", None)
        skip = continuation["resume_local_step"]
        if skip and (not callable(generator) or not hasattr(sampler, "_cached_generated_batches")):
            raise ValueError("Mid-epoch resume requires the packing sampler's fast-skip API; refusing to replay consumed batches")
        epoch_steps = [len(generator(epoch)) if callable(generator) else len(self.train_loader)
                       for epoch in range(self.current_epoch, self.config.num_epochs)]
        if skip > epoch_steps[0]:
            raise ValueError("Checkpoint local_step exceeds the source epoch's packed batch count")
        epoch_steps[0] -= skip
        total_steps = sum(epoch_steps)
        if total_steps < 1:
            raise ValueError("No optimizer steps remain in this budget; increase EPOCHS")
        warmup_steps = int(total_steps * continuation["warmup_ratio"])
        if continuation["lr_policy"] == "restore":
            spec = continuation["source_scheduler_spec"]
            self.schedulers = restore_schedulers(self.optimizers, self.checkpointer.prev_path, spec, self.global_step)
            peaks = [scheduler.base_lrs for scheduler in self.schedulers]
        else:
            self.config = self.config._replace(scheduler_total_steps=total_steps, scheduler_warmup_steps=warmup_steps)
            self.schedulers = []
            for opt, rates in zip(self.optimizers, peaks, strict=True):
                for group, rate in zip(opt.param_groups, rates, strict=True):
                    group["lr"] = group["initial_lr"] = rate
                self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, lambda step:
                    lr_multiplier(step, total_steps, warmup_steps)))
            spec = dict(kind="restart_cosine", origin_step=self.global_step,
                total_steps=total_steps, warmup_steps=warmup_steps, planned_epochs=self.config.num_epochs)
        self._joint_resume_context = dict(version=1, scheduler_spec=spec,
            trainer_world_size=continuation["trainer_world_size"])
        if self.rank == 0:
            report = dict(continuation, steps_per_epoch=len(self.train_loader),
                full_epochs=self.config.num_epochs, planned_optimizer_steps=self.global_step + total_steps,
                additional_steps=total_steps, warmup_steps=warmup_steps, additional_epoch_steps=epoch_steps,
                restored_global_step=self.global_step, source_epoch_index=continuation["source_epoch_index"],
                scheduler_spec=spec,
                optimizer_types=[type(opt).__name__ for opt in self.optimizers], optimizer_peak_lrs=peaks,
                optimizer_state_entries=[len(opt.state) for opt in self.optimizers],
                backbone_checkpointing=self.config.gradient_checkpointing,
                selector_checkpointing=settings.get("selector_checkpointing", True))
            (Path(settings["output_dir"]) / "resolved_schedule.json").write_text(json.dumps(report, indent=2) + "\n")
            print("JOINT_RESUME_LOADED " + json.dumps(report), flush=True)

    def save(self, epoch, local_step=0):
        old_save(self, epoch, local_step=local_step)
        # Make future resumes independent of implicit LambdaLR closures. Do not
        # invent progress for legacy string-named interrupted checkpoints.
        if self.rank == 0 and isinstance(epoch, int):
            directory = Path(self.checkpointer.path) / str(epoch)
            path = directory / "training_state.json"
            if path.is_file() and json.loads(path.read_text()).get("global_step") == self.global_step:
                (directory / "resume_context.json").write_text(json.dumps(self._joint_resume_context, indent=2) + "\n")

    def run(self):
        raw = getattr(self.model, "module", self.model)
        raw.joint_step = self.global_step
        # Evaluate the selected saved weights before updating them, on the same
        # cohort used throughout this new run. Preserve mid-epoch baseline files.
        baseline = continuation["baseline_checkpoint"]
        metrics = self.val_epoch(baseline)
        self.maybe_update_best(baseline, metrics)
        if self.is_distributed:
            dist.barrier()
        return joint_run(self)

    cli.build_draft_model, cli.create_train_val_loaders = build, loaders
    trainer_module.SingleGPUCheckpointer = JointContinuationCheckpointer
    Trainer.setup_optimizer, Trainer.run_training = setup, run
    Trainer.maybe_save_checkpoint = save


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    import speculators.train.cli as cli
    import speculators.train.trainer as trainer_module
    from speculators.train.config import TrainConfig
    config = TrainConfig.resolve()
    flat = config.flatten()
    output = Path(flat["save_path"]).resolve().parent
    settings = json.loads((output / "joint_settings.json").read_text())
    record = json.loads((output / "continuation.json").read_text())
    if (not record["staged"] or flat["no_resume_from_checkpoint"] or flat["fsdp_shard"]
            or flat["max_steps"] is not None or flat["epochs"] != record["total_epochs"]):
        raise ValueError("Use a fresh prepared full-epoch continuation, not the scratch launcher")
    install_hooks(cli, trainer_module, settings, record)
    cli.main(config)


if __name__ == "__main__":
    main()
