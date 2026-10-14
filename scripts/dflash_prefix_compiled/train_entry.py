#!/usr/bin/env python3
"""Frozen-backbone pilot; evaluation windows are not full dataset epochs."""
import json
import math
import os
from itertools import islice
from pathlib import Path


class FixedSampler:
    def set_epoch(self, epoch):
        pass


class WindowLoader:
    """Continue one data iterator across evaluation windows, without replaying skips."""
    def __init__(self, loader, steps, window):
        self.loader, self.steps, self.window = loader, steps, window
        self.completed, self.data_epoch, self.iterator = 0, 0, None
        self.batch_sampler = FixedSampler()

    def __len__(self):
        return min(self.window, max(0, self.steps - self.completed))

    def __iter__(self):
        count = len(self)
        for _ in range(count):
            if self.iterator is None:
                if hasattr(self.loader.batch_sampler, "set_epoch"):
                    self.loader.batch_sampler.set_epoch(self.data_epoch)
                self.iterator = iter(self.loader)
            try:
                item = next(self.iterator)
            except StopIteration:
                self.data_epoch += 1
                if hasattr(self.loader.batch_sampler, "set_epoch"):
                    self.loader.batch_sampler.set_epoch(self.data_epoch)
                self.iterator = iter(self.loader)
                item = next(self.iterator)
            self.completed += 1
            yield item


class ValidationLoader:
    def __init__(self, loader, batches):
        self.loader, self.batches = loader, batches
        self.batch_sampler = FixedSampler()

    def __len__(self):
        return min(self.batches, len(self.loader))

    def __iter__(self):
        if hasattr(self.loader.batch_sampler, "set_epoch"):
            self.loader.batch_sampler.set_epoch(0)
        return islice(iter(self.loader), len(self))


def install_hooks(cli, trainer_module, settings):
    import torch
    from speculators.models.dflash_prefix.compiled_training import build_compiled_pilot
    Trainer = trainer_module.Trainer
    old_loaders, old_val = cli.create_train_val_loaders, Trainer.val_epoch

    def build(args, model_class, t2d, d2t, draft_vocab_size):
        if not args.no_resume_from_checkpoint or args.fsdp_shard:
            raise ValueError("Pilot requires no resume and no FSDP")
        model = build_compiled_pilot(settings["teacher_checkpoint"], settings["target_path"], settings)
        if model.config.block_size != args.block_size:
            raise ValueError("Pilot model/training block size mismatch")
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        return model

    def loaders(**kwargs):
        train, val = old_loaders(**kwargs)
        if len(train) < 1 or val is None or len(val) < 1:
            raise ValueError("Pilot requires nonempty train and validation splits")
        return WindowLoader(train, settings["pilot_steps"], settings["eval_every"]), ValidationLoader(val, settings["val_batches"])

    def validate(self, epoch):
        devices = [] if self.device_type == "cpu" else [torch.get_device_module(self.device_type).current_device()]
        with torch.random.fork_rng(devices=devices, device_type=self.device_type):
            torch.random.default_generator.manual_seed(4242)
            if self.device_type != "cpu":
                torch.get_device_module(self.device_type).manual_seed(4242)
            metrics = old_val(self, epoch)
        if self.rank == 0:
            record = {"window": epoch, "step": self.global_step, "metrics": metrics,
                      "note": "Reference-prefix validation only; evaluate serving tau and latency separately"}
            with (Path(self.checkpointer.path) / "pilot_history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            link = Path(self.checkpointer.path) / f"step_{self.global_step}"
            if link.exists() and not link.is_symlink():
                raise ValueError(f"Refusing to overwrite ordinary path: {link}")
            link.unlink(missing_ok=True)
            link.symlink_to(str(epoch), target_is_directory=True)
        return metrics

    cli.build_draft_model, cli.create_train_val_loaders = build, loaders
    Trainer.val_epoch = validate
    # Keep every numbered checkpoint. A proxy loss is not a serving speed metric.
    Trainer.maybe_update_best = lambda self, epoch, metrics: None


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    import speculators.train.cli as cli
    import speculators.train.trainer as trainer_module
    from speculators.train.config import TrainConfig
    config = TrainConfig.resolve()
    flat = config.flatten()
    settings = json.loads((Path(flat["save_path"]).resolve().parent / "settings.json").read_text())
    if flat["epochs"] != math.ceil(settings["pilot_steps"] / settings["eval_every"]) or flat["max_steps"] is not None:
        raise ValueError("Use the prepared bounded evaluation-window config")
    install_hooks(cli, trainer_module, settings)
    cli.main(config)


if __name__ == "__main__":
    main()
