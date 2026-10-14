"""Exercise actual CLI model construction and two complete Trainer epochs."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from test_joint import helper, settings
from speculators.train.config import TrainConfig
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.models.dflash_prefix.joint_validation import CachedValidationLoader


@pytest.mark.parametrize("optimizer,recompute", [("adamw", True), ("muon", True), ("muon", False)])
def test_actual_cli_builder_and_full_epochs(tmp_path, monkeypatch, optimizer, recompute):
    import speculators.train.cli as cli
    import speculators.train.trainer as trainer_module
    from speculators.losses import resolve_loss_config
    s = dict(settings(), output_dir=str(tmp_path), epochs=2,
             backbone_checkpointing=recompute, selector_checkpointing=recompute)
    fixture, batch = helper("check_training").make_fixture(tmp_path, s)
    del fixture
    config = TrainConfig.resolve(["--speculator-type", "dflash_prefix",
        "--verifier-name-or-path", str(tmp_path / "tiny_target"), "--num-layers", "1",
        "--draft-arch", "qwen3", "--draft-attn-impl", "sdpa", "--target-layer-ids", "0", "1",
        "--block-size", "8", "--mask-token-id", "0", "--prefix-rank", "16", "--prefix-top-k", "16",
        "--no-resume-from-checkpoint", "--no-embed-requires-grad"])
    model = helper("train_entry").build_model(cli.build_draft_model,
        SimpleNamespace(**config.flatten()), None, None, 64, s)
    assert model.config.prefix_selector_kind == "compiled_prefix_v3"
    assert model.config.prefix_attention_heads == 2
    assert model.prefix_head.successor_projection.weight.count_nonzero() == 0

    class Loader:
        def __init__(self, count):
            self.count, self.epochs = count, []
            self.batch_sampler = self
        def __len__(self):
            return self.count
        def set_epoch(self, epoch):
            self.epochs.append(epoch)
        def __iter__(self):
            for _ in range(self.count):
                yield {**{k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}, "error_records": 0}

    train = Loader(3)
    val = CachedValidationLoader(Loader(2), 2, tmp_path / "cache")
    # Only device placement and the accelerator timing fence are adapted for CPU.
    original_setup = Trainer.setup_model
    def cpu_setup(self):
        self.local_rank, self.device_type = "cpu", "cpu"
        original_setup(self)
    monkeypatch.setattr(Trainer, "setup_model", cpu_setup)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    for key in ("setup_optimizer", "_optimizers_step", "run_training", "val_epoch", "maybe_update_best"):
        monkeypatch.setattr(Trainer, key, getattr(Trainer, key))
    monkeypatch.setattr(cli, "build_draft_model", cli.build_draft_model)
    monkeypatch.setattr(cli, "create_train_val_loaders", cli.create_train_val_loaders)
    monkeypatch.setattr(trainer_module, "build_optimizers", trainer_module.build_optimizers)
    helper("train_entry").install_hooks(cli, trainer_module, s)
    kwargs = dict(max_anchors=4, loss_config=resolve_loss_config("ce", "eager"))
    trainer = Trainer(model, TrainerConfig(lr=1e-4, optimizer=optimizer, muon_lr=.001,
        num_epochs=2, save_path=str(tmp_path / "checkpoints"),
        scheduler_type="cosine", scheduler_warmup_steps=1, gradient_checkpointing=recompute,
        train_call_kwargs=kwargs, val_call_kwargs=kwargs, log_freq=100), train, val)
    trainer.run_training()
    assert trainer.global_step == 6 and train.epochs == [0, 1]
    assert trainer.schedulers[-1].get_last_lr() == [0., 0.]
    history = [json.loads(line) for line in (tmp_path / "checkpoints/joint_history.jsonl").read_text().splitlines()]
    assert [x["step"] for x in history] == [3, 6]
    assert all(x["reload"]["actual_disk_reload"] for x in history)
    assert (tmp_path / "checkpoints/checkpoint_best").is_symlink()
    for epoch in (0, 1):
        exported = json.loads((tmp_path / f"checkpoints/{epoch}/config.json").read_text())
        assert exported["architectures"] == ["DFlashPrefixDraftModel"]
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
