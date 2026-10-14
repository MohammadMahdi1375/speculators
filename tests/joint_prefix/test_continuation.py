"""Actual selected checkpoint reload, scheduler restore/extension and training."""
import copy
import json
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import load_file

from test_joint import helper, settings, ROOT
from speculators.losses import resolve_loss_config
from speculators.train.config import TrainConfig
from speculators.train.config.schema import CONFIG_DESTS, nest_flat
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.models.dflash_prefix.joint_validation import CachedValidationLoader
from speculators.models.dflash_prefix.joint_continuation import (
    FixedContinuationValidation, JointContinuationCheckpointer, lr_multiplier, sha256)


class Loader:
    def __init__(self, batch, count=2, epoch_counts=None):
        self.batch, self.count, self.epochs = batch, count, []
        self.epoch, self.epoch_counts = 0, epoch_counts or {}
        self._cached_generated_batches = (-1, [])
        self.yielded = []
        self.batch_sampler = self
    def __len__(self):
        return len(self._generate_batches(self.epoch))
    def _generate_batches(self, epoch):
        if self._cached_generated_batches[0] != epoch:
            self._cached_generated_batches = (epoch, [[i] for i in range(self.epoch_counts.get(epoch, self.count))])
        return self._cached_generated_batches[1]
    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        self.epoch = epoch
    def __iter__(self):
        for index, in self._generate_batches(self.epoch):
            self.yielded.append((self.epoch, index))
            yield {**{k: v.clone() for k, v in self.batch.items() if isinstance(v, torch.Tensor)}, "error_records": 0}


def cpu_and_hooks(monkeypatch):
    import speculators.train.cli as cli
    import speculators.train.trainer as tm
    original_setup = Trainer.setup_model
    def setup(self):
        self.local_rank, self.device_type = "cpu", "cpu"
        original_setup(self)
    monkeypatch.setattr(Trainer, "setup_model", setup)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    for key in ("setup_optimizer", "_optimizers_step", "run_training", "val_epoch", "maybe_update_best", "maybe_save_checkpoint"):
        monkeypatch.setattr(Trainer, key, getattr(Trainer, key))
    for key in ("build_draft_model", "create_train_val_loaders"):
        monkeypatch.setattr(cli, key, getattr(cli, key))
    for key in ("build_optimizers", "SingleGPUCheckpointer"):
        monkeypatch.setattr(tm, key, getattr(tm, key))
    return cli, tm


def create_source(directory, optimizer, monkeypatch, capture_mid=False):
    directory.mkdir()
    s = dict(settings(), output_dir=str(directory), epochs=5, selector_kind="compiled_prefix_v3",
             backbone_checkpointing=False, selector_checkpointing=False, memory_log_every=0,
             packed_tokens_per_rank=32)
    fixture, batch = helper("check_training").make_fixture(directory, s)
    del fixture
    data = directory / "dataset"
    data.mkdir()
    config = TrainConfig.resolve(["--speculator-type", "dflash_prefix", "--verifier-name-or-path", str(directory / "tiny_target"),
        "--data-path", str(data), "--num-layers", "1", "--draft-arch", "qwen3", "--draft-attn-impl", "sdpa",
        "--target-layer-ids", "0", "1", "--block-size", "8", "--mask-token-id", "0", "--prefix-rank", "16", "--prefix-top-k", "16",
        "--epochs", "5", "--lr", "0.0001", "--muon-lr", "0.001", "--optimizer", optimizer,
        "--scheduler-type", "cosine", "--scheduler-warmup-steps", "1",
        "--total-seq-len", "32", "--save-path", str(directory / "checkpoints"),
        "--no-resume-from-checkpoint", "--no-embed-requires-grad"])
    flat = config.flatten()
    nested = nest_flat({k: v for k, v in flat.items() if k in CONFIG_DESTS})
    nested["backend"] = dict(config.backend_args)
    (directory / "train_config.yaml").write_text(yaml.safe_dump({"train": nested}))
    (directory / "joint_settings.json").write_text(json.dumps(s))
    with monkeypatch.context() as m:
        cli, tm = cpu_and_hooks(m)
        model = helper("train_entry").build_model(cli.build_draft_model, SimpleNamespace(**flat), None, None, 64, s)
        helper("train_entry").install_hooks(cli, tm, s)
        if capture_mid:
            original_save = Trainer.maybe_save_checkpoint
            def save(self, epoch, local_step=0):
                original_save(self, epoch, local_step)
                if epoch == 2 and local_step == 1:
                    shutil.copytree(directory / "checkpoints/2", directory / "checkpoints/mid_epoch2")
            m.setattr(Trainer, "maybe_save_checkpoint", save)
        kwargs = dict(max_anchors=4, loss_config=resolve_loss_config("ce", "eager"))
        trainer = Trainer(model, TrainerConfig(lr=flat["lr"], muon_lr=flat["muon_lr"], optimizer=optimizer,
            num_epochs=5, save_path=flat["save_path"], scheduler_type="cosine", scheduler_warmup_steps=1,
            checkpoint_freq=.5 if capture_mid else 1.,
            train_call_kwargs=kwargs, val_call_kwargs=kwargs, log_freq=100), Loader(batch),
            CachedValidationLoader(Loader(batch), 2, directory / "validation_inputs/rank_0"))
        trainer.run_training()
    assert trainer.global_step == 10
    return batch


def environment(source, output, check="0"):
    return dict(RESUME_CHECKPOINT=str(source / "checkpoints/4"), OUTPUT_DIR=str(output), EPOCHS="8", CHECK_ONLY=check,
                VLLM_NPUS="0", VLLM_DP="1", TRAIN_NPUS="1", NUM_TRAIN_NPUS="1", VLLM_PORT="8092", VLLM_RPC_PORT="29692")


def assert_saved_optimizer_loaded(optimizers, checkpoint):
    payload = torch.load(checkpoint / "optimizer_state_dict.pt", weights_only=True, map_location="cpu")
    for opt, saved in zip(optimizers, payload, strict=True):
        actual = opt.state_dict()
        assert actual["state"].keys() == saved["state"].keys()
        for index, values in saved["state"].items():
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(actual["state"][index][name].cpu(), value.float()), name
                else:
                    assert actual["state"][index][name] == value


@pytest.mark.parametrize("optimizer,variable_packing", [("adamw", False), ("muon", False), ("muon", True)])
def test_five_completed_epochs_continue_for_three_without_source_changes(tmp_path, monkeypatch, optimizer, variable_packing):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/dflash_prefix_joint"))
    source, output = tmp_path / "source", tmp_path / "continuation"
    batch = create_source(source, optimizer, monkeypatch)
    before = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    prep = helper("prepare_continuation")
    check = tmp_path / "check_only"
    prep.prepare(environment(source, check, "1"))
    assert not (check / "checkpoints").exists()
    prep.prepare(environment(source, output))
    flat = TrainConfig.resolve(["--config", str(output / "train_config.yaml")]).flatten()
    s = json.loads((output / "joint_settings.json").read_text())
    record = json.loads((output / "continuation.json").read_text())
    assert flat["epochs"] == 8 and not flat["no_resume_from_checkpoint"]
    assert flat["lr"] == pytest.approx(1e-5) and s["head_lr"] == pytest.approx(3e-5)
    assert flat["total_seq_len"] == 32 and not s["selector_checkpointing"]
    assert s["guard_mode"] == "project"
    with monkeypatch.context() as m:
        cli, tm = cpu_and_hooks(m)
        original_builder = cli.build_draft_model
        entry = helper("continue_entry")
        entry.install_hooks(cli, tm, s, record)
        model = cli.build_draft_model(SimpleNamespace(**flat), None, None, None, 64)
        train = Loader(batch, epoch_counts={5: 2, 6: 3, 7: 4} if variable_packing else {})
        val = FixedContinuationValidation(output / "validation_inputs/rank_0", record["validation_manifests"]["rank_0"])
        kwargs = dict(max_anchors=4, loss_config=resolve_loss_config("ce", "eager"))
        trainer = Trainer(model, TrainerConfig(lr=flat["lr"], muon_lr=flat["muon_lr"], optimizer=optimizer,
            num_epochs=8, save_path=flat["save_path"], resume_from_checkpoint=True,
            scheduler_type="cosine", train_call_kwargs=kwargs, val_call_kwargs=kwargs, log_freq=100), train, val)
        assert trainer.current_epoch == 5 and trainer.global_step == 10
        assert model.selector_guard.replay.data is None
        disk = load_file(str(source / "checkpoints/4/model.safetensors"))
        for name, tensor in disk.items():
            assert torch.equal(model.state_dict()[name], tensor.float()), name
        assert_saved_optimizer_loaded(trainer.optimizers, source / "checkpoints/4")
        assert trainer.schedulers[-1].get_last_lr() == pytest.approx([3e-5, 3e-5])
        trainer.run_training()
        expected_steps = [10, 12, 15, 19] if variable_packing else [10, 12, 14, 16]
        assert trainer.global_step == expected_steps[-1] and train.epochs == [5, 6, 7]
        assert all(all(rate == 0 for rate in sch.get_last_lr()) for sch in trainer.schedulers)
        assert model.selector_guard.replay.data is not None
        assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    history = [json.loads(x) for x in (output / "checkpoints/joint_history.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in history] == [4, 5, 6, 7]
    assert [x["step"] for x in history] == expected_steps
    for epoch, steps in zip((5, 6, 7), expected_steps[1:], strict=True):
        assert json.loads((output / f"checkpoints/{epoch}/training_state.json").read_text())["global_step"] == steps
    assert int((output / "checkpoints/checkpoint_best").readlink().name) in (4, 5, 6, 7)
    assert before == {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fresh"):
        prep.prepare(environment(source, output))
    with pytest.raises(ValueError, match="world size"):
        prep.prepare(dict(environment(source, tmp_path / "wrong_ranks"), TRAIN_NPUS="1,2", NUM_TRAIN_NPUS="2"))
    state_path = source / "checkpoints/4/training_state.json"
    state = json.loads(state_path.read_text())
    state["local_step"] = -1
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="training progress"):
        prep.prepare(environment(source, tmp_path / "mid_epoch"))


def test_rewarm_and_cosine_terminate_at_zero():
    values = [lr_multiplier(i, 10, 2) for i in range(11)]
    assert values[:3] == [0., .5, 1.]
    assert all(a > b for a, b in zip(values[2:-1], values[3:]))
    assert values[-1] == 0 and lr_multiplier(15, 10, 2) == 0
    with pytest.raises(ValueError):
        lr_multiplier(0, 10, 10)


def test_invalid_or_missing_optimizer_state_is_rejected(tmp_path):
    root = tmp_path / "checkpoints"
    directory = root / "4"
    directory.mkdir(parents=True)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    payload = optimizer.state_dict()
    payload["state"][0]["exp_avg"] = torch.zeros(17)
    torch.save(payload, directory / "optimizer_state_dict.pt")
    with pytest.raises(ValueError, match="shape mismatch"):
        JointContinuationCheckpointer(root).load_optimizer_state_dict(model, optimizer)


def test_continuation_two_rank_ddp_reload_backward_guard_and_validation(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("continuation_ddp_runner.py")),
        "--output", str(tmp_path)], capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        report = json.loads((tmp_path / f"rank_{rank}/result.json").read_text())
        assert report["pass_test"] and report["continued_optimizer_steps"] == 6 and report["final_global_step"] == 16


def resumed_trainer(output, batch, monkeypatch, fresh_validation=False):
    flat = TrainConfig.resolve(["--config", str(output / "train_config.yaml")]).flatten()
    s = json.loads((output / "joint_settings.json").read_text())
    record = json.loads((output / "continuation.json").read_text())
    cli, tm = cpu_and_hooks(monkeypatch)
    if fresh_validation:
        monkeypatch.setattr(cli, "create_train_val_loaders", lambda **kwargs: (Loader(batch), Loader(batch)))
    helper("continue_entry").install_hooks(cli, tm, s, record)
    model = cli.build_draft_model(SimpleNamespace(**flat), None, None, None, 64)
    if fresh_validation:
        train, val = cli.create_train_val_loaders()
    else:
        train = Loader(batch)
        val = FixedContinuationValidation(output / "validation_inputs/rank_0", record["validation_manifests"]["rank_0"])
    kwargs = dict(max_anchors=4, loss_config=resolve_loss_config("ce", "eager"))
    trainer = Trainer(model, TrainerConfig(lr=flat["lr"], muon_lr=flat["muon_lr"], optimizer=flat["optimizer"],
        num_epochs=flat["epochs"], save_path=flat["save_path"], resume_from_checkpoint=True,
        scheduler_type="cosine", train_call_kwargs=kwargs, val_call_kwargs=kwargs, log_freq=100), train, val)
    return trainer, record


@pytest.mark.parametrize("selection", ["earlier", "linked", "copied", "mid_epoch", "uncached_mid"])
def test_explicit_checkpoint_selection_restores_schedule_and_skips_consumed_batches(tmp_path, monkeypatch, selection):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/dflash_prefix_joint"))
    source, output = tmp_path / "source", tmp_path / "resumed"
    mid = selection in ("mid_epoch", "uncached_mid")
    batch = create_source(source, "muon", monkeypatch, capture_mid=mid)
    selected = source / ("checkpoints/mid_epoch2" if mid else "checkpoints/1")
    env = dict(environment(source, output), RESUME_CHECKPOINT=str(selected), EPOCHS="4")
    if selection == "linked":
        alias = source / "checkpoints/chosen_alias"
        alias.symlink_to("1", target_is_directory=True)
        env["RESUME_CHECKPOINT"] = str(alias)
    if selection == "copied":
        selected = tmp_path / "copied_checkpoint"
        shutil.copytree(source / "checkpoints/1", selected)
        env.update(RESUME_CHECKPOINT=str(selected), RESUME_RUN_DIR=str(source))
    if selection == "uncached_mid":
        shutil.rmtree(source / "validation_inputs")
        (source / "checkpoints/train_command.txt").write_text("# World size: 1\n")
    before = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    helper("prepare_continuation").prepare(env)
    with monkeypatch.context() as m:
        trainer, record = resumed_trainer(output, batch, m, fresh_validation=selection == "uncached_mid")
        assert record["lr_policy"] == "restore"
        assert trainer.current_epoch == 2 and trainer.global_step == (5 if mid else 4)
        assert trainer._resume_local_step == (1 if mid else 0)
        assert_saved_optimizer_loaded(trainer.optimizers, selected)
        states = torch.load(selected / "scheduler_state_dict.pt", map_location="cpu", weights_only=True)
        for sch, opt, state in zip(trainer.schedulers, trainer.optimizers, states, strict=True):
            assert sch.state_dict() == state
            assert [group["lr"] for group in opt.param_groups] == state["_last_lr"]
        trainer.run_training()
        assert trainer.global_step == 8
        assert trainer.train_loader.epochs == [2, 3]
        assert trainer.train_loader.yielded == ([(2, 1)] if mid else [(2, 0), (2, 1)]) + [(3, 0), (3, 1)]
        # Reconstructed LR functions agree after further training, not just at load.
        expected = torch.load(source / "checkpoints/3/scheduler_state_dict.pt", map_location="cpu", weights_only=True)
        for sch, state in zip(trainer.schedulers, expected, strict=True):
            assert sch.state_dict() == state
    assert before == {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    if mid:
        assert sha256(output / "checkpoints/resume_source/model.safetensors") == sha256(selected / "model.safetensors")
    assert (output / "checkpoints/3/resume_context.json").is_file()
    with pytest.raises(ValueError, match="extend beyond"):
        helper("prepare_continuation").prepare(dict(env, OUTPUT_DIR=str(tmp_path / "bad_lr"),
            EPOCHS="10", RESUME_LR_POLICY="restore"))


def test_resume_from_an_earlier_checkpoint_of_a_resumed_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/dflash_prefix_joint"))
    source, first, second = tmp_path / "source", tmp_path / "first", tmp_path / "second"
    batch = create_source(source, "adamw", monkeypatch)
    prep = helper("prepare_continuation")
    prep.prepare(environment(source, first))
    with monkeypatch.context() as m:
        trainer, record = resumed_trainer(first, batch, m)
        assert record["lr_policy"] == "restart"
        trainer.run_training()
    prep.prepare(dict(environment(first, second), RESUME_CHECKPOINT=str(first / "checkpoints/5"), EPOCHS="8"))
    with monkeypatch.context() as m:
        trainer, record = resumed_trainer(second, batch, m)
        assert record["lr_policy"] == "restore" and trainer.global_step == 12
        assert record["source_scheduler_spec"]["origin_step"] == 10
        trainer.run_training()
        assert trainer.global_step == 16 and trainer.train_loader.epochs == [6, 7]
        expected = torch.load(first / "checkpoints/7/scheduler_state_dict.pt", map_location="cpu", weights_only=True)
        assert [s.state_dict() for s in trainer.schedulers] == expected


def test_no_implicit_latest_and_incomplete_exports_rejected(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/dflash_prefix_joint"))
    source, output = tmp_path / "source", tmp_path / "new"
    create_source(source, "adamw", monkeypatch)
    env = environment(source, output)
    missing = dict(env)
    del missing["RESUME_CHECKPOINT"]
    with pytest.raises(ValueError, match="no checkpoint is chosen"):
        helper("prepare_continuation").prepare(missing)
    (source / "checkpoints/4/training_state.json").unlink()
    with pytest.raises(ValueError, match="legacy interrupted"):
        helper("prepare_continuation").prepare(env)
    assert not output.exists()


def test_epochs_is_the_total_target_and_rejects_completed_budget(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/dflash_prefix_joint"))
    source = tmp_path / "source"
    create_source(source, "adamw", monkeypatch)
    prep = helper("prepare_continuation")
    for value in ("0", "-1", "5", "3", "not-an-int"):
        out = tmp_path / ("invalid_" + value)
        with pytest.raises(ValueError, match="EPOCHS"):
            prep.prepare(dict(environment(source, out), EPOCHS=value))
        assert not out.exists()
    env = environment(source, tmp_path / "missing")
    del env["EPOCHS"]
    with pytest.raises(ValueError, match="EPOCHS"):
        prep.prepare(env)
    out = tmp_path / "six_total"
    prep.prepare(dict(environment(source, out), EPOCHS="6", CHECK_ONLY="1", ADDITIONAL_EPOCHS="99"))
    record = json.loads((out / "continuation.json").read_text())
    assert record["start_epoch_index"] == 5 and record["total_epochs"] == 6
    assert record["remaining_epoch_completions"] == 1
    assert "additional_epochs" not in record
    assert TrainConfig.resolve(["--config", str(out / "train_config.yaml")]).flatten()["epochs"] == 6
