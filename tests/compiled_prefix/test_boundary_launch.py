import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from speculators.train.config import TrainConfig
from speculators.train.config.schema import CONFIG_DESTS, nest_flat

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "scripts/dflash_prefix_boundary" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("boundary_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    checkpoint = tmp_path / "run/checkpoints/3"
    checkpoint.mkdir(parents=True)
    cfg = {"speculators_model_type": "dflash_prefix", "prefix_selector_kind": "compiled_prefix_v3",
           "block_size": 16, "prefix_top_k": 16, "prefix_rank": 256,
           "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33], "mask_token_id": 151669,
           "transformer_layer_config": {"hidden_size": 2560, "vocab_size": 151936, "num_hidden_layers": 5}}
    (checkpoint / "config.json").write_text(json.dumps(cfg))
    (checkpoint / "model.safetensors").write_bytes(b"metadata test only")
    target, data = tmp_path / "target", tmp_path / "dataset"
    target.mkdir()
    data.mkdir()
    (target / "config.json").write_text(json.dumps(dict(hidden_size=2560, vocab_size=151936, num_hidden_layers=36)))
    config = TrainConfig.resolve(["--verifier-name-or-path", str(target), "--data-path", str(data)])
    nested = nest_flat({k: v for k, v in config.flatten().items() if k in CONFIG_DESTS})
    nested["backend"] = dict(config.backend_args)
    (checkpoint.parent / "run.yaml").write_text(yaml.safe_dump({"train": nested}))
    return dict(INIT_CHECKPOINT=str(checkpoint), MODEL="", DATASET="", OUTPUT_DIR=str(tmp_path / "out"),
                VLLM_NPUS="0,1", VLLM_DP="2", TRAIN_NPUS="2,3,4,5,6", NUM_TRAIN_NPUS="5",
                VLLM_PORT="8092", VLLM_RPC_PORT="29692", PILOT_STEPS="21", EVAL_EVERY="10")


@pytest.mark.parametrize("objective", ["dual_boundary", "boundary", "control"])
def test_real_config_roundtrip_paths_ablations_and_no_resume(tmp_path, objective):
    env = fixture(tmp_path)
    module = load("prepare_experiment")
    path = module.prepare(dict(env, OBJECTIVE=objective))
    cfg = TrainConfig.resolve(["--config", str(path)]).flatten()
    settings = json.loads((path.parent / "settings.json").read_text())
    assert cfg["data_path"] == str(tmp_path / "dataset")
    assert cfg["epochs"] == 3 and cfg["max_steps"] is None
    assert cfg["scheduler_total_steps"] == 21 and cfg["lr"] == 3e-5
    assert cfg["no_resume_from_checkpoint"] and not cfg["prefix_freeze_backbone"]
    assert settings["objective"] == objective
    assert settings["backbone_lr"] == 3e-6
    assert cfg["noise_std"] == 0.0
    assert settings["train_last_layers"] == "all"
    assert settings["boundary_weight"] == .5
    assert settings["support_weight"] == .25
    assert settings["full_ce_weight"] == 1.0
    assert (path.parent / "model_path.txt").read_text().strip() == str(tmp_path / "target")
    assert json.loads((path.parent / "experiment.json").read_text())["expected_checkpoint_steps"] == {"0": 10, "1": 20, "2": 21}
    with pytest.raises(ValueError, match="OUTPUT_DIR already exists"):
        module.prepare(env)


def test_relocation_and_invalid_settings_fail_before_output(tmp_path):
    env = fixture(tmp_path)
    module = load("prepare_experiment")
    new_data = tmp_path / "moved_data"
    new_data.mkdir()
    for override in (dict(OBJECTIVE="bad"), dict(HEAD_LR="nan"), dict(TRAIN_NPUS="0,2,3,4,5"),
                     dict(BLOCK_SIZE="8"), dict(ANCHOR_CHUNK_SIZE="64")):
        with pytest.raises(ValueError):
            module.prepare(dict(env, **override))
        assert not Path(env["OUTPUT_DIR"]).exists()
    path = module.prepare(dict(env, DATASET=str(new_data)))
    assert TrainConfig.resolve(["--config", str(path)]).flatten()["data_path"] == str(new_data)


def test_launcher_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "examples/train/train_dflash_prefix_boundary.sh")], check=True)


def test_windows_do_not_replay_batches():
    module = load("train_entry")
    class Sampler:
        def set_epoch(self, epoch):
            pass
    class Loader:
        batch_sampler = Sampler()
        def __len__(self):
            return 6
        def __iter__(self):
            return iter(range(6))
    loader = module.WindowLoader(Loader(), 9, 4)
    assert list(loader) == [0, 1, 2, 3]
    assert list(loader) == [4, 5, 0, 1]
    assert list(loader) == [2]
    assert len(loader) == 0


@pytest.mark.parametrize("objective,witness,should_run", [
    ("boundary", .2, True), ("boundary", 0., False), ("control", 0., True),
])
def test_initial_validation_gate_and_checkpoint_links(tmp_path, objective, witness, should_run):
    """An inactive new signal must stop before updates, without a fake checkpoint."""
    from types import SimpleNamespace
    import torch

    entry = load("train_entry")
    called = []

    class Trainer:
        device_type, rank, global_step = "cpu", 0, 0
        checkpointer = SimpleNamespace(path=tmp_path)
        model = torch.nn.Linear(2, 2)

        def val_epoch(self, epoch):
            called.append(("val", epoch))
            torch.rand(3)
            return {"refresh_teachable_frontier_fraction_epoch": witness}

        def run_training(self):
            called.append(("train", 0))
            self.global_step = 500
            (tmp_path / "0").mkdir()
            self.val_epoch(0)

    cli = SimpleNamespace(create_train_val_loaders=lambda **kw: None)
    entry.install_hooks(cli, SimpleNamespace(Trainer=Trainer), {
        "objective": objective, "boundary_weight": 1.,
    })
    rng = torch.random.get_rng_state().clone()
    if should_run:
        Trainer().run_training()
        assert called == [("val", -1), ("train", 0), ("val", 0)]
        assert (tmp_path / "step_500").resolve() == tmp_path / "0"
    else:
        with pytest.raises(ValueError, match="No target-checked"):
            Trainer().run_training()
        assert called == [("val", -1)]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not (tmp_path / "step_0").exists()
    records = [json.loads(x) for x in (tmp_path / "boundary_history.jsonl").read_text().splitlines()]
    assert records[0]["step"] == 0 and records[0]["window"] == -1


@pytest.mark.parametrize("support_witness,should_run", [(0., False), (.2, True)])
def test_dual_boundary_precheck_accepts_support_signal(tmp_path, support_witness, should_run):
    from types import SimpleNamespace
    import torch
    entry = load("train_entry")
    called = []
    class Trainer:
        device_type, rank, global_step = "cpu", 0, 0
        checkpointer = SimpleNamespace(path=tmp_path)
        model = torch.nn.Linear(2, 2)
        def val_epoch(self, epoch):
            return {"support_teachable_miss_fraction_epoch": support_witness,
                    "refresh_teachable_frontier_fraction_epoch": 0.}
        def run_training(self):
            called.append("train")
    entry.install_hooks(SimpleNamespace(create_train_val_loaders=lambda **kw: None),
                        SimpleNamespace(Trainer=Trainer), {
                            "objective": "dual_boundary", "boundary_weight": .5,
                            "support_weight": .25,
                        })
    if should_run:
        Trainer().run_training()
        assert called == ["train"]
    else:
        with pytest.raises(ValueError, match="No target-checked"):
            Trainer().run_training()
        assert not called
