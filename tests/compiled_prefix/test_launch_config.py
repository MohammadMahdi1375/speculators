import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "scripts/dflash_prefix_compiled" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher(tmp_path, kind="local_prefix_v2"):
    path = tmp_path / "run/checkpoints/0"
    path.mkdir(parents=True)
    cfg = {"speculators_model_type": "dflash_prefix", "prefix_selector_kind": kind,
           "block_size": 16, "prefix_top_k": 16, "prefix_rank": 256,
           "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33], "mask_token_id": 151669,
           "transformer_layer_config": {"hidden_size": 2560, "vocab_size": 151936}}
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "model.safetensors").write_bytes(b"placeholder; metadata test only")
    return path


def test_teacher_validation_and_recipe_discovery(tmp_path):
    module = load("prepare_experiment")
    path = teacher(tmp_path)
    assert module.load_teacher(path)[1]["block_size"] == 16
    recipe = path.parent.parent / "train_config.yaml"
    recipe.write_text("train: {}")
    assert module.find_base_config(path) == recipe
    with pytest.raises(ValueError, match="trained local_prefix_v2"):
        module.load_teacher(teacher(tmp_path / "bad", "compiled_prefix_v3"))


def test_recipe_missing_is_actionable(tmp_path):
    module = load("prepare_experiment")
    with pytest.raises(ValueError, match="BASE_TRAIN_CONFIG"):
        module.find_base_config(teacher(tmp_path))


def test_npu_allocation(tmp_path):
    module = load("prepare_experiment")
    assert module.allocation("0,1", "2,3,4,5,6", 2, 5) == ([0, 1], [2, 3, 4, 5, 6])
    for args in (("0,1", "1,2", 2, 2), ("0,0", "1,2", 2, 2), ("0,1", "2,3", 1, 2)):
        with pytest.raises(ValueError):
            module.allocation(*args)


class Sampler:
    def __init__(self):
        self.epochs = []
    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class Loader:
    def __init__(self):
        self.batch_sampler = Sampler()
    def __len__(self):
        return 6
    def __iter__(self):
        return iter(range(6))


def test_training_windows_continue_and_end_exactly():
    module = load("train_entry")
    base = Loader()
    wrapped = module.WindowLoader(base, 9, 4)
    assert list(wrapped) == [0, 1, 2, 3]
    assert list(wrapped) == [4, 5, 0, 1]
    assert len(wrapped) == 1
    assert list(wrapped) == [2]
    assert len(wrapped) == 0
    assert base.batch_sampler.epochs == [0, 1]


def test_validation_repeats_same_prefix():
    module = load("train_entry")
    wrapped = module.ValidationLoader(Loader(), 3)
    assert list(wrapped) == list(wrapped) == [0, 1, 2]


def test_serving_rejects_teacher_and_infers_proposals(tmp_path):
    module = load("check_checkpoint")
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({"hidden_size": 2560, "vocab_size": 151936}))
    old = teacher(tmp_path)
    with pytest.raises(ValueError, match="compiled_prefix_v3"):
        module.speculative_config(old, target)
    compiled = teacher(tmp_path / "compiled", "compiled_prefix_v3")
    config = module.speculative_config(compiled, target)
    assert config["num_speculative_tokens"] == 15
    assert config["enable_adaptive_verification"] is False
    with pytest.raises(ValueError, match="BLOCK_SIZE"):
        module.speculative_config(compiled, target, 8)


def test_shell_syntax_and_validate_default():
    scripts = [ROOT / "examples/train/train_dflash_prefix_compiled.sh", ROOT / "examples/serve_dflash_prefix_compiled.sh"]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    serving = scripts[1].read_text()
    assert 'MODE="${MODE:-validate}"' in serving
    assert 'serve "$TARGET_MODEL"' in serving
    assert '--max-num-seqs 1' in serving


def test_real_schema_roundtrip_and_fresh_output(tmp_path):
    from speculators.train.config import TrainConfig
    from speculators.train.config.schema import CONFIG_DESTS, nest_flat
    import yaml
    module = load("prepare_experiment")
    checkpoint = teacher(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({"hidden_size": 2560, "vocab_size": 151936, "num_hidden_layers": 36}))
    data = tmp_path / "dataset"
    data.mkdir()
    base = TrainConfig.resolve(["--verifier-name-or-path", "/old/server/target", "--data-path", "/old/server/data"])
    nested = nest_flat({k: v for k, v in base.flatten().items() if k in CONFIG_DESTS})
    nested["backend"] = dict(base.backend_args)
    recipe = checkpoint.parent / "run.yaml"
    recipe.write_text(yaml.safe_dump({"train": nested}))
    output = tmp_path / "new_pilot"
    env = {"TEACHER_CHECKPOINT": str(checkpoint), "MODEL": str(target) + "/", "DATASET": str(data),
           "OUTPUT_DIR": str(output), "VLLM_NPUS": "0,1", "VLLM_DP": "2", "TRAIN_NPUS": "2,3,4,5,6",
           "NUM_TRAIN_NPUS": "5", "VLLM_PORT": "8092", "VLLM_RPC_PORT": "29692", "PILOT_STEPS": "21", "EVAL_EVERY": "10"}
    cfg_path = module.prepare(env)
    cfg = TrainConfig.resolve(["--config", str(cfg_path)]).flatten()
    settings = json.loads((output / "settings.json").read_text())
    assert cfg["data_path"] == str(data)
    assert cfg["verifier_name_or_path"] == str(target)
    assert cfg["max_steps"] is None and cfg["epochs"] == 3
    assert cfg["scheduler_total_steps"] == 21
    assert int((output / "server_max_model_len.txt").read_text()) == cfg["total_seq_len"] + 256
    assert settings["cap"] == 8 and settings["kd_weight"] == .5
    assert settings["history_weight"] == .1 and settings["intervention_fraction"] == .25
    assert cfg["draft_attn_impl"] == settings["draft_attn_impl"] == "sdpa"
    assert json.loads((output / "experiment.json").read_text())["expected_checkpoint_steps"] == {"0": 10, "1": 20, "2": 21}
    with pytest.raises(ValueError, match="OUTPUT_DIR already exists"):
        module.prepare(env)
    invalid_output = tmp_path / "invalid_pilot"
    for override, error in (({"ANCHOR_CHUNK_SIZE": "65"}, "anchor chunk size"),
                            ({"ANCHOR_CHUNK_SIZE": "0"}, "anchor chunk size"),
                            ({"DRAFT_ATTN_IMPL": "simple_flex_attention"}, "DRAFT_ATTN_IMPL")):
        with pytest.raises(ValueError, match=error):
            module.prepare(dict(env, OUTPUT_DIR=str(invalid_output), **override))
        assert not invalid_output.exists()
    eager_output = tmp_path / "eager_pilot"
    eager_cfg = module.prepare(dict(env, OUTPUT_DIR=str(eager_output), DRAFT_ATTN_IMPL="eager"))
    assert TrainConfig.resolve(["--config", str(eager_cfg)]).flatten()["draft_attn_impl"] == "eager"
    assert json.loads((eager_output / "settings.json").read_text())["draft_attn_impl"] == "eager"
