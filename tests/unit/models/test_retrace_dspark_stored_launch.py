"""Regression coverage for repository-root shadowing of editable vLLM installs."""

import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from speculators.models.retrace_dspark.stored_launch import (
    IMPORT_CHECK,
    source_environment,
)

EDITABLE_FINDER = r"""
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])

class EditableFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            source = root / "vllm" / fullname.replace(".", "/")
            filename = (
                source / "__init__.py" if source.is_dir() else source.with_suffix(".py")
            )
            if filename.is_file():
                return importlib.util.spec_from_file_location(fullname, filename)

sys.meta_path.append(EditableFinder)
"""


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "workspace"
    files = {
        "vllm/vllm/__init__.py": "from .sampling_params import SamplingParams\n",
        "vllm/vllm/sampling_params.py": "class SamplingParams: pass\n",
        "vllm/vllm/engine/__init__.py": "",
        "vllm/vllm/engine/arg_utils.py": "class EngineArgs: pass\n",
        "vllm-ascend/vllm_ascend/__init__.py": "",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "speculators/src").mkdir(parents=True)
    return root


def test_repository_root_can_hide_top_level_editable_exports(checkout):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout / "speculators/src")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            EDITABLE_FINDER
            + "\nimport vllm.sampling_params\nfrom vllm import SamplingParams\n",
            str(checkout),
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert (
        "cannot import name 'SamplingParams' from 'vllm' (unknown location)"
        in result.stderr
    )


@pytest.mark.parametrize("working_directory", [".", "speculators"])
def test_explicit_sources_load_regular_package_and_pass_preflight(
    checkout, working_directory
):
    env = source_environment(checkout, os.environ, ["8", "9"])
    result = subprocess.run(
        [sys.executable, "-S", "-c", EDITABLE_FINDER + IMPORT_CHECK, str(checkout)],
        cwd=checkout / working_directory,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Target import check passed:" in result.stdout
    assert str(checkout / "vllm/vllm/__init__.py") in result.stdout


def test_launch_environment_keeps_runtime_workarounds_and_separates_devices(checkout):
    inherited = {
        "PYTHONPATH": "/existing/python/path",
        "LD_PRELOAD": "/conda/lib/libstdc++.so.6",
        "LD_LIBRARY_PATH": "/conda/lib:/cann/lib64",
        "NO_PROXY": "example.internal",
        "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
    }
    before = inherited.copy()
    server = source_environment(checkout, inherited, ["8", "9"])
    trainer = source_environment(
        checkout, inherited, ["10", "11", "12", "13", "14", "15"]
    )
    assert inherited == before
    assert server["ASCEND_RT_VISIBLE_DEVICES"] == "8,9"
    assert trainer["ASCEND_RT_VISIBLE_DEVICES"] == "10,11,12,13,14,15"
    assert server["LD_PRELOAD"] == inherited["LD_PRELOAD"]
    assert server["LD_LIBRARY_PATH"] == inherited["LD_LIBRARY_PATH"]
    assert server["PYTHONPATH"].endswith(os.pathsep + inherited["PYTHONPATH"])
    assert server["NO_PROXY"] == "example.internal,127.0.0.1,localhost"
    assert server["no_proxy"] == "127.0.0.1,localhost"


def test_full_bash_locks_checkpoint_epoch_count_and_device_partition(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples/train/train_retrace_dspark_stored_full.sh"
    )
    env = dict(
        os.environ,
        RETRACE_DSPARK_FULL_OUTPUT=str(tmp_path / "full"),
        RETRACE_STORED_OUTPUT="/wrong/dflash/smoke",
        RETRACE_DSPARK_STORED_OUTPUT="/wrong/dspark/smoke",
    )
    result = subprocess.run(
        ["bash", str(script), "--print-command"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stdout)

    def value(flag):
        return command[command.index(flag) + 1]

    assert (
        value("--base-dspark")
        == "/home/n84449292/m84379596/Huggingface/Qwen3-4B-DSpark-block7"
    )
    assert value("--epochs") == "8"
    assert value("--max-records") == "40000"
    assert value("--server-npus") == "8,9"
    assert value("--trainer-npus") == "10,11,12,13,14,15"
    assert value("--port") == "8523"
    assert value("--output") == str(tmp_path / "full")
    assert "/DFlash/vLLM_NPU_spec_main/speculators/examples/train/" in command[1]
    assert "--smoke" not in command


def test_launcher_constructs_target_and_trainer_for_dspark_only(tmp_path, monkeypatch):
    from speculators.models.retrace_dspark import stored_launch as launch

    root, target, base, data, output = [
        tmp_path / name for name in ("root", "target", "base", "data", "output")
    ]
    for path in (
        root / "speculators/scripts/launch_vllm.py",
        target / "config.json",
        base / "config.json",
        data / "state.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    args = [
        "stored_launch",
        "--root",
        str(root),
        "--target",
        str(target),
        "--base-dspark",
        str(base),
        "--data",
        str(data),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    # Only process orchestration is mocked: no device or socket access occurs.
    config = SimpleNamespace(
        block_size=7,
        target_vocab_size=151936,
        aux_hidden_state_layer_ids=[2, 10, 18, 26, 34],
        transformer_layer_config=SimpleNamespace(hidden_size=2560),
    )
    monkeypatch.setattr(
        launch.ReTraceDSparkSpeculatorConfig, "from_pretrained", lambda *a: config
    )
    monkeypatch.setattr(launch, "prepare_weights", lambda *a: None)
    monkeypatch.setattr(launch, "prepare_data", lambda *a, **kw: {"total_tokens": 100})
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: None)

    class PortCheck:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def bind(self, *a):
            pass

    monkeypatch.setattr(launch.socket, "socket", lambda *a: PortCheck())

    class Opener:
        def open(self, *a, **kw):
            return io.BytesIO(b'{"data":[{"id":"retrace-dspark-stored-target"}]}')

    monkeypatch.setattr(launch.urllib.request, "build_opener", lambda *a: Opener())
    commands, stopped = [], []

    class Child:
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr(launch.subprocess, "Popen", Child)
    monkeypatch.setattr(launch, "stop_group", lambda child: stopped.append(child))
    monkeypatch.setattr(launch.signal, "signal", lambda *a: None)
    launch.main()
    server, trainer = commands
    assert server[1]["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "8,9"
    assert trainer[1]["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "10,11,12,13,14,15"
    assert server[0][server[0].index("--block-size") + 1] == "128"
    assert server[0][server[0].index("--port") + 1] == "8523"
    assert "--provenance-dir" in server[0]
    assert "speculators.models.retrace_dspark.stored_train" in trainer[0]
    assert trainer[0][trainer[0].index("--nproc_per_node") + 1] == "6"
    run = json.loads((output / "stored_run.json").read_text())
    assert run["epochs"] == 8 and run["max_records"] == 40000
    assert run["initial_checkpoint"].endswith("initial/retrace_dspark")
    assert run["protocol"] == "dspark_stored_pair_v1"
    assert not run["table2_reproduction_validated"]
    assert (
        "train_retrace_dspark_stored_vllm.sh"
        in (output / "resume_stored.sh").read_text()
    )
    assert len(stopped) == 2
    # A resume with changed settings is rejected before starting any process.
    monkeypatch.setattr(sys, "argv", args + ["--resume", "--epochs", "1"])
    with pytest.raises(SystemExit):
        launch.main()
    assert len(commands) == 2
