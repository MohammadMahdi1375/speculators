"""Protocol/lifecycle tests for the diagnostic client; no NPU timing claims."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error
from unittest.mock import MagicMock
import zipfile

import pytest


def load_client():
    path = Path(__file__).resolve().parents[2] / "scripts/dflash_prefix_compiled/trace_server.py"
    spec = importlib.util.spec_from_file_location("pipeline_trace_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_routes_accept_empty_200_responses():
    client = load_client()
    opener = MagicMock()
    response = opener.open.return_value.__enter__.return_value
    response.read.return_value = b""
    client.profile_control(opener, "http://127.0.0.1:12345", "start_profile")
    request = opener.open.call_args.args[0]
    assert request.full_url.endswith("/start_profile") and request.method == "POST"
    response.read.assert_called_once()
    with pytest.raises(ValueError):
        client.profile_control(opener, "http://127.0.0.1:12345", "v1/completions")


def test_profile_http_error_includes_diagnostic():
    client = load_client()
    opener = MagicMock()
    opener.open.side_effect = urllib.error.HTTPError("url", 404, "not found", {}, io.BytesIO(b"no route"))
    with pytest.raises(RuntimeError, match="start_profile: HTTP 404: no route"):
        client.profile_control(opener, "http://127.0.0.1:12345", "start_profile")


def test_archiver_selects_analyzed_data_and_requires_kernel_rows(tmp_path):
    client = load_client()
    (tmp_path / "raw.bin").write_bytes(b"raw")
    (tmp_path / "trace_view.json").write_text('{"traceEvents": []}')
    kernel = tmp_path / "kernel_details.csv"
    kernel.write_text("Name,Duration(us)\n")
    (tmp_path / "unsafe.csv").symlink_to(kernel)
    files = client.readable_trace_files(tmp_path)
    assert {p.name for p in files} == {"trace_view.json", "kernel_details.csv"}
    assert not client.has_kernel_rows(files)
    kernel.write_text("Name,Duration(us)\nMatMul,3.0\n")
    assert client.has_kernel_rows(files)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    client = load_client()
    helpers = client.load_helpers()
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({"prefix_selector_kind": "compiled_prefix_v3", "block_size": 16}))
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text('{"model_type": "qwen3"}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "thinking": False, "target_config_sha256": helpers.digest(target / "config.json"),
        "datasets": {"gsm8k": {"results": [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3, 4]}]}}}))
    args = SimpleNamespace(spec_main=tmp_path, draft=draft, target=target, manifest=manifest,
                           output=tmp_path / "output", npu="6", port=8213, steps=12, model="qwen3-4b-dflash")
    events = []
    launched = {}
    process = SimpleNamespace(pid=12345, poll=lambda: None)
    def popen(command, **kwargs):
        launched.update(command=command, **kwargs)
        events.append("launch")
        return process
    def completion(opener, base_url, model, ids, tokens):
        if tokens == 512:
            events.append("warmup")
            with (args.output / "server.log").open("a") as stream:
                stream.write("COMPILED_PREFIX_PARITY_PASS checked_blocks=32; steady-state fast path enabled\n")
        else:
            events.append("probe")
            traces = args.output / "traces" / "worker_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
            traces.mkdir(parents=True)
            (traces / "kernel_details.csv").write_text("Name,Duration(us)\nMatMul,3.0\n")
            (traces / "trace_view.json").write_text('{"traceEvents": [{"name": "MatMul", "dur": 3}]}')
        return {"completion_tokens": tokens, "diagnostic_only": True}
    monkeypatch.setattr(client, "load_helpers", lambda: helpers)
    monkeypatch.setattr(client, "source_digests", lambda root, helpers: {"fixture": "abc"})
    monkeypatch.setattr(client.socket, "socket", MagicMock())
    monkeypatch.setattr(client.subprocess, "Popen", popen)
    monkeypatch.setattr(helpers, "read_json", lambda *a, **kw: {"data": [{"id": args.model}]})
    monkeypatch.setattr(helpers, "completion", completion)
    monkeypatch.setattr(helpers, "stop_owned_server", lambda proc: events.append("cleanup_12345"))
    monkeypatch.setattr(client, "profile_control", lambda opener, url, action: events.append(action))
    monkeypatch.setattr(client.time, "sleep", lambda delay: None)
    return client, helpers, args, events, launched


def test_full_protocol_warms_before_trace_and_collects_before_cleanup(fixture):
    client, _, args, events, launched = fixture
    assert client.run(args) == 0
    assert events == ["launch", "warmup", "start_profile", "probe", "stop_profile", "cleanup_12345"]
    profile = json.loads(launched["command"][-1])
    assert profile["max_iterations"] == 12 and profile["delay_iterations"] == 2
    assert profile["ignore_frontend"] and not profile["torch_profiler_with_stack"]
    assert launched["start_new_session"]
    assert launched["env"]["SELECTOR_PROFILE_STEPS"] == "0"
    assert launched["env"]["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] == "1"
    assert "DFLASH_PREFIX_PROFILE_DIR" not in launched["env"]
    record = json.loads((args.output / "run.json").read_text())
    assert record["complete"] and record["production_fast_gate_passed"]
    assert len(record["trace_files"]) == 2 and record["diagnostic_only"]
    with zipfile.ZipFile(args.output.with_suffix(".zip")) as stream:
        assert "run.json" in stream.namelist()
        assert any(name.endswith("kernel_details.csv") for name in stream.namelist())


def test_failed_start_still_stops_profiler_and_only_owned_server(fixture, monkeypatch):
    client, _, args, events, _ = fixture
    def control(opener, url, action):
        events.append(action)
        if action == "start_profile":
            raise RuntimeError("HTTP connection lost")
    monkeypatch.setattr(client, "profile_control", control)
    assert client.run(args) == 1
    assert events == ["launch", "warmup", "start_profile", "stop_profile", "cleanup_12345"]
    record = json.loads((args.output / "run.json").read_text())
    assert not record["complete"] and "HTTP connection lost" in record["error"]
    assert args.output.with_suffix(".zip").is_file()


def test_busy_port_refuses_to_launch_or_create_output(fixture, monkeypatch):
    client, _, args, events, _ = fixture
    sock = MagicMock()
    sock.__enter__.return_value.bind.side_effect = OSError("already occupied")
    monkeypatch.setattr(client.socket, "socket", lambda: sock)
    with pytest.raises(ValueError, match="busy"):
        client.run(args)
    assert not events and not args.output.exists()


def test_missing_parity_never_starts_profiler(fixture, monkeypatch):
    client, helpers, args, events, _ = fixture
    monkeypatch.setattr(helpers, "completion", lambda *a: {"completion_tokens": 512})
    assert client.run(args) == 1
    assert events == ["launch", "cleanup_12345"]
    assert "Production parity gate" in json.loads((args.output / "run.json").read_text())["error"]


def test_existing_output_is_not_overwritten(fixture):
    client, _, args, events, _ = fixture
    args.output.mkdir()
    (args.output / "run.json").write_text("existing")
    with pytest.raises(FileExistsError):
        client.run(args)
    assert not events and (args.output / "run.json").read_text() == "existing"
