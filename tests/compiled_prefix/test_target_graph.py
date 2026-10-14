"""Launcher/probe contracts on CPU; these do not establish NPU graph parity."""
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import urllib.request
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "scripts/dflash_prefix_compiled" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_args(tmp_path, kind="compiled_prefix_v3", block=16):
    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir(parents=True)
    draft.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "hidden_size": 2560, "vocab_size": 151936,
    }))
    (draft / "config.json").write_text(json.dumps({
        "speculators_model_type": "dflash" if kind == "dflash" else "dflash_prefix",
        "prefix_selector_kind": kind, "prefix_top_k": 16, "block_size": block,
        "transformer_layer_config": {"hidden_size": 2560, "vocab_size": 151936},
    }))
    (draft / "model.safetensors").write_bytes(b"fixture metadata only")
    return SimpleNamespace(spec_main=ROOT.parent, target=target, draft=draft,
                           npu="6", port=8213, target_execution="graph", mode="fast",
                           served_model_name="test-dflash", max_model_len=32768,
                           max_batched_tokens=32768, memory_utilization=.96,
                           graph_metrics=0, check_only=1)


@pytest.mark.parametrize("kind", ["compiled_prefix_v3", "local_prefix_v2", "dflash"])
@pytest.mark.parametrize("block", [8, 16])
def test_graph_enables_only_target_and_keeps_full_trained_proposals(tmp_path, kind, block):
    module = load("target_graph_server")
    args = fixture_args(tmp_path, kind, block)
    before = (args.draft / "config.json").read_bytes()
    _, cmd, env, record = module.command_and_record(args)
    assert cmd[cmd.index("serve") + 1] == str(args.target)
    assert "--enforce-eager" not in cmd
    config = json.loads(cmd[cmd.index("--compilation-config") + 1])
    assert config == {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY", "custom_ops": ["all"],
                      "cudagraph_capture_sizes": [block], "max_cudagraph_capture_size": block}
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec["enforce_eager"] is True and spec["num_speculative_tokens"] == block - 1
    assert spec["enable_adaptive_verification"] is False
    assert record["proposals"] == block - 1
    assert (args.draft / "config.json").read_bytes() == before
    assert env["DFLASH_PREFIX_INFERENCE"] == ("reference" if kind == "dflash" else "fast")


def test_eager_control_differs_only_in_target_graph_settings(tmp_path):
    module = load("target_graph_server")
    args = fixture_args(tmp_path)
    _, _, graph_env, graph = module.command_and_record(args)
    args.target_execution = "eager"
    _, cmd, eager_env, eager = module.command_and_record(args)
    assert "--enforce-eager" in cmd
    assert graph_env == eager_env
    assert eager["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE", "custom_ops": ["all"]}
    for key in ("additional_config", "speculative_config", "source_sha256", "selector_environment"):
        assert eager[key] == graph[key]


def test_inherited_audit_and_fusion_cannot_leak_into_compiled_serving():
    module = load("target_graph_server")
    env = module.serving_environment({"KEEP": "value", "DFLASH_PREFIX_FUSION": "local_unpaired",
                                      "DFLASH_PREFIX_SELECTOR_PROFILE_STEPS": "8",
                                      "DFLASH_PREFIX_PROFILE_DIR": "/old/report",
                                      "TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1"}, "compiled_prefix_v3", "fast")
    assert env["KEEP"] == "value"
    assert env["DFLASH_PREFIX_FUSION"] == "off"
    assert env["DFLASH_PREFIX_SELECTOR_PROFILE_STEPS"] == "0"
    assert "DFLASH_PREFIX_PROFILE_DIR" not in env
    assert "TORCH_COMPILE_DISABLE" not in env and "TORCHDYNAMO_DISABLE" not in env
    local = module.serving_environment({}, "local_prefix_v2", "fast")
    assert local["DFLASH_PREFIX_FUSION"] == "local_unpaired"
    assert local["DFLASH_PREFIX_HISTORY_GROUP_SIZE"] == "1"
    assert local["DFLASH_PREFIX_TOKEN_CACHE"] == "0"


def test_config_activates_graph_in_the_actual_installed_runner_predicate(tmp_path):
    # Execute the small real predicate without importing the NPU model runner.
    # This catches mode=0 with FULL_DECODE_ONLY, which the Ascend v1 runner
    # does not treat as enabled despite a superficially plausible CLI command.
    path = ROOT.parent / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_use_aclgraph")
    namespace = {"CompilationMode": SimpleNamespace(VLLM_COMPILE=3),
                 "CUDAGraphMode": SimpleNamespace(NONE="NONE"),
                 "breakable_cudagraph": SimpleNamespace(is_breakable_cudagraph_enabled=lambda: False)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    module = load("target_graph_server")
    for execution in ("graph", "eager"):
        cfg, _ = module.execution_config(execution, 16)
        runner = SimpleNamespace(compilation_config=SimpleNamespace(**cfg),
                                 model_config=SimpleNamespace(enforce_eager=execution == "eager"))
        assert namespace["_use_aclgraph"](runner) is (execution == "graph")


def test_rejects_wrong_target_and_disabled_selector_before_launch(tmp_path):
    module = load("target_graph_server")
    args = fixture_args(tmp_path)
    with pytest.raises(ValueError, match="original Qwen3"):
        module.checkpoint(args.draft, args.draft)
    config = json.loads((args.draft / "config.json").read_text())
    config["prefix_disable_selector"] = True
    (args.draft / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="enabled"):
        module.command_and_record(args)


def test_probe_fails_on_token_or_acceptance_drift_even_if_other_matches():
    module = load("probe_target_graph")
    reference = {"complete": True, "launch": {"npu": "6"}, "manifest_sha256": "same",
                 "tokens": 512, "rows": [{"prompt_sha256": "same", "output_token_ids": [1, 2],
                                          "finish_reason": "length", "speculative": {"num_spec_steps": 1}}]}
    assert module.compare(reference, copy.deepcopy(reference))["pass"]
    altered = copy.deepcopy(reference)
    altered["rows"][0]["output_token_ids"][-1] = 3
    assert not module.compare(reference, altered)["pass"]
    altered = copy.deepcopy(reference)
    altered["rows"][0]["speculative"]["num_spec_steps"] = 2
    comparison = module.compare(reference, altered)
    assert comparison["output_tokens_equal"] and not comparison["acceptance_counters_equal"]
    assert not comparison["pass"]


def test_probe_rejects_metadata_and_incomplete_http_outputs():
    module = load("probe_target_graph")
    response = {"choices": [{"token_ids": [1, 2], "finish_reason": "length"}],
                "usage": {"completion_tokens": 2}, "metrics": {"speculative_decoding": {
                    "num_spec_steps": 1, "num_accepted_draft_tokens": 1,
                    "num_draft_tokens": 15, "num_spec_tokens": 15}}}
    assert module.validate_response(response, 15)["output_token_ids"] == [1, 2]
    with pytest.raises(ValueError, match="proposal count"):
        module.validate_response(response, 7)
    response["choices"][0]["token_ids"] = [1]
    with pytest.raises(ValueError, match="complete output"):
        module.validate_response(response, 15)


def test_http_error_keeps_bounded_response_body():
    module = load("probe_target_graph")
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:8213/v1/completions", 500, "Internal Server Error", {},
            io.BytesIO(b'{"error":"worker stream changed"}' + b'x' * 12000),
        )
    with pytest.raises(RuntimeError, match="HTTP 500.*worker stream changed") as error:
        module.read_json(SimpleNamespace(open=fail), "http://127.0.0.1:8213/v1/completions", {})
    assert len(str(error.value)) < 8400


def test_first_warmup_failure_still_writes_probe_report(tmp_path, monkeypatch):
    module = load("probe_target_graph")
    launch = tmp_path / "launch.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "probe.json"
    launch.write_text(json.dumps({"check_only": False, "port": 8213,
                                  "served_model_name": "qwen3-4b-dflash", "proposals": 15}))
    manifest.write_text("{}")
    monkeypatch.setattr(module, "prompts", lambda *args: [[1, 2, 3]])
    def request(opener, url, payload=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "qwen3-4b-dflash"}]}
        raise RuntimeError("HTTP 500: worker stream changed")
    monkeypatch.setattr(module, "read_json", request)
    monkeypatch.setattr(sys, "argv", ["probe", "--launch-record", str(launch),
                                      "--manifest", str(manifest), "--output", str(output)])
    with pytest.raises(RuntimeError, match="HTTP 500"):
        module.main()
    report = json.loads(output.read_text())
    assert report["complete"] is False and report["stage"] == "warmup"
    assert report["warmup_rounds"] == 0 and report["warmup_request"] == 1
    assert report["rows"] == [] and "worker stream changed" in report["error"]


def test_check_only_shell_prepares_config_without_loading_torch_or_vllm(tmp_path):
    args = fixture_args(tmp_path)
    envfile = tmp_path / "set_env.sh"
    envfile.write_text(":\n")
    output = tmp_path / "launch"
    # If CHECK_ONLY accidentally imports serving libraries this environment is
    # intentionally just the test interpreter plus config files, with no NPU.
    env = dict(os.environ, SPEC_MAIN=str(ROOT.parent), ENV_ROOT=str(Path(sys.executable).parent.parent),
               PYTHON_BIN=sys.executable, CANN_ENV_SH=str(envfile), ATB_ENV_SH=str(envfile),
               DRAFT=str(args.draft), TARGET_MODEL=str(args.target), CHECK_ONLY="1",
               RUN_DIR=str(output), TARGET_EXECUTION="graph", NPU="6")
    command = ["bash", str(ROOT / "examples/serve_dflash_target_graph.sh")]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "No server started" in result.stdout
    record = json.loads((output / "launch.json").read_text())
    assert record["check_only"] and record["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    second = subprocess.run(command, env=env, text=True, capture_output=True)
    assert second.returncode != 0  # Existing run records cannot be overwritten.


def test_dispatch_check_requires_real_full_runtime_rows():
    module = load("check_target_graph")
    assert not module.full_dispatch_observed("requested FULL_DECODE_ONLY capture=[16]", 16)
    assert not module.full_dispatch_observed("| 16 | 16 | 0 | NONE | 100 |", 16)
    assert not module.full_dispatch_observed("| 8 | 8 | 0 | FULL | 100 |", 16)
    assert module.full_dispatch_observed("| 16 | 16 | 0 | FULL | 3 |", 16)


@pytest.mark.parametrize("failure", [None, "probe", "server"])
def test_owned_server_is_stopped_on_success_or_failure(tmp_path, monkeypatch, failure):
    module = load("check_target_graph")
    args = SimpleNamespace(spec_main=ROOT.parent, draft=tmp_path, target=tmp_path,
                           npu="6", port=8213, manifest=tmp_path / "manifest.json",
                           num_prompts=8, tokens=512)
    owned = SimpleNamespace(pid=12345, poll=lambda: 1 if failure == "server" else None)
    stopped = []
    helper = SimpleNamespace(urllib=SimpleNamespace(request=urllib.request),
                             read_json=lambda *a, **kw: {"data": [{"id": "qwen3-4b-dflash"}]},
                             stop_owned_server=lambda process: stopped.append(process.pid))
    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        run = Path(kwargs["env"]["RUN_DIR"])
        run.mkdir()
        (run / "launch.json").write_text(json.dumps({"block_size": 16, "kind": "compiled_prefix_v3"}))
        return owned
    def probe(command, **kwargs):
        if failure == "probe":
            report = Path(command[command.index("--output") + 1])
            report.write_text('{"stage":"warmup","error":"HTTP 500: worker failed"}')
            raise subprocess.CalledProcessError(1, command)
        report = Path(command[command.index("--output") + 1])
        report.write_text('{"comparison":{"pass":true}}')
        with (tmp_path / "graph.log").open("a") as stream:
            stream.write("| 16 | 16 | 0 | FULL | 10 |\n")
            stream.write("COMPILED_PREFIX_PARITY_PASS; steady-state fast path enabled\n")
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.subprocess, "run", probe)
    if failure:
        with pytest.raises(RuntimeError) as error:
            module.run_phase(args, "graph", tmp_path, helper)
        if failure == "probe":
            assert "stage=warmup" in str(error.value) and "graph.log" in str(error.value)
    else:
        assert module.run_phase(args, "graph", tmp_path, helper).is_file()
    assert stopped == [12345]


def test_failed_check_automatically_collects_existing_log_and_saves_path(tmp_path, monkeypatch):
    module = load("check_target_graph")
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output = tmp_path / "check"
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, *args):
            pass
    monkeypatch.setattr(module.socket, "socket", Socket)
    monkeypatch.setattr(module, "helpers", lambda filename="profile_server.py": (
        SimpleNamespace(collect=lambda directory: directory / "failure.zip")
        if filename == "collect_target_graph_failure.py" else SimpleNamespace()
    ))
    def phase(args, phase, output, helper, reference=None):
        if phase == "eager":
            return output / "eager_probe.json"
        (output / "graph.log").write_text("RuntimeError: fixture graph failure")
        raise RuntimeError("graph probe failed at stage=warmup")
    monkeypatch.setattr(module, "run_phase", phase)
    monkeypatch.setattr(sys, "argv", ["check", "--spec-main", str(ROOT.parent),
                                      "--draft", str(draft), "--target", str(target),
                                      "--manifest", str(manifest), "--output", str(output)])
    assert module.main() == 1
    report = json.loads((output / "result.json").read_text())
    assert not report["complete"] and "stage=warmup" in report["error"]
    assert report["diagnostic_zip"] == str(output / "failure.zip")
