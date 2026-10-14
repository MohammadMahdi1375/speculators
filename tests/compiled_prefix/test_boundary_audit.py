import importlib.util
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_repair import compiled_checkpoint
from test_training import tiny_batch
from speculators.models.dflash_prefix.boundary_training import build_boundary_model
from speculators.models.dflash_prefix.boundary_objective import boundary_loss_terms
from speculators.models.dflash_prefix.boundary_validation import exported_precision

# tests/compiled_prefix -> tests -> repository
SCRIPT = Path(__file__).parents[2] / "scripts/dflash_prefix_boundary/audit_checkpoint.py"
spec = importlib.util.spec_from_file_location("boundary_loss_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_first_loss_partition_and_severity_without_double_counting():
    ref = torch.ones(6, 3, dtype=torch.long)
    base = ref.clone()
    student = ref.clone()
    student[0, 1:] = 9
    student[1, 2] = 9
    base[2, 1:] = 9
    student[3, 0] = base[3, 0] = 9
    valid = torch.ones_like(ref, dtype=torch.bool)
    valid[4] = False
    pool = torch.tensor([1, 2]).expand(6, 3, 2).clone()
    pool[0, 1] = torch.tensor([3, 4])
    # Coverage at later rejected positions must not affect first-error attribution.
    pool[0, 2] = torch.tensor([1, 2])
    preserve = torch.ones_like(valid)
    preserve[0, 1] = False
    stats = audit.split_first_losses(student, base, pool, ref, valid, preserve)
    assert stats["anchors"] == 5
    assert stats["gained_reference_tokens"] == 2
    assert stats["lost_reference_tokens"] == 3
    assert stats["support_lost_reference_tokens"] == 2
    assert stats["selection_lost_reference_tokens"] == 1
    assert stats["support_loss_anchors"] == stats["selection_loss_anchors"] == 1
    assert stats["lost_reference_tokens_not_guarded_by_v2"] == 2
    assert stats["lost_reference_tokens_guarded_by_v2"] == 1
    summary = audit.summarize({k: v.item() for k, v in stats.items()})
    assert summary["paired_reference_gain"] == pytest.approx(-.2)


def test_v2_preservation_does_not_cover_an_evicted_native_correct_token():
    ref = torch.ones(1, 3, dtype=torch.long)
    candidates = torch.tensor([[[1, 2], [3, 4], [1, 2]]])
    scores = torch.tensor([[[2., 0.], [2., 0.], [2., 0.]]], requires_grad=True)
    usable = torch.tensor([[True, True, False]])
    terms, _ = boundary_loss_terms(scores, scores.detach(), scores.detach(),
        candidates, ref, usable, usable, usable, {})
    # Only slot 0 is protected; the evicted correct choice at slot 1 vanishes.
    assert terms["preserve_kl"][1] == 1
    terms["preserve_kl"][0].backward()
    assert scores.grad[0, 1].abs().sum() == 0
    base = ref
    student = torch.tensor([[1, 3, 1]])
    stats = audit.split_first_losses(student, base, candidates, ref,
        torch.ones_like(usable), torch.tensor([[True, False, False]]))
    assert stats["support_lost_reference_tokens"] == 2


def test_invalid_cohort_and_shapes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="No usable"):
        audit.summarize({"anchors": 0})
    with pytest.raises(ValueError, match="rank directories"):
        audit.cohort_inventory(tmp_path, 2, 32)
    folder = tmp_path / "validation_inputs/rank_0"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps([{"file": "../outside.pt"}]))
    with pytest.raises(ValueError, match="inventory"):
        audit.cohort_inventory(tmp_path, 1, 32)


def model_and_candidate(tmp_path):
    original, target = compiled_checkpoint(tmp_path)
    settings = dict(init_checkpoint=str(original), target_path=str(target),
        draft_attn_impl="eager", train_last_layers="all", objective="dual_boundary",
        val_batches=1, anchor_chunk_size=2)
    model = build_boundary_model(original, target, settings).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        batch = tiny_batch(model)
    batch.pop("max_anchors")
    with torch.no_grad():
        # Deterministic parameter perturbation simulates an exported student.
        model.prefix_head.head_weights.add_(.125)
        model.layers[0].input_layernorm.weight.mul_(1.125)
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/3"
    with exported_precision(model):
        model.save_pretrained(checkpoint)
    return original, target, settings, model, batch, run, checkpoint


def test_tiny_replay_matches_existing_forward_and_changes_no_weights(tmp_path):
    original, target, settings, trained, batch, _, checkpoint = model_and_candidate(tmp_path)
    model = build_boundary_model(original, target, settings).eval()
    teacher = {k: v.clone() for k, v in model.teacher_model.state_dict().items()}
    audit.load_student(model, checkpoint, original)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with exported_precision(trained), exported_precision(model), torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, _, expected = trained(**batch, max_anchors=3)
        actual = audit.summarize({k: v.item() for k, v in audit.audit_batch(model, batch, 3, 2).items()})
    for key in ("student_reference_eal", "baseline_reference_eal", "paired_reference_gain", "reference_tokens_gained", "reference_tokens_lost"):
        target_value = (expected[key + "_sum"] / expected[key + "_total"]).item()
        assert actual[key] == pytest.approx(target_value, abs=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key]), key
    for key, value in model.teacher_model.state_dict().items():
        assert torch.equal(value, teacher[key]), key


def test_loading_refuses_changed_target_owned_tensor(tmp_path):
    original, target, settings, _, _, _, checkpoint = model_and_candidate(tmp_path)
    state = load_file(str(checkpoint / "model.safetensors"))
    # Some exports omit target-owned weights. Adding a changed one must fail too.
    model = build_boundary_model(original, target, settings).eval()
    state["embed_tokens.weight"] = model.embed_tokens.weight.detach().cpu().clone() + 1
    save_file(state, str(checkpoint / "model.safetensors"))
    with pytest.raises(ValueError, match="Frozen target-owned"):
        audit.load_student(model, checkpoint, original)


def test_loading_refuses_changed_architecture(tmp_path):
    original, target, settings, _, _, _, checkpoint = model_and_candidate(tmp_path)
    path = checkpoint / "config.json"
    cfg = json.loads(path.read_text())
    cfg["prefix_top_k"] = 8
    path.write_text(json.dumps(cfg))
    model = build_boundary_model(original, target, settings).eval()
    with pytest.raises(ValueError, match="configuration differs"):
        audit.load_student(model, checkpoint, original)


def test_complete_cli_replay_and_report_without_training_or_server(tmp_path):
    original, target, settings, model, batch, run, checkpoint = model_and_candidate(tmp_path)
    (run / "settings.json").write_text(json.dumps(settings))
    (run / "experiment.json").write_text(json.dumps({
        "initial_config_sha256": audit.sha256(original / "config.json"),
        "initial_weight_sha256": {p.name: audit.sha256(p) for p in original.glob("*.safetensors")},
        "trainer_npus": ["2"], "train_config": {"max_anchors": 3},
    }))
    directory = run / "validation_inputs/rank_0"
    directory.mkdir(parents=True)
    cache = directory / "batch_0000.pt"
    torch.save(batch, cache)
    (directory / "manifest.json").write_text(json.dumps([{
        "file": cache.name, "bytes": cache.stat().st_size, "sha256": audit.sha256(cache)}]))
    with exported_precision(model), torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, _, raw = model(**batch, max_anchors=3)
    metrics = {k[:-4] + "_epoch": (value / raw[k[:-4] + "_total"]).item()
               for k, value in raw.items() if k.endswith("_sum")}
    (run / "checkpoints/boundary_history.jsonl").write_text(json.dumps({"window": 3, "step": 2000, "metrics": metrics}) + "\n")
    before = {p: audit.sha256(p) for p in run.rglob("*") if p.is_file()}
    output = run / "audit.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(run),
        "--device", "cpu", "--output", str(output)], capture_output=True, text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["complete"] and report["interpretation_ready"]
    assert report["history_comparison"]["pass"]
    assert report["audit_version"] == 2
    assert report["original_forward_comparison"]["pass"]
    assert report["precision"]["floating_parameter_dtypes"] == ["torch.bfloat16"]
    assert report["precision"]["floating_buffer_dtypes"]["rotary_emb.inv_freq"] == "torch.float32"
    assert report["precision"]["floating_buffer_dtypes"]["teacher_model.rotary_emb.inv_freq"] == "torch.float32"
    for path, digest in before.items():
        assert audit.sha256(path) == digest
    # Existing paths, including reports, must never be overwritten.
    second = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(run),
        "--device", "cpu", "--output", str(output)], capture_output=True, text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    assert second.returncode != 0


def test_parameter_only_precision_preserves_rope_where_blanket_cast_changes_it(tmp_path):
    original, target = compiled_checkpoint(tmp_path)
    model = build_boundary_model(original, target, {"train_last_layers": "all", "draft_attn_impl": "eager"}).eval()
    hidden = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
    positions = torch.tensor([[0, 127, 1023, 2047]])
    original_buffers = {k: v.clone() for k, v in model.named_buffers() if v.is_floating_point()}
    assert original_buffers["rotary_emb.inv_freq"].dtype == torch.float32
    with torch.no_grad():
        expected = model.rotary_emb(hidden, positions)
        with exported_precision(model):
            actual = model.rotary_emb(hidden, positions)
            for k, value in original_buffers.items():
                current = dict(model.named_buffers())[k]
                assert current.dtype == value.dtype and torch.equal(current, value)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        bad = copy.deepcopy(model).to(dtype=torch.bfloat16)
        changed = bad.rotary_emb(hidden, positions)
        assert bad.rotary_emb.inv_freq.dtype == torch.bfloat16
        assert max(float((a.float() - b.float()).abs().max()) for a, b in zip(changed, expected)) > 1e-4


def test_internal_forward_check_does_not_tolerate_real_prefix_count_drift():
    keys = ("student_reference_eal", "baseline_reference_eal", "paired_reference_gain", "reference_tokens_gained", "reference_tokens_lost")
    raw = {key + suffix: torch.tensor(10. if suffix == "_total" else 20.)
           for key in keys for suffix in ("_sum", "_total")}
    matched = {key: 2. for key in keys}
    assert audit.compare_forward(raw, matched)["pass"]
    changed = dict(matched, student_reference_eal=2.1)
    assert not audit.compare_forward(raw, changed)["pass"]
