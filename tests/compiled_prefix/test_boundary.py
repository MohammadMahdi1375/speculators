import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from test_repair import compiled_checkpoint
from test_training import tiny_batch
from speculators.model import SpeculatorModel
from speculators.models.dflash_prefix.boundary_objective import (
    boundary_loss_terms, boundary_global_loss, boundary_settings,
    dense_boundary_anchors, refresh_rows,
)
from speculators.models.dflash_prefix.boundary_training import build_boundary_model
from speculators.models.dflash_prefix.boundary_validation import CachedValidationLoader, exported_precision
from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
from speculators.train.checkpointer import SingleGPUCheckpointer


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def loss_fixture(native=True):
    scores = torch.tensor([[[0., 2.], [0., 2.], [0., 2.]]], requires_grad=True)
    old = scores.detach().clone().requires_grad_()
    fresh = torch.tensor([[[2., 0.], [2., 0.], [2., 0.]]], requires_grad=True)
    candidates = torch.tensor([[[7, 9], [7, 9], [7, 9]]])
    target = torch.full((1, 3), 7, dtype=torch.long)
    valid = torch.ones(1, 3, dtype=torch.bool)
    terms, stats = boundary_loss_terms(scores, old, fresh, candidates, target,
        valid, valid, valid if native else ~valid, {})
    return scores, old, fresh, terms, stats


def test_refresh_is_same_absolute_token_and_starts_strictly_before_query():
    active, proposals = 5, 7
    rr = refresh_rows(active, proposals, device="cpu")
    anchors = 19 + torch.arange(active + proposals - 1)
    queries = anchors[:active, None] + torch.arange(1, proposals+1)[None]
    assert torch.equal(anchors[rr] + 1, queries)
    assert (anchors[rr] < queries).all()
    assert rr.max() == len(anchors) - 1


def test_future_target_token_cannot_affect_refreshed_backbone_prediction(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    model = build_boundary_model(checkpoint, target, {"draft_attn_impl": "eager"}).eval()
    batch = tiny_batch(model)
    batch.pop("max_anchors")
    anchors = torch.tensor([2, 3, 4, 5, 6])
    model.teacher_model._boundary_anchors = anchors, torch.ones(5, dtype=torch.bool)
    def compute(data):
        return model.teacher_model._backbone_forward(**data, max_anchors=5)[0].reshape(5, 4, -1)[2, 1].clone()
    with torch.no_grad():
        expected = compute(batch)
        changed = {k: v.clone() for k, v in batch.items()}
        # Query is token 5 from refreshed anchor 4. Change the query token and
        # everything later, including target features unavailable at that anchor.
        changed["input_ids"][:, 5:] = (changed["input_ids"][:, 5:] + 1) % 16
        changed["hidden_states"][:, 5:] += 100
        changed["verifier_last_hidden_states"][:, 5:] -= 100
        actual = compute(changed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_validation_anchors_are_rng_independent_and_never_wrap():
    mask = torch.ones(1, 100)
    torch.manual_seed(1)
    first = dense_boundary_anchors(mask, 46, 16, training=False)
    torch.rand(1000)
    second = dense_boundary_anchors(mask, 46, 16, training=False)
    assert torch.equal(first[0], second[0])
    assert first[0][-1] + 15 < 100
    with pytest.raises(ValueError, match="fit"):
        dense_boundary_anchors(torch.ones(1, 20), 16, 16, training=False)


def test_boundary_teacher_detached_and_first_position_never_distilled():
    scores, old, fresh, terms, _ = loss_fixture()
    assert terms["boundary_kl"][1] == 2
    terms["boundary_kl"][0].backward()
    assert scores.grad[0, 1, 0] < 0
    assert torch.equal(scores.grad[0, 0], torch.zeros(2))
    assert old.grad is None and fresh.grad is None


def test_teacher_wrong_in_natural_pool_is_not_trusted():
    scores, _, _, terms, _ = loss_fixture(native=False)
    assert terms["boundary_kl"][1] == 0
    terms["boundary_kl"][0].backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_uncovered_targets_and_unknown_prefixes_do_not_get_fake_labels():
    s = torch.randn(1, 3, 2, requires_grad=True)
    ids = torch.tensor([[[1, 2], [1, 2], [1, 2]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    terms, _ = boundary_loss_terms(s, s.detach(), s.detach(), ids,
        torch.tensor([[7, 1, 1]]), torch.tensor([[True, False, False]]), valid, valid, {})
    assert all(den == 0 for _, den in terms.values())
    sum(value for value, _ in terms.values()).backward()
    assert torch.isfinite(s.grad).all() and s.grad.abs().sum() == 0


def test_control_removes_only_boundary_gradient():
    s, _, _, terms, _ = loss_fixture()
    terms["full_ce"] = (s.sum() * 0, s.new_tensor(1))
    a = boundary_global_loss(terms, {})
    b = boundary_global_loss(terms, {"objective": "control"})
    torch.testing.assert_close(a - b, .5 * terms["boundary_kl"][0] / terms["boundary_kl"][1])


def test_global_denominators_equal_pooled_gradient(monkeypatch):
    import speculators.models.dflash_prefix.boundary_objective as objective
    x = torch.tensor(2., requires_grad=True)
    names = ("full_ce", "selector_ce", "boundary_kl", "preserve_kl")
    den = torch.tensor([5., 3., 0., 7.])
    terms = {k: (x * (i+1), torch.tensor(0. if i == 2 else 1.)) for i, k in enumerate(names)}
    monkeypatch.setattr(objective.dist, "is_available", lambda: True)
    monkeypatch.setattr(objective.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(objective.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(objective.dist, "all_reduce", lambda t: t.copy_(den))
    boundary_global_loss(terms, {}).backward()
    assert x.grad.item() / 2 == pytest.approx(sum(w*(i+1)/max(den[i].item(), 1) for i,w in enumerate((.25,1,.5,.1))))


def test_joint_training_changes_last_layer_and_head_but_not_teacher_or_frozen_weights(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    model = build_boundary_model(checkpoint, target, {"draft_attn_impl": "eager"})
    SpeculatorModel.verify_training_compatible(model)
    saved = load_file(str(checkpoint / "model.safetensors"))
    for k, v in saved.items():
        torch.testing.assert_close(model.state_dict()[k], v.to(model.state_dict()[k]), rtol=0, atol=0)
    frozen = {k: v.detach().clone() for k,v in model.named_parameters() if not v.requires_grad}
    mutable = {k: v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4, weight_decay=0)
    model.train()
    assert not model.teacher_model.training
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss, metrics = model(**tiny_batch(model))
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["full_ce_total"] == 9
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert name.startswith(("prefix_head.", "layers.0.", "norm."))
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
    optimizer.step()
    for k,v in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[k], v, rtol=0, atol=0)
    changed = [k for k,v in mutable.items() if not torch.equal(dict(model.named_parameters())[k], v)]
    assert any(k.startswith("layers.") for k in changed)
    assert any(k.startswith("prefix_head.") for k in changed)


def test_export_has_original_state_shapes_and_no_teacher_or_new_serving_class(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    model = build_boundary_model(checkpoint, target)
    original = load_file(str(checkpoint / "model.safetensors"))
    checkpointer = SingleGPUCheckpointer(tmp_path / "checkpoints")
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad])
    checkpointer.save_checkpoint(model, optimizer, 0, float_dtype=torch.bfloat16)
    export = tmp_path / "checkpoints/0"
    state = load_file(str(export / "model.safetensors"))
    assert {k:v.shape for k,v in state.items()} == {k:v.shape for k,v in original.items()}
    cfg = json.loads((export / "config.json").read_text())
    assert cfg["architectures"] == ["DFlashPrefixDraftModel"]
    assert cfg["prefix_selector_kind"] == "compiled_prefix_v3"
    serving = DFlashPrefixDraftModel.from_pretrained(export, local_files_only=True, dtype="auto")
    assert not hasattr(serving, "teacher_model")
    with exported_precision(model):
        for k,v in state.items():
            torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    assert model.prefix_head.head_weights.dtype == torch.float32


def test_export_precision_restores_parameter_identity_storage_and_optimizer_after_error():
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters())
    originals = [(p, p.data, p.detach().clone()) for p in model.parameters()]
    with pytest.raises(RuntimeError):
        with exported_precision(model):
            assert all(p.dtype == torch.bfloat16 for p in model.parameters())
            raise RuntimeError("test")
    for p, storage, value in originals:
        assert p.data.data_ptr() == storage.data_ptr()
        assert any(p is q for group in opt.param_groups for q in group["params"])
        torch.testing.assert_close(p, value, rtol=0, atol=0)
    model(torch.ones(2,3)).sum().backward()
    opt.step()


def test_validation_replays_same_target_inputs_without_fetching_again(tmp_path):
    class Loader:
        count = 0
        def __len__(self): return 2
        def __iter__(self):
            self.count += 1
            for _ in range(2):
                yield {k: torch.rand(1,4,2) for k in CachedValidationLoader.FIELDS}
    raw = Loader()
    cached = CachedValidationLoader(raw, 2, tmp_path / "cohort")
    first, second = list(cached), list(cached)
    assert raw.count == 1
    for a,b in zip(first, second):
        for k in a:
            assert torch.equal(a[k], b[k])
    assert len(json.loads((tmp_path / "cohort/manifest.json").read_text())) == 2


def test_empty_mask_has_zero_loss_and_connected_gradients(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    model = build_boundary_model(checkpoint, target, {"draft_attn_impl": "eager"})
    _, loss, _ = model(**tiny_batch(model, valid=False))
    loss.backward()
    assert loss == 0
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_fixed_bf16_validation_keeps_frozen_baseline_after_student_update(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    model = build_boundary_model(checkpoint, target, {"draft_attn_impl": "eager"})
    batch = tiny_batch(model)
    model.eval()
    with torch.no_grad(), exported_precision(model), torch.autocast("cpu", dtype=torch.bfloat16):
        _, _, initial = model(**batch)
    assert initial["paired_reference_gain_sum"] == 0
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    _, loss, _ = model(**batch)
    loss.backward()
    opt.step()
    torch.rand(991)
    model.eval()
    with torch.no_grad(), exported_precision(model), torch.autocast("cpu", dtype=torch.bfloat16):
        _, _, after = model(**batch)
    torch.testing.assert_close(initial["baseline_reference_eal_sum"], after["baseline_reference_eal_sum"], rtol=0, atol=0)
    torch.testing.assert_close(initial["baseline_reference_eal_total"], after["baseline_reference_eal_total"], rtol=0, atol=0)


@pytest.mark.parametrize("settings", [{"objective":"renewal"}, {"backbone_lr":0}, {"train_last_layers":0}, {"boundary_weight":float("nan")}])
def test_bad_settings_fail(settings):
    with pytest.raises(ValueError):
        boundary_settings(settings)
