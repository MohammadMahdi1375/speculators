"""CPU checks of credit assignment, actual training, and serving checkpoint export."""

import itertools
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_training import make_checkpoint, tiny_batch
from speculators.model import SpeculatorModel
from speculators.models.dflash_prefix.compiled_training import build_compiled_pilot
from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
from speculators.models.dflash_prefix.repair_objective import (
    global_masked_loss, prefix_credit, repair_loss_terms, repair_settings,
)
from speculators.models.dflash_prefix.repair_training import build_repair_model


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def compiled_checkpoint(root):
    _, teacher, target = make_checkpoint(root)
    model = build_compiled_pilot(teacher, target)
    with torch.no_grad():
        model.prefix_head.head_weights.fill_(.3)
    checkpoint = root / "compiled"
    model.save_pretrained(checkpoint)
    return checkpoint, target


def test_repair_span_matches_interacting_prefix_rollout_exhaustively():
    # Toy policy genuinely depends on ALL previous token choices; predictions
    # after the uncorrected mistake differ from reference-conditioned scores.
    for bits in itertools.product((False, True), repeat=5):
        for known_length in range(6):
            correct = torch.tensor([bits])
            usable = torch.arange(5)[None] < known_length
            frontier, spans, accepted, _ = prefix_credit(correct, usable)
            def rollout(force=None):
                prefix = []
                for i in range(known_length):
                    choice = (0 if bits[i] else 1) ^ (sum(prefix) % 2)
                    if i == force:
                        choice = 0
                    prefix.append(choice)
                count = 0
                for token in prefix:
                    if token:
                        break
                    count += 1
                return count
            original = rollout()
            assert accepted.sum().item() == original
            if original < known_length:
                assert frontier[0, original]
                assert spans.sum().item() == rollout(force=original) - original
            else:
                assert not frontier.any()
                assert spans.sum() == 0


def sample_terms(scores, baseline=None, usable=None, objective="repair"):
    baseline = scores.detach().clone() if baseline is None else baseline
    n, p, k = scores.shape
    ids = torch.arange(k)[None, None].expand(n, p, k)
    target = torch.zeros(n, p, dtype=torch.long)
    valid = torch.ones(n, p, dtype=torch.bool)
    context = valid if usable is None else usable
    cfg = repair_settings(dict(objective=objective))
    return (*repair_loss_terms(scores, baseline, ids, target, valid, context, cfg), cfg)


def test_wrong_teacher_is_not_a_distillation_target_and_repair_span_has_credit():
    scores = torch.tensor([[[0., 1.], [2., 0.], [2., 0.], [0., 1.]]], requires_grad=True)
    terms, stats, cfg = sample_terms(scores)
    assert stats["repair_span"][0] == 3
    assert terms["accepted_kl"][1] == 0
    assert terms["protect_margin"][1] == 0
    terms["repair_rank"][0].backward()
    assert scores.grad[0, 0, 0] < 0 and scores.grad[0, 0, 1] > 0
    assert torch.equal(scores.grad[0, 1:], torch.zeros_like(scores.grad[0, 1:]))
    other, _, _ = sample_terms(scores.detach().requires_grad_(), objective="frontier")
    torch.testing.assert_close(terms["repair_rank"][0], 3 * other["repair_rank"][0])


def test_protection_is_zero_initially_and_penalizes_losing_an_accepted_choice():
    baseline = torch.tensor([[[2., 0.], [1., 0.], [0., 1.]]])
    same, _, _ = sample_terms(baseline.clone(), baseline)
    assert same["protect_margin"][0] == 0
    changed = baseline.clone()
    changed[0, 0] = torch.tensor([0., 2.])
    changed.requires_grad_(True)
    terms, stats, _ = sample_terms(changed, baseline)
    assert terms["protect_margin"][0] > 0
    assert stats["protected_choice_damage"][0] == 1
    terms["protect_margin"][0].backward()
    assert changed.grad[0, 0, 0] < 0


def test_missing_target_and_unknown_prefix_are_absorbing_and_not_fabricated_labels():
    scores = torch.tensor([[[2., 0.], [0., 2.], [0., 2.], [0., 2.]]], requires_grad=True)
    candidates = torch.tensor([[[0, 1], [2, 3], [0, 1], [0, 1]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    terms, stats = repair_loss_terms(scores, scores.detach(), candidates,
                                    torch.zeros(1, 4, dtype=torch.long), valid, valid, {})
    assert terms["target_ce"][1] == 1
    assert stats["repairable_frontier_fraction"][0] == 0
    assert stats["shortlist_blocked_fraction"][0] == 1
    global_masked_loss(terms, {}).backward()
    assert torch.equal(scores.grad[:, 1:], torch.zeros_like(scores.grad[:, 1:]))
    scores2 = scores.detach().requires_grad_()
    terms, _, _ = sample_terms(scores2, usable=torch.tensor([[True, False, True, True]]))
    assert terms["target_ce"][1] == 1


def test_empty_labels_have_connected_finite_zero_gradients():
    scores = torch.randn(2, 7, 16, requires_grad=True)
    ids = torch.arange(16)[None, None].expand_as(scores)
    empty = torch.zeros(2, 7, dtype=torch.bool)
    terms, _ = repair_loss_terms(scores, scores.detach(), ids, empty.long(), empty, empty, {})
    loss = global_masked_loss(terms, {})
    loss.backward()
    assert loss == 0
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_ddp_denominators_match_global_gradient_with_unequal_rank_masks(monkeypatch):
    # Communication is mocked; arithmetic is independently compared with the
    # concatenated global objective. Real Gloo/HCCL is a separate integration test.
    import speculators.models.dflash_prefix.repair_objective as objective
    x = torch.tensor(2., requires_grad=True)
    counts = [3., 0., 7., 2.]
    names = ("target_ce", "repair_rank", "protect_margin", "accepted_kl")
    coefficients = (.25, .5, 1., .05)
    terms = {name: (x * (i + 1), torch.tensor(0. if i == 1 else 1.)) for i, name in enumerate(names)}
    monkeypatch.setattr(objective.dist, "is_available", lambda: True)
    monkeypatch.setattr(objective.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(objective.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(objective.dist, "all_reduce", lambda tensor: tensor.copy_(tensor.new_tensor(counts)))
    loss = global_masked_loss(terms, {})
    loss.backward()
    expected = sum(coef * (i + 1) / max(counts[i], 1) for i, coef in enumerate(coefficients))
    assert x.grad.item() / 2 == pytest.approx(expected)


def test_initialize_train_both_terms_export_identical_serving_architecture(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    original = load_file(str(checkpoint / "model.safetensors"))
    model = build_repair_model(checkpoint, target)
    SpeculatorModel.verify_training_compatible(model)
    for name, parameter in model.state_dict().items():
        if name in original:
            torch.testing.assert_close(parameter, original[name].to(parameter), rtol=0, atol=0)
    assert model.prefix_head.predecessor_projection.weight.requires_grad
    assert model.prefix_head.history_value.weight.requires_grad
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=0)
    model.train()
    assert not model.teacher_head.training
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss, metrics = model(**tiny_batch(model))
        loss.backward()
        assert torch.isfinite(loss)
        assert metrics["target_ce_total"] > 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert name.startswith("prefix_head.")
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert model.prefix_head.successor_projection.weight.grad.abs().sum() > 0
        assert model.prefix_head.history_value.weight.grad.abs().sum() > 0
        optimizer.step()
    for name, value in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    exported = tmp_path / "repaired"
    model.save_pretrained(exported, state_dict=model.state_dict())
    saved = load_file(str(exported / "model.safetensors"))
    assert set(saved) == set(original)
    assert not any(key.startswith("teacher_head.") for key in saved)
    assert {key: value.shape for key, value in saved.items()} == {key: value.shape for key, value in original.items()}
    raw = json.loads((exported / "config.json").read_text())
    assert raw["prefix_selector_kind"] == "compiled_prefix_v3"
    assert raw["architectures"] == ["DFlashPrefixDraftModel"]
    serving = DFlashPrefixDraftModel.from_pretrained(exported, local_files_only=True, dtype="auto").eval()
    # Actual serving candidate table/walk remains usable after exporting.
    torch.manual_seed(611)
    h, ids = torch.randn(1, 3, 16), torch.arange(16)[None, None].expand(1, 3, 16)
    u, anchor = torch.randn(1, 3, 16), torch.tensor([2])
    with torch.no_grad():
        a = model.prefix_head.greedy(h, ids, u, anchor, embedding_weight=model.embed_tokens.weight)
        b = serving.prefix_head.greedy(h, ids, u, anchor, embedding_weight=serving.embed_tokens.weight)
    assert torch.equal(a, b)


def test_reject_missing_history_weights(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    path = checkpoint / "model.safetensors"
    state = load_file(str(path))
    del state["prefix_head.history_key.weight"]
    save_file(state, str(path))
    with pytest.raises(ValueError, match="Incomplete"):
        build_repair_model(checkpoint, target)


def test_bf16_starting_checkpoint_keeps_initial_decisions_and_scores(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    path = checkpoint / "model.safetensors"
    state = load_file(str(path))
    save_file({k: v.to(torch.bfloat16) for k, v in state.items()}, str(path))
    raw = json.loads((checkpoint / "config.json").read_text())
    raw["dtype"] = raw["torch_dtype"] = "bfloat16"
    (checkpoint / "config.json").write_text(json.dumps(raw))
    model = build_repair_model(checkpoint, target).eval()
    h, ids = torch.randn(1, 3, 16).bfloat16(), torch.arange(16)[None, None].expand(1, 3, 16)
    u, anchor = torch.randn(1, 3, 16), torch.tensor([2])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        args = (h, ids, u, anchor)
        kw = dict(embedding_weight=model.embed_tokens.weight)
        expected = model.teacher_head.walk(model.teacher_head.prepare_tables(*args, **kw), return_scores=True)
        actual = model.prefix_head.walk(model.prefix_head.prepare_tables(*args, **kw), return_scores=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [dict(objective="invented"), dict(margin=float("nan")),
                                     dict(span_cap=0), dict(anchor_chunk_size=17), dict(ce_weight=0)])
def test_invalid_settings(bad):
    with pytest.raises(ValueError):
        repair_settings(bad)
