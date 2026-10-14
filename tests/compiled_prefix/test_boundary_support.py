import pytest
import torch
from safetensors.torch import load_file

from test_repair import compiled_checkpoint
from test_training import tiny_batch
from speculators.models.dflash_prefix.boundary_support import support_transfer_terms
from speculators.models.dflash_prefix.boundary_objective import boundary_global_loss
from speculators.models.dflash_prefix.boundary_training import build_boundary_model
from speculators.models.dflash_prefix.boundary_validation import exported_precision
from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
from speculators.train.checkpointer import SingleGPUCheckpointer


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    s = torch.tensor([[[2., 1., 0., -2.]] * 3], requires_grad=True)
    old = s.detach().clone().requires_grad_()
    fresh = torch.tensor([[[0., 0., 0., 4.]] * 3], requires_grad=True)
    target = torch.full((1, 3), 3, dtype=torch.long)
    valid = torch.ones_like(target, dtype=torch.bool)
    covered = s.detach().topk(2, dim=-1).indices.eq(target[..., None]).any(-1)
    return s, old, fresh, target, valid, covered


def test_missing_candidate_gets_full_vocab_gradient_without_insertion():
    s, old, fresh, target, valid, covered = fixture()
    before = s.detach().topk(2, -1).indices.clone()
    terms, stats = support_transfer_terms(s, old, fresh, target, valid, covered)
    assert terms["support_kl"][1] == 2
    assert stats["support_teachable_miss_fraction"][0] == 2
    terms["support_kl"][0].backward()
    assert (s.grad[:, 1:, 3] < 0).all()
    assert s.grad[:, 0].abs().sum() == 0
    assert old.grad is None and fresh.grad is None
    assert torch.equal(before, s.detach().topk(2, -1).indices)


@pytest.mark.parametrize("case", ["wrong_teacher", "no_advantage", "unknown_context"])
def test_support_rejects_invalid_teacher_or_context(case):
    s, old, fresh, target, valid, covered = fixture()
    if case == "wrong_teacher": fresh = old.detach()
    elif case == "no_advantage": old = fresh.detach()
    else: valid = ~valid
    terms, _ = support_transfer_terms(s, old, fresh, target, valid, covered)
    assert terms["support_kl"][1] == 0
    terms["support_kl"][0].backward()
    assert torch.isfinite(s.grad).all() and s.grad.abs().sum() == 0


def test_support_loss_is_full_vocabulary_kl_and_logit_shift_invariant():
    s, old, fresh, target, valid, covered = fixture()
    terms, _ = support_transfer_terms(s, old, fresh, target, valid, covered)
    shifted, _ = support_transfer_terms(s + 5, old - 3, fresh + 2, target, valid, covered)
    manual = torch.nn.functional.kl_div(s[:, 1:].log_softmax(-1),
                                      fresh.detach()[:, 1:].softmax(-1), reduction="sum")
    torch.testing.assert_close(terms["support_kl"][0], manual)
    torch.testing.assert_close(terms["support_kl"][0], shifted["support_kl"][0])


def test_three_arms_separate_both_distillation_signals():
    x = torch.tensor(2., requires_grad=True)
    terms = {k: (x * (i+1), x.new_tensor(2.)) for i,k in enumerate(
        ("full_ce", "selector_ce", "boundary_kl", "preserve_kl", "support_kl"))}
    control = boundary_global_loss(terms, {"objective": "control"})
    selector = boundary_global_loss(terms, {"objective": "boundary"})
    dual = boundary_global_loss(terms, {"objective": "dual_boundary"})
    torch.testing.assert_close(selector - control, .5 * terms["boundary_kl"][0] / 2)
    torch.testing.assert_close(dual - selector, .25 * terms["support_kl"][0] / 2)


def test_five_global_denominators_include_support(monkeypatch):
    import speculators.models.dflash_prefix.boundary_objective as objective
    x = torch.tensor(2., requires_grad=True)
    names = ("full_ce", "selector_ce", "boundary_kl", "preserve_kl", "support_kl")
    denominators = torch.tensor([5., 3., 0., 7., 2.])
    terms = {k: (x * (i+1), x.new_tensor(1.)) for i,k in enumerate(names)}
    monkeypatch.setattr(objective.dist, "is_available", lambda: True)
    monkeypatch.setattr(objective.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(objective.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(objective.dist, "all_reduce", lambda t: t.copy_(denominators))
    boundary_global_loss(terms, {"objective": "dual_boundary"}).backward()
    expected = sum(w*(i+1)/max(denominators[i].item(),1)
                   for i,w in enumerate((.25,1,.5,.1,.25)))
    assert x.grad.item() / 2 == pytest.approx(expected)


def test_all_backbone_training_preserves_target_teacher_and_export(tmp_path):
    checkpoint, target = compiled_checkpoint(tmp_path)
    # Two layers exercise the new 'all' scope rather than a one-layer shortcut.
    import copy
    original_model = DFlashPrefixDraftModel.from_pretrained(checkpoint, local_files_only=True, dtype="auto")
    config = copy.deepcopy(original_model.config)
    config.transformer_layer_config.num_hidden_layers = 2
    config.transformer_layer_config.layer_types = ["full_attention", "full_attention"]
    source = DFlashPrefixDraftModel(config)
    missing, unexpected = source.load_state_dict(original_model.state_dict(), strict=False)
    assert missing and all(k.startswith("layers.1.") for k in missing) and not unexpected
    checkpoint = tmp_path / "two_layer_compiled"
    source.save_pretrained(checkpoint)
    model = build_boundary_model(checkpoint, target,
        {"draft_attn_impl": "eager", "train_last_layers": "all", "objective": "dual_boundary"})
    frozen = {k:p.clone() for k,p in model.named_parameters() if not p.requires_grad}
    mutable = {k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
    assert "fc.weight" in mutable and "hidden_norm.weight" in mutable
    assert any(k.startswith("layers.0.") for k in mutable)
    assert any(k.startswith("layers.1.") for k in mutable)
    assert "embed_tokens.weight" not in mutable and "lm_head.weight" not in mutable
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch = tiny_batch(model)
    model.eval()
    with torch.no_grad(), exported_precision(model), torch.autocast("cpu", dtype=torch.bfloat16):
        _, _, initial = model(**batch)
    assert initial["paired_reference_gain_sum"] == 0
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss, metrics = model(**batch)
    assert "support_kl_sum" in metrics
    assert "support_teachable_miss_fraction_sum" in metrics
    loss.backward()
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert name.startswith(("prefix_head.", "layers.", "norm.", "fc.", "hidden_norm."))
        else: assert p.grad is None, name
    opt.step()
    for name,expected in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0)
    assert not torch.equal(model.fc.weight, mutable["fc.weight"])
    for layer in ("layers.0.", "layers.1."):
        assert any(k.startswith(layer) and not torch.equal(dict(model.named_parameters())[k], v)
                   for k,v in mutable.items())
    model.eval()
    with torch.no_grad(), exported_precision(model), torch.autocast("cpu", dtype=torch.bfloat16):
        _, _, after = model(**batch)
    torch.testing.assert_close(initial["baseline_reference_eal_sum"], after["baseline_reference_eal_sum"], rtol=0, atol=0)
    for key in ("oracle_reference_eal_sum", "unary_reference_eal_sum"):
        assert key in after
    checkpointer = SingleGPUCheckpointer(tmp_path / "export")
    checkpointer.save_checkpoint(model, opt, 0, float_dtype=torch.bfloat16)
    saved = load_file(str(tmp_path / "export/0/model.safetensors"))
    original = load_file(str(checkpoint / "model.safetensors"))
    assert {k:v.shape for k,v in saved.items()} == {k:v.shape for k,v in original.items()}
    serving = DFlashPrefixDraftModel.from_pretrained(tmp_path / "export/0", local_files_only=True, dtype="auto")
    assert not hasattr(serving, "teacher_model")
