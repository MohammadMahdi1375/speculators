import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
import yaml

from speculators.losses import resolve_loss_config
from speculators.models.dflash_prefix.joint_training import selector_weights
from speculators.models.dflash_prefix.joint_guard import (
    FeatureReplay, SelectorUpdateGuard, bf16_replay_copy, flatten_tensors,
    project_displacement, target_prefix_mask,
)
from speculators.models.dflash_prefix.joint_validation import checkpoint_parameters
from speculators.train.config import TrainConfig
from speculators.train.config.schema import CONFIG_DESTS, nest_flat
from speculators.train.checkpointer import SingleGPUCheckpointer


ROOT = Path(__file__).resolve().parents[2]


def helper(name):
    spec = importlib.util.spec_from_file_location("joint_" + name, ROOT / "scripts/dflash_prefix_joint" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings():
    return dict(block_size=8, rank=16, heads=2, top_k=16, compiled_logit_cap=8.,
        epochs=5, loss_alpha=.25, delay_steps=0, head_warmup_steps=1,
        focus_start_steps=2, focus_warmup_steps=2, dense_floor=.25, anchor_chunk_size=2,
        guard_mode="project", guard_start_steps=0, guard_every=1,
        guard_replay_anchors=8, guard_add_anchors=4, guard_chunk_size=2, guard_backtracks=3,
        head_lr=3e-4, head_weight_decay=.01, val_batches=2)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("block", [8, 16])
def test_three_joint_updates_saved_reload_and_gradients(block):
    report = helper("check_training").check(dict(settings(), block_size=block), "cpu")
    assert report["backbone_and_head_backward"] and report["optimizer_steps"] == 3
    assert all(r["accepted_lost_tokens"] == 0 for r in report["guard_reports"])


def test_dense_curriculum_retains_late_supervision_and_omits_pool_misses():
    valid = torch.tensor([[True, True, True, False]])
    covered = torch.tensor([[True, False, True, True]])
    reachable = torch.tensor([[True, True, False, False]])
    assert selector_weights(valid, covered, reachable, 0., .25).tolist() == [[1, 0, 1, 0]]
    assert selector_weights(valid, covered, reachable, 1., .25).tolist() == [[1, 0, .25, 0]]


def test_target_context_does_not_extend_after_reference_divergence():
    target = torch.tensor([[1, 5, 3, 4]])
    ref = torch.tensor([[1, 2, 3, 4]])
    mask = target_prefix_mask(target, ref, target, torch.ones_like(target, dtype=torch.bool))
    assert mask.tolist() == [[True, True, False, False]]


def test_projection_constrains_actual_displacement_not_raw_gradient():
    # An adaptive/momentum update may point against the instantaneous gradient.
    delta, g = torch.tensor([2., 3.]), torch.tensor([1., 1.])
    adjusted, inner, _ = project_displacement(delta, g)
    assert inner == 5 and torch.dot(g, adjusted).abs() < 1e-7
    unchanged, _, _ = project_displacement(-delta, g)
    assert torch.equal(unchanged, -delta)
    unchanged, _, _ = project_displacement(delta, torch.zeros_like(g))
    assert torch.equal(unchanged, delta)


def test_all_invalid_training_batch_has_connected_finite_gradients(tmp_path):
    model, batch = helper("check_training").make_fixture(tmp_path, settings())
    batch["loss_mask"].zero_()
    batch["loss_config"] = resolve_loss_config("ce", "eager")
    model.joint_step = 3
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss, _ = model(**batch)
    assert loss.item() == 0
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name


def populated_guard(model, batch):
    model.joint_step = 3
    batch["loss_config"] = resolve_loss_config("ce", "eager")
    with torch.no_grad():
        model.prefix_head.successor_projection.weight.normal_(std=.1)
        model.prefix_head.head_weights.fill_(.2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(**batch)
    guard = model.selector_guard
    f = guard.replay.data
    assert f is not None
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        chosen = bf16_replay_copy(model.prefix_head).greedy(f["hidden"], f["candidates"], f["unary"], f["anchors"], embedding_weight=model.embed_tokens.weight)
    # Synthetic teacher labels equal initial rollout to exercise every prefix.
    f["reference"], f["target"] = chosen.clone(), chosen.clone()
    f["valid"].fill_(True)
    return guard


def test_rejected_update_restores_weights_and_optimizer_state(tmp_path, monkeypatch):
    model, batch = helper("check_training").make_fixture(tmp_path, settings())
    guard = populated_guard(model, batch)
    guard.max_backtracks = 0
    params = list(model.prefix_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=10.)
    torch.manual_seed(219)
    for p in params:
        p.grad = torch.randn_like(p)
    before = flatten_tensors([p.detach() for p in params])
    before_mask = guard.prefix_masks(model.prefix_head, model.embed_tokens.weight)
    # Isolate the hard nonlinear check, which must protect even when the local
    # linearization is uninformative (zero gradient).
    monkeypatch.setattr(guard, "protection_gradient", lambda h, e, m, p: (torch.zeros_like(before), m.float().sum()))
    report = guard.step(model.prefix_head, optimizer, model.embed_tokens.weight)
    assert report["rejected"] == 1 and report["proposed_lost_tokens"] > 0
    assert torch.equal(flatten_tensors([p.detach() for p in params]), before)
    assert not optimizer.state
    assert torch.equal(guard.prefix_masks(model.prefix_head, model.embed_tokens.weight), before_mask)


def test_guard_preserves_main_gradients_and_checks_exported_parameters(tmp_path):
    model, batch = helper("check_training").make_fixture(tmp_path, settings())
    guard = populated_guard(model, batch)
    head = model.prefix_head
    opt = torch.optim.AdamW(head.parameters(), lr=1e-6)
    grads = {}
    for name, p in head.named_parameters():
        p.grad = torch.randn_like(p)
        grads[name] = p.grad.clone()
    before = guard.prefix_masks(head, model.embed_tokens.weight).sum(-1)
    report = guard.step(head, opt, model.embed_tokens.weight)
    after = guard.prefix_masks(head, model.embed_tokens.weight).sum(-1)
    assert bool((after >= before).all()) and report["accepted_lost_tokens"] == 0
    for name, p in head.named_parameters():
        assert p.dtype == torch.float32
        assert torch.equal(p.grad, grads[name])


def test_auxiliary_guard_gradients_do_not_trigger_live_parameter_hooks(tmp_path):
    model, batch = helper("check_training").make_fixture(tmp_path, settings())
    guard = populated_guard(model, batch)
    calls = []
    handles = [p.register_hook(lambda g: calls.append(True)) for p in model.prefix_head.parameters()]
    try:
        masks = guard.prefix_masks(model.prefix_head, model.embed_tokens.weight)
        gradient, count = guard.protection_gradient(model.prefix_head, model.embed_tokens.weight,
            masks, list(model.prefix_head.parameters()))
        assert count > 0 and torch.isfinite(gradient).all() and not calls
    finally:
        for handle in handles:
            handle.remove()


def test_checkpoint_replay_uses_disk_and_restores_master_weights_and_buffers(tmp_path):
    model, batch = helper("check_training").make_fixture(tmp_path, settings())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    root = tmp_path / "checkpoints"
    SingleGPUCheckpointer(root).save_checkpoint(model, opt, 0)
    with torch.no_grad():
        model.fc.weight.add_(.125)
    masters = {n: (p, p.data, p.detach().clone()) for n, p in model.named_parameters()}
    buffers = {n: (b.dtype, b.clone()) for n, b in model.named_buffers()}
    from safetensors.torch import load_file
    saved = load_file(str(root / "0/model.safetensors"))
    with checkpoint_parameters(model, root / "0") as (replay_model, info):
        assert info["actual_disk_reload"]
        assert torch.equal(replay_model.fc.weight, saved["fc.weight"])
        assert replay_model.fc.weight.dtype == torch.bfloat16
        assert replay_model is not model and not replay_model.training
        assert replay_model.rotary_emb.inv_freq.dtype == torch.float32
        assert replay_model.embed_tokens.weight is model.embed_tokens.weight
        assert model.fc.weight.dtype == torch.float32
        assert torch.equal(model.fc.weight, masters["fc.weight"][2])
        assert not any(p.requires_grad for p in replay_model.parameters())
    for name, p in model.named_parameters():
        original, storage, value = masters[name]
        assert p is original and p.data.data_ptr() == storage.data_ptr() and torch.equal(p, value)
    for name, b in model.named_buffers():
        assert b.dtype == buffers[name][0] and torch.equal(b, buffers[name][1])


def recipe_fixture(tmp_path):
    target, dataset = tmp_path / "new_target", tmp_path / "new_dataset"
    target.mkdir(); dataset.mkdir()
    (target / "config.json").write_text(json.dumps(dict(vocab_size=151936, hidden_size=2560, num_hidden_layers=36)))
    cfg = TrainConfig.resolve(["--speculator-type", "dflash", "--verifier-name-or-path", "/old/missing/target", "--data-path", "/old/missing/data", "--optimizer", "muon", "--lr", ".0006", "--total-seq-len", "3072"])
    nested = nest_flat({k: v for k, v in cfg.flatten().items() if k in CONFIG_DESTS})
    nested["backend"] = dict(cfg.backend_args)
    recipe = tmp_path / "run.yaml"
    recipe.write_text(yaml.safe_dump({"train": nested}))
    return dict(MODEL=str(target), DATASET=str(dataset), BASE_TRAIN_CONFIG=str(recipe), OUTPUT_DIR=str(tmp_path / "out"),
                VLLM_NPUS="0,1", VLLM_DP="2", TRAIN_NPUS="2,3,4,5,6", NUM_TRAIN_NPUS="5",
                VLLM_PORT="8092", VLLM_RPC_PORT="29692")


@pytest.mark.parametrize("guard,block", [("project", 16), ("off", 8)])
def test_config_real_epochs_relocation_and_ablations(tmp_path, guard, block):
    path = helper("prepare_experiment").prepare(dict(recipe_fixture(tmp_path), GUARD_MODE=guard, BLOCK_SIZE=str(block)))
    cfg = TrainConfig.resolve(["--config", str(path)]).flatten()
    s = json.loads((path.parent / "joint_settings.json").read_text())
    assert cfg["epochs"] == 5 and cfg["max_steps"] is None and cfg["scheduler_total_steps"] is None
    assert not cfg["from_pretrained"] and not cfg["prefix_backbone_init"] and cfg["no_resume_from_checkpoint"]
    assert not cfg["prefix_detach_backbone"] and not cfg["prefix_freeze_backbone"]
    assert cfg["lr"] == .0006 and cfg["muon_lr"] == pytest.approx(.006)
    assert s["selector_kind"] == "compiled_prefix_v3" and s["guard_mode"] == guard
    assert cfg["data_path"] == str(tmp_path / "new_dataset") and s["block_size"] == block
    assert (path.parent / "server_max_model_len.txt").read_text().strip() == "3328"


@pytest.mark.parametrize("overrides", [dict(HEAD_LR="nan"), dict(GUARD_MODE="unknown"), dict(TRAIN_NPUS="0,3,4,5,6"),
    dict(PREFIX_LAYERS="2"), dict(GUARD_START_STEPS="100"), dict(PREFIX_TOP_K="32")])
def test_bad_settings_fail_before_output(tmp_path, overrides):
    env = recipe_fixture(tmp_path)
    with pytest.raises(ValueError):
        helper("prepare_experiment").prepare(dict(env, **overrides))
    assert not Path(env["OUTPUT_DIR"]).exists()


def test_lr_schedule_is_full_horizon_and_launcher_is_valid():
    entry = helper("train_entry")
    assert entry.head_lr_multiplier(0, 10000, 100, 100) == 0
    assert entry.head_lr_multiplier(150, 10000, 100, 100) == .5
    assert entry.head_lr_multiplier(200, 10000, 100, 100) == 1
    assert entry.head_lr_multiplier(10000, 10000, 100, 100) == 0
    subprocess.run(["bash", "-n", str(ROOT / "examples/train/train_dflash_prefix_joint.sh")], check=True)
