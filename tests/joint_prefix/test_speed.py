"""Execution controls preserve the objective; packed batches are opt-in."""
import copy
import json

import pytest
import torch

from test_joint import helper, settings, recipe_fixture, populated_guard
from speculators.losses import resolve_loss_config
from speculators.train.config import TrainConfig


def test_prepare_records_real_recomputation_and_packing_controls(tmp_path):
    env = dict(recipe_fixture(tmp_path), BACKBONE_CHECKPOINTING="0",
               SELECTOR_CHECKPOINTING="0", TOTAL_SEQ_LEN="6144", MAX_ANCHORS="1024",
               ANCHOR_CHUNK_SIZE="32", GUARD_CHUNK_SIZE="16", PREFETCH_FACTOR="2")
    path = helper("prepare_experiment").prepare(env)
    flat = TrainConfig.resolve(["--config", str(path)]).flatten()
    s = json.loads((path.parent / "joint_settings.json").read_text())
    assert not flat["gradient_checkpointing"] and not s["backbone_checkpointing"]
    assert not s["selector_checkpointing"] and not flat["fsdp_shard"]
    assert flat["total_seq_len"] == s["packed_tokens_per_rank"] == 6144
    assert flat["max_anchors"] == 1024 and flat["prefetch_factor"] == 2
    assert flat["epochs"] == 5 and flat["max_steps"] is None
    assert flat["lr"] == .0006 and s["guard_every"] == 1
    assert int((path.parent / "server_max_model_len.txt").read_text()) == 6400


@pytest.mark.parametrize("key,value", [("BACKBONE_CHECKPOINTING", "yes"),
    ("SELECTOR_CHECKPOINTING", "2"), ("TOTAL_SEQ_LEN", "0"), ("PREFETCH_FACTOR", "0")])
def test_bad_execution_controls_rejected(tmp_path, key, value):
    env = dict(recipe_fixture(tmp_path), **{key: value})
    with pytest.raises(ValueError):
        helper("prepare_experiment").prepare(env)
    assert not (tmp_path / "out").exists()


def forward_and_grads(model, batch, *, amp):
    model.train()
    model.joint_step = 10
    model.zero_grad(set_to_none=True)
    torch.manual_seed(311)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        _, loss, _ = model(**batch)
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


def test_switching_both_recomputation_paths_preserves_bf16_loss_and_gradients(tmp_path):
    torch.set_num_threads(1)
    s = dict(settings(), backbone_checkpointing=True, selector_checkpointing=True)
    original, batch = helper("check_training").make_fixture(tmp_path, s)
    with torch.no_grad():
        original.prefix_head.head_weights.fill_(.1)
        original.prefix_head.successor_projection.weight.normal_(std=.02)
    fast = copy.deepcopy(original)
    fast.gradient_checkpointing_disable()
    fast.joint_settings["selector_checkpointing"] = False
    batch["loss_config"] = resolve_loss_config("ce", "eager")
    loss_a, grads_a = forward_and_grads(original, batch, amp=True)
    loss_b, grads_b = forward_and_grads(fast, batch, amp=True)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=0, atol=0, msg=name)


def test_larger_selector_chunks_preserve_fp32_objective_and_gradients(tmp_path):
    torch.set_num_threads(1)
    s = dict(settings(), backbone_checkpointing=False, selector_checkpointing=False,
             anchor_chunk_size=1)
    original, batch = helper("check_training").make_fixture(tmp_path, s)
    original.float()
    batch = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
             for k, v in batch.items()}
    batch["loss_config"] = resolve_loss_config("ce", "eager")
    with torch.no_grad():
        original.prefix_head.head_weights.fill_(.1)
        original.prefix_head.successor_projection.weight.normal_(std=.02)
    fast = copy.deepcopy(original)
    fast.joint_settings["anchor_chunk_size"] = 32
    loss_a, grads_a = forward_and_grads(original, batch, amp=False)
    loss_b, grads_b = forward_and_grads(fast, batch, amp=False)
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-6)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=2e-4, atol=2e-6, msg=name)


def test_fast_profile_only_changes_execution_not_training_recipe(tmp_path):
    env = dict(recipe_fixture(tmp_path), AR_LOSS_WEIGHT="0.1", AR_PRETRAIN_STEPS="0")
    prepare = helper("prepare_experiment").prepare
    memory_path = prepare(dict(env, EXECUTION_PROFILE="memory"))
    fast_path = prepare(dict(env, EXECUTION_PROFILE="fast", OUTPUT_DIR=str(tmp_path / "fast")))
    memory = TrainConfig.resolve(["--config", str(memory_path)]).flatten()
    fast = TrainConfig.resolve(["--config", str(fast_path)]).flatten()
    for key in memory:
        if key not in {"gradient_checkpointing", "save_path", "hidden_states_path", "log_dir"}:
            assert memory[key] == fast[key], key
    baseline = json.loads((memory_path.parent / "joint_settings.json").read_text())
    optimized = json.loads((fast_path.parent / "joint_settings.json").read_text())
    execution_keys = set(helper("prepare_experiment").execution_settings({}))
    for key in baseline:
        if key not in execution_keys | {"output_dir"}:
            assert baseline[key] == optimized[key], key
    assert not any(optimized[k] for k in ("backbone_checkpointing", "selector_checkpointing", "ar_logit_checkpointing"))
    assert optimized["anchor_chunk_size"] == 8
    assert optimized["ddp_gradient_as_bucket_view"] and optimized["guard_reuse_workspaces"]


def test_resume_execution_override_does_not_use_fresh_run_defaults():
    resolve = helper("prepare_experiment").execution_settings
    inherited = dict(settings(), **resolve({"EXECUTION_PROFILE": "memory"}), ar_loss_weight=.1)
    result = resolve({"EXECUTION_PROFILE": "fast", "AR_LOSS_WEIGHT": ".9"},
                     prefix="RESUME_", inherited=inherited)
    assert result == {key: inherited[key] for key in result}
    fast = resolve({"RESUME_EXECUTION_PROFILE": "fast", "RESUME_AR_LOGIT_CHECKPOINTING": "1"},
                   prefix="RESUME_", inherited=inherited)
    assert fast["ar_logit_checkpointing"] and not fast["backbone_checkpointing"]
    assert fast["anchor_chunk_size"] == 8 and "ar_loss_weight" not in fast
    custom = resolve({"RESUME_EXECUTION_PROFILE": "fast"}, prefix="RESUME_",
                     inherited=dict(inherited, anchor_chunk_size=3))
    assert custom["anchor_chunk_size"] == 3


@pytest.mark.parametrize("amp", [False, True])
def test_joint_ar_fast_profile_matches_loss_gradients_and_adam_step(tmp_path, amp):
    torch.set_num_threads(1)
    resolve = helper("prepare_experiment").execution_settings
    s = dict(settings(), **resolve({"EXECUTION_PROFILE": "memory"}), ar_loss_weight=.1,
             ar_max_anchors=4, ar_logit_chunk_size=7)
    original, batch = helper("check_training").make_fixture(tmp_path, s)
    with torch.no_grad():
        original.prefix_head.head_weights.fill_(.1)
        original.prefix_head.successor_projection.weight.normal_(std=.02)
    if not amp:
        original.float()
        batch = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                 for k, v in batch.items()}
    fast = copy.deepcopy(original)
    fast.gradient_checkpointing_disable()
    fast.joint_settings.update(resolve({"EXECUTION_PROFILE": "fast"}))
    # Both profiles retain the same chunk geometry to avoid BF16 GEMM drift.
    # Exercise multiple chunks without changing the number of sampled anchors.
    original.joint_settings["anchor_chunk_size"] = fast.joint_settings["anchor_chunk_size"] = 2
    batch["loss_config"] = resolve_loss_config("ce", "eager")
    loss_a, grads_a = forward_and_grads(original, batch, amp=amp)
    loss_b, grads_b = forward_and_grads(fast, batch, amp=amp)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=0, atol=0, msg=name)
    for model in (original, fast):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4).step()
    for (name, a), (_, b) in zip(original.named_parameters(), fast.named_parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


def test_reused_guard_workspaces_match_fresh_copies_across_updates(tmp_path):
    model, batch = helper("check_training").make_fixture(tmp_path, settings())
    guard = populated_guard(model, batch)
    fast = copy.deepcopy(model)
    cached = fast.selector_guard
    cached.reuse_workspaces = True
    opts = [torch.optim.AdamW(m.prefix_head.parameters(), lr=1e-4) for m in (model, fast)]
    previous = None
    for step in range(3):
        torch.manual_seed(120 + step)
        for a, b in zip(model.prefix_head.parameters(), fast.prefix_head.parameters()):
            a.grad = torch.randn_like(a) * .01
            b.grad = a.grad.clone()
        before = {n: p.grad.clone() for n, p in fast.prefix_head.named_parameters()}
        reports = [g.step(m.prefix_head, o, m.embed_tokens.weight)
                   for g, m, o in zip((guard, cached), (model, fast), opts)]
        assert reports[0] == reports[1] and reports[1]["protected_positions"] > 0
        for (name, a), (_, b) in zip(model.prefix_head.named_parameters(), fast.prefix_head.named_parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
            torch.testing.assert_close(b.grad, before[name], rtol=0, atol=0)
        for a, b in zip(opts[0].state.values(), opts[1].state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        identities = tuple(id(value[1]) for value in cached._workspaces.values())
        if step:
            assert identities == previous
        previous = identities
    live = dict(fast.prefix_head.named_parameters())
    for _, shadow in cached._workspaces.values():
        for name, p in shadow.named_parameters():
            assert p.data_ptr() != live[name].data_ptr()


def test_packed_metrics_preserve_values_and_reduce_collective_count(monkeypatch):
    import speculators.train.trainer as trainer_module
    metrics = {f"metric_{i}": torch.tensor(float(i), requires_grad=True) for i in range(64)}
    metrics["large_count"] = torch.tensor(2**60 + 7, dtype=torch.int64)
    metrics["double"] = torch.tensor(1.2345678901234567, dtype=torch.float64)
    before = {key: value.detach().clone() for key, value in metrics.items()}
    calls = []
    def reduce(value, op):
        calls.append((value.dtype, value.numel()))
        value.mul_(2)
    monkeypatch.setattr(trainer_module.dist, "all_reduce", reduce)
    result = trainer_module._reduce_training_metrics(metrics, True)
    assert len(calls) == 3  # one per dtype, instead of 66 scalar collectives
    for key in metrics:
        assert result[key] == 2 * before[key].item()
        torch.testing.assert_close(metrics[key].detach(), before[key], atol=0, rtol=0)
