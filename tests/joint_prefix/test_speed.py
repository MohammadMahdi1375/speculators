"""Execution controls preserve the objective; packed batches are opt-in."""
import copy
import json

import pytest
import torch

from test_joint import helper, settings, recipe_fixture
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
