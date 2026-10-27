"""Causality, objective, gradient and DDP regressions for joint AR training."""

import copy
import datetime as dt
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel

from scripts.dflash_prefix_joint.check_training import make_fixture
from speculators.losses import resolve_loss_config
from speculators.models.attention import create_float_mask
from speculators.models.dflash_prefix import joint_training
from speculators.models.dflash_prefix.ar_training import (
    ar_backbone_hidden,
    causal_ar_loss,
    create_ar_block_mask_mod,
    ddp_token_mean,
    validate_ar_settings,
)


def settings(**overrides):
    return {
        "block_size": 8,
        "rank": 16,
        "heads": 2,
        "top_k": 16,
        "compiled_logit_cap": 8.0,
        "delay_steps": 0,
        "head_warmup_steps": 1,
        "focus_start_steps": 0,
        "focus_warmup_steps": 1,
        "loss_alpha": 0.25,
        "dense_floor": 0.25,
        "anchor_chunk_size": 2,
        "backbone_checkpointing": False,
        "selector_checkpointing": False,
        "guard_mode": "off",
        "guard_start_steps": 0,
        "guard_replay_anchors": 4,
        "guard_add_anchors": 4,
        "guard_backtracks": 2,
        "guard_chunk_size": 2,
    } | overrides


def fixture(directory, **overrides):
    model, batch = make_fixture(directory, settings(**overrides))
    model.float()
    batch = {
        key: value.float()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in batch.items()
    }
    batch["loss_config"] = resolve_loss_config("ce", "eager")
    return model, batch


@pytest.mark.parametrize("window", [None, 2])
def test_mask_is_causal_and_document_and_anchor_isolated(window):
    docs = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, -1])
    anchors = torch.tensor([3, 8])
    mask, q_len, kv_len = create_ar_block_mask_mod(docs, 10, anchors, 3, window)
    zero = torch.tensor(0)
    for query in range(q_len):
        anchor = int(anchors[query // 3])
        for key in range(kv_len):
            prefix = (
                key < anchor
                and key < 10
                and docs[key] == docs[anchor]
                and (window is None or key >= anchor - window)
            )
            clean = 10 + (query // 3) * 3 <= key <= 10 + query
            actual = mask(zero, zero, torch.tensor(query), torch.tensor(key))
            assert bool(actual) == bool(prefix or clean)


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_predictions_cannot_read_current_label_or_future_features(tmp_path, backend):
    model, batch = fixture(tmp_path)
    # Two layers exercise indirect leakage through earlier hidden states too.
    layer = copy.deepcopy(model.layers[0])
    model.layers.append(layer)
    for decoder in model.layers:
        decoder.self_attn.config._attn_implementation = backend
    if backend == "eager":
        model._create_mask_fn = create_float_mask
    model.eval()
    indices = torch.arange(4, 12).unsqueeze(0)
    args = (
        model,
        batch["hidden_states"],
        batch["input_ids"],
        batch["document_ids"],
        indices,
    )
    with torch.no_grad():
        before = ar_backbone_hidden(*args)
        changed_tokens = batch["input_ids"].clone()
        changed_tokens[:, 7] = (changed_tokens[:, 7] + 1) % 64
        after = ar_backbone_hidden(
            model,
            batch["hidden_states"],
            changed_tokens,
            batch["document_ids"],
            indices,
        )
        # Output 2 predicts token 7; its own label cannot enter the residual
        # stream or any attention layer. Output 3 can see token 7.
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=0, rtol=0)
        assert not torch.equal(before[:, 3:], after[:, 3:])
        changed_features = batch["hidden_states"].clone()
        changed_features[:, 4:] += torch.randn_like(changed_features[:, 4:]) * 10
        after = ar_backbone_hidden(
            model, changed_features, batch["input_ids"], batch["document_ids"], indices
        )
        torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_ar_loss_has_shifted_labels_and_token_normalization(tmp_path):
    model, batch = fixture(
        tmp_path,
        ar_loss_weight=0.1,
        ar_max_anchors=2,
        ar_logit_chunk_size=3,
        ar_logit_checkpointing=False,
    )
    blocks = torch.stack((torch.arange(4, 12), torch.arange(20, 28)))
    valid = torch.tensor([[True] * 7, [True, True] + [False] * 5])
    hidden = ar_backbone_hidden(
        model, batch["hidden_states"], batch["input_ids"], batch["document_ids"], blocks
    )
    logits = model.lm_head(hidden).reshape(-1, 64)
    labels = batch["input_ids"][0, blocks[:, 1:].reshape(-1)]
    expected = (
        functional.cross_entropy(logits, labels, reduction="none") * valid.reshape(-1)
    ).sum() / 9
    loss, metrics = causal_ar_loss(
        model,
        batch["hidden_states"],
        batch["input_ids"],
        batch["document_ids"],
        blocks.reshape(-1),
        valid,
    )
    torch.testing.assert_close(loss, expected)
    assert metrics["ar_ce_total"] == 9
    loss.backward()
    for module in (model.layers, model.fc, model.hidden_norm, model.norm):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(p.grad is None for p in model.prefix_head.parameters())
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert model.verifier_lm_head.weight.grad is None


def test_no_valid_ar_labels_returns_connected_finite_zero(tmp_path):
    model, batch = fixture(tmp_path, ar_loss_weight=0.1)
    blocks = torch.arange(4, 12)
    loss, metrics = causal_ar_loss(
        model,
        batch["hidden_states"],
        batch["input_ids"],
        batch["document_ids"],
        blocks,
        torch.zeros(1, 7, dtype=torch.bool),
    )
    assert loss.item() == 0
    assert metrics["ar_ce_total"].item() == 0
    loss.backward()
    assert model.fc.weight.grad is not None
    assert torch.count_nonzero(model.fc.weight.grad) == 0


def test_disabled_ar_preserves_loss_gradients_rng_and_state_keys(tmp_path):
    model, batch = fixture(tmp_path)
    other = copy.deepcopy(model).configure_joint(settings(ar_loss_weight=0.0))
    assert set(model.state_dict()) == set(other.state_dict())
    records = []
    for current in (model, other):
        current.train()
        current.joint_step = 10
        torch.manual_seed(900)
        _, loss, metrics = current(**batch)
        loss.backward()
        records.append(
            (
                loss.detach(),
                torch.get_rng_state(),
                metrics,
                {n: p.grad for n, p in current.named_parameters() if p.requires_grad},
            )
        )
    for left, right in zip(records[0][:2], records[1][:2], strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert records[0][2].keys() == records[1][2].keys()
    for name, gradient in records[0][3].items():
        torch.testing.assert_close(gradient, records[1][3][name], atol=0, rtol=0)


def test_chunk_checkpointing_matches_gradients(tmp_path):
    model, batch = fixture(
        tmp_path,
        ar_loss_weight=0.1,
        ar_max_anchors=2,
        ar_logit_chunk_size=3,
        ar_logit_checkpointing=False,
    )
    other = copy.deepcopy(model)
    other.joint_settings["ar_logit_checkpointing"] = True
    other.gradient_checkpointing_enable()
    losses = []
    for current in (model, other):
        current.train()
        current.joint_step = 10
        torch.manual_seed(31)
        _, loss, _ = current(**batch)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses)
    for (name, left), (_, right) in zip(
        model.named_parameters(), other.named_parameters(), strict=True
    ):
        if left.requires_grad:
            assert left.grad is not None, name
            assert torch.isfinite(left.grad).all(), name
            torch.testing.assert_close(left.grad, right.grad)


def test_warmup_switches_objective_at_saved_step(tmp_path):
    model, batch = fixture(
        tmp_path, ar_loss_weight=0.1, ar_pretrain_steps=2, delay_steps=2
    )
    model.train()
    for step in (0, 1, 2, 4):
        model.joint_step = step
        model.zero_grad(set_to_none=True)
        _, loss, metrics = model(**batch)
        ar = metrics["ar_ce_sum"] / metrics["ar_ce_total"]
        if step < 2:
            torch.testing.assert_close(loss.detach(), ar)
            assert metrics["parallel_weight_sum"] == 0
            assert metrics["selector_alpha_sum"] == 0
        else:
            expected = (
                metrics["unary_loss_sum"]
                + 0.1 * ar
                + metrics["selector_alpha_sum"] * metrics["selector_ce_sum"]
            )
            torch.testing.assert_close(loss.detach(), expected)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        if step < 2:
            assert all(
                torch.count_nonzero(p.grad) == 0 for p in model.prefix_head.parameters()
            )


def test_joint_ar_stops_at_document_and_loss_boundaries(tmp_path, monkeypatch):
    model, batch = fixture(tmp_path, ar_loss_weight=0.1)

    original = joint_training.causal_ar_loss
    captured = []

    def capture(model, hidden, ids, docs, indices, valid, positions):  # noqa: PLR0917
        captured.append((indices.reshape(-1, 8), valid))
        return original(model, hidden, ids, docs, indices, valid, positions)

    monkeypatch.setattr(joint_training, "causal_ar_loss", capture)
    batch["document_ids"][:, 10:25] = 1
    batch["document_ids"][:, 25:] = -1
    batch["loss_mask"][:, 15:18] = 0
    model(**batch)
    blocks, valid = captured[0]
    for block, mask in zip(blocks, valid, strict=True):
        previous = True
        for offset in range(1, 8):
            previous = (
                previous
                and bool(batch["loss_mask"][0, block[offset]])
                and batch["document_ids"][0, block[offset]]
                == batch["document_ids"][0, block[0]]
                and batch["document_ids"][0, block[0]] != -1
            )
            if not previous:
                assert not mask[offset - 1]


@pytest.mark.parametrize(
    "invalid",
    [
        {"ar_loss_weight": float("nan")},
        {"ar_loss_weight": -1},
        {"ar_pretrain_steps": 1},
        {"ar_max_anchors": 0},
        {"ar_logit_chunk_size": 0},
        {"ar_logit_checkpointing": "false"},
    ],
)
def test_invalid_settings_fail_early(invalid):
    with pytest.raises(ValueError):
        validate_ar_settings(invalid)


def _distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory) / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=dt.timedelta(seconds=60),
    )
    try:
        # Unequal token counts, including an empty rank, must match one global
        # token mean after DDP averages the local parameter gradients.
        for counts in ((1, 3), (0, 3)):
            parameter = torch.tensor(2.0, requires_grad=True)
            numerator = parameter * ((rank + 1) * counts[rank])
            loss = ddp_token_mean(numerator, torch.tensor(float(counts[rank])))
            loss.backward()
            gradient = parameter.grad.clone()
            dist.all_reduce(gradient)
            gradient /= 2
            expected = (counts[0] + 2 * counts[1]) / sum(counts)
            torch.testing.assert_close(gradient, torch.tensor(expected))

        local = Path(directory) / str(rank)
        local.mkdir()
        model, batch = fixture(
            local,
            ar_loss_weight=0.1,
            ar_pretrain_steps=1,
            delay_steps=1,
            backbone_checkpointing=True,
            selector_checkpointing=True,
        )
        model.train()
        batch["loss_mask"][:, 18:] = rank
        wrapped = DistributedDataParallel(model)
        for step in range(3):
            model.joint_step = step
            wrapped.zero_grad(set_to_none=True)
            _, loss, _ = wrapped(**batch)
            loss.backward()
            for parameter in model.parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None
                    assert torch.isfinite(parameter.grad).all()
                    expected = parameter.grad.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(parameter.grad, expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("RUN_JOINT_DDP_TEST") != "1",
    reason="Requires local Gloo sockets; use RUN_JOINT_DDP_TEST=1 on a host",
)
def test_two_rank_ddp_with_shared_checkpointed_layers(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)


@pytest.mark.parametrize("counts", [(1, 3), (0, 3), (0, 0)])
def test_ddp_token_scaling_matches_global_mean_without_sockets(monkeypatch, counts):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dist, "all_reduce", lambda tensor, **kwargs: tensor.fill_(sum(counts))
    )
    gradients = []
    for rank in range(2):
        parameter = torch.tensor(2.0, requires_grad=True)
        loss = ddp_token_mean(
            parameter * ((rank + 1) * counts[rank]), torch.tensor(float(counts[rank]))
        )
        loss.backward()
        gradients.append(parameter.grad)
    expected = (counts[0] + 2 * counts[1]) / max(1, sum(counts))
    torch.testing.assert_close(torch.stack(gradients).mean(), torch.tensor(expected))
