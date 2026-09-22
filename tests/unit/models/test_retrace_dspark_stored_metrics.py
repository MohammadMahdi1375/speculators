"""Metric aggregation and update equivalence for the isolated stored loop."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from speculators.models.retrace_dspark import stored_loop
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import normalize_counted_metrics


def raw_metrics(rank):
    values = {
        key: torch.tensor(float(rank + index + 1))
        for index, key in enumerate(stored_loop.METRIC_KEYS)
    }
    values["error_records_sum"] = torch.tensor(rank, dtype=torch.int32)
    values["error_records_total"] = torch.tensor(1.0 if rank == 0 else 0.0)
    return values


def test_six_rank_packed_sums_keep_weighted_metrics_and_local_loss(monkeypatch):
    ranks = [raw_metrics(rank) for rank in range(6)]
    expected = {
        key: sum(float(row[key]) for row in ranks) for key in stored_loop.METRIC_KEYS
    }
    original = {key: value.clone() for key, value in ranks[0].items()}
    calls = []

    def all_reduce(packed, op):
        calls.append((packed.shape, packed.dtype, op))
        assert not packed.requires_grad
        for row in ranks[1:]:
            packed[:-1].add_(
                torch.tensor([float(row[key]) for key in stored_loop.METRIC_KEYS])
            )

    monkeypatch.setattr(stored_loop.dist, "all_reduce", all_reduce)
    # Input insertion order and count tensor dtypes must not affect the wire schema.
    metrics = dict(reversed(list(ranks[0].items())))
    metrics["loss_sum"].requires_grad_(True)
    result = stored_loop.sum_training_metrics(metrics, "cpu", True)
    assert result == expected
    assert normalize_counted_metrics(result.copy(), 6) == normalize_counted_metrics(
        expected.copy(), 6
    )
    assert len(calls) == 1 and calls[0][:2] == (torch.Size([21]), torch.float32)
    for key in metrics:
        torch.testing.assert_close(metrics[key], original[key])


@pytest.mark.parametrize("local_error", [True, False])
def test_schema_error_uses_same_wire_size_then_fails_every_rank(
    monkeypatch, local_error
):
    metrics = raw_metrics(0)
    if local_error:
        del metrics["loss_total"]

    def all_reduce(packed, op):
        assert packed.shape == (21,)
        if local_error:
            assert packed[-1] == 1
        else:
            packed[-1] += 1  # Simulate a schema error reported by another rank.

    monkeypatch.setattr(stored_loop.dist, "all_reduce", all_reduce)
    with pytest.raises(RuntimeError, match="schema mismatch"):
        stored_loop.sum_training_metrics(metrics, "cpu", True)


def test_missing_gradients_are_reported_before_optimizer():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="weight, bias"):
        stored_loop.check_gradients(model)
    model(torch.ones(1, 2)).sum().backward()
    stored_loop.check_gradients(model)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, features, **_):
        loss = self.projection(features).float().square().mean()
        metrics = raw_metrics(0)
        metrics.pop("error_records_sum")
        metrics.pop("error_records_total")
        metrics["loss_sum"] = loss.detach()
        return None, loss, metrics


class Loader(list):
    batch_sampler = SimpleNamespace(set_epoch=lambda epoch: None)


class TinyTrainer(Trainer):
    def __init__(self, model, root):
        self.model = model
        self.config = TrainerConfig(
            lr=0.01,
            num_epochs=1,
            save_path=str(root),
            checkpoint_freq=0.5,
            hidden_states_dtype=torch.bfloat16,
            log_freq=1,
        )
        self.local_rank, self.rank, self.device_type = "cpu", 1, "cpu"
        self.is_distributed = False
        self.global_step = 0
        self.train_loader = Loader(
            [
                {
                    "features": torch.tensor([[index + 1.0, 2.0]]),
                    "document_ids": torch.zeros(1, 2, dtype=torch.long),
                    "error_records": 0,
                }
                for index in range(4)
            ]
        )
        self.optimizers = [torch.optim.AdamW(self.model.parameters(), lr=0.01)]
        self.schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers[0], gamma=0.9)
        ]
        self.saves = []

    def maybe_save_checkpoint(self, epoch, local_step=0):
        self.saves.append((epoch, local_step, self.global_step))


def test_isolated_loop_matches_stock_updates_scheduler_and_save_boundaries(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RETRACE_DSPARK_STORED_TRACE_STEPS", "1")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    torch.manual_seed(6)
    model = TinyTrainingModel()
    baseline = TinyTrainer(copy.deepcopy(model), tmp_path / "baseline/checkpoints")
    isolated = TinyTrainer(copy.deepcopy(model), tmp_path / "isolated/checkpoints")
    baseline.train_epoch(0)
    stored_loop.train_stored_epoch(isolated, 0)
    assert baseline.global_step == isolated.global_step == 4
    assert baseline.saves == isolated.saves == [(0, 2, 2)]
    assert baseline.schedulers[0].state_dict() == isolated.schedulers[0].state_dict()
    for name, value in baseline.model.state_dict().items():
        torch.testing.assert_close(
            value, isolated.model.state_dict()[name], rtol=0, atol=0
        )
    a, b = baseline.optimizers[0].state_dict(), isolated.optimizers[0].state_dict()
    assert a["param_groups"] == b["param_groups"]
    for key, state in a["state"].items():
        for item, value in state.items():
            torch.testing.assert_close(value, b["state"][key][item], rtol=0, atol=0)
    logs = list((tmp_path / "isolated/rank_diagnostics").glob("*.jsonl"))
    assert len(logs) == 1
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert {record["global_step"] for record in records} == {0}
    stages = [record["stage"] for record in records]
    assert stages.index("backward_synced") < stages.index("optimizer_done")
    assert stages.index("metrics_all_reduce_done") < stages.index("update_done")
