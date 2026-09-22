import copy

import pytest
import torch

from speculators.models.retrace.batched_training import (
    BatchedDraftCache,
    BatchedTarget,
    collect_batched_trajectories,
)
from speculators.models.retrace.clean_training import (
    backward_trajectory,
    collect_trajectory,
)
from speculators.models.retrace.target import LocalTarget

from .test_retrace_performance import models


def test_batched_target_padding_rollback_and_row_removal(tmp_path):
    target, draft = models(tmp_path)
    prompts = [[1, 2, 3], [11, 12, 13, 14, 15, 16, 17]]
    teachers = [LocalTarget(target, draft.target_layer_ids) for _ in prompts]
    batch = BatchedTarget(LocalTarget(target, draft.target_layer_ids), 40)
    ids = torch.zeros(2, 7, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        ids[i, : len(prompt)] = torch.tensor(prompt)
    pref = batch.states(ids, [3, 7], prefill=True)
    for i, (teacher, prompt) in enumerate(zip(teachers, prompts, strict=True)):
        expected = teacher.states(prompt)
        torch.testing.assert_close(
            pref.auxiliary[i, : len(prompt)], expected.auxiliary[0]
        )
        torch.testing.assert_close(pref.scoring[i, : len(prompt)], expected.scoring[0])
    queries = torch.tensor([[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]])
    actual = batch.states(queries, [3, 7])
    for i, teacher in enumerate(teachers):
        expected = teacher.states(prompts[i] + queries[i].tolist())
        torch.testing.assert_close(actual.scoring[i], expected.scoring[0, -5:])
    # Different rejection points, with a corrected token different from the
    # previous branch. Compare against each isolated target's real rollback.
    retained = [1, 4]
    committed = [
        p + q[:n].tolist() for p, q, n in zip(prompts, queries, retained, strict=True)
    ]
    next_query = torch.tensor([[40, 41, 42, 43, 44], [35, 36, 37, 38, 39]])
    actual = batch.states(next_query, [len(p) for p in committed])
    for i, teacher in enumerate(teachers):
        expected = teacher.states(committed[i] + next_query[i].tolist())
        torch.testing.assert_close(actual.scoring[i], expected.scoring[0, -5:])
        torch.testing.assert_close(actual.auxiliary[i], expected.auxiliary[0, -5:])
    # Request identity follows the cache when another request finishes.
    batch.select(torch.tensor([1]))
    prefix = committed[1] + next_query[1, :2].tolist()
    final = torch.tensor([[2, 5, 8]])
    actual = batch.states(final, [len(prefix)])
    expected = teachers[1].states(prefix + final[0].tolist())
    torch.testing.assert_close(actual.scoring[0], expected.scoring[0, -3:])
    with pytest.raises(ValueError, match="capacity"):
        batch.states(final, [39])


@pytest.mark.parametrize("eos", [set(), {47}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}])
def test_complete_batch_matches_serial_trajectories_and_memory(tmp_path, eos):
    target, draft = models(tmp_path)
    prompts = [[1, 2, 3], [11, 12, 13, 14, 15, 16, 17], [30, 31]]
    teacher = LocalTarget(target, draft.target_layer_ids)
    reference = [
        collect_trajectory(
            teacher,
            draft,
            p,
            31,
            eos,
            0.6,
            performance_mode="cached",
            trace_storage="device",
        )
        for p in prompts
    ]
    actual = collect_batched_trajectories(teacher, draft, prompts, 31, eos, 0.6)
    assert actual.target_forwards == 1 + max(len(t.positions) for t in actual.traces)
    assert actual.draft_forwards == actual.target_forwards - 1
    assert actual.target_forwards < sum(t.target_calls for t in actual.traces)
    for a, b in zip(actual.traces, reference, strict=True):
        assert (
            a.tokens,
            a.positions,
            a.accepted,
            a.committed,
            a.proposed,
            a.target_calls,
        ) == (
            b.tokens,
            b.positions,
            b.accepted,
            b.committed,
            b.proposed,
            b.target_calls,
        )
        torch.testing.assert_close(a.context, b.context, atol=3e-6, rtol=1e-4)
        for ma, mb in zip(a.memories, b.memories, strict=True):
            assert torch.equal(ma.valid, mb.valid)
            assert ma.draft.dtype == ma.target.dtype == torch.float16
            assert not ma.draft.requires_grad and not ma.target.requires_grad
            torch.testing.assert_close(ma.draft, mb.draft, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(ma.target, mb.target, atol=2e-3, rtol=2e-3)
    # Same per-prompt objective and every prompt contributes exactly once.
    other = copy.deepcopy(draft)
    loss_a = sum(
        backward_trajectory(
            draft.train(), t, 0.6, blocks_per_forward=4, optimized=True
        )["loss"]
        for t in actual.traces
    )
    loss_b = sum(
        backward_trajectory(
            other.train(), t, 0.6, blocks_per_forward=4, optimized=True
        )["loss"]
        for t in reference
    )
    assert abs(loss_a - loss_b) < 2e-5
    for a, b in zip(draft.parameters(), other.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-4, rtol=2e-3)


def test_immediate_eos_and_single_token_budget(tmp_path):
    target, draft = models(tmp_path)
    teacher = LocalTarget(target, draft.target_layer_ids)
    for budget, eos in [(1, set()), (12, set(range(48)))]:
        batch = collect_batched_trajectories(
            teacher, draft, [[1], [4, 2, 6]], budget, eos, 1
        )
        assert batch.target_forwards == 1
        assert batch.draft_forwards == 0
        assert all(not t.positions for t in batch.traces)
        assert all(len(t.tokens) == t.prompt_length + 1 for t in batch.traces)


def test_batched_cache_rejects_gradient_forwards_and_updated_parameters(tmp_path):
    _, draft = models(tmp_path)
    cache = BatchedDraftCache(draft, 20)
    with pytest.raises(RuntimeError, match="no-grad"):
        cache.update(torch.randn(1, 3, 64), torch.arange(3)[None])
    with torch.no_grad():
        draft.fc.weight.add_(0.01)
        with pytest.raises(RuntimeError, match="parameter update"):
            cache.update(torch.randn(1, 3, 64), torch.arange(3)[None])


def test_all_accepted_bonus_and_truncated_final_block(tmp_path):
    target, draft = models(tmp_path)
    with torch.no_grad():
        target.lm_head.weight.zero_()
        draft.lm_head.weight.zero_()
    teacher = LocalTarget(target, draft.target_layer_ids)
    prompts = [[2, 3], [9, 8, 7, 6]]
    expected = [
        collect_trajectory(teacher, draft, p, 19, set(), 1, performance_mode="cached")
        for p in prompts
    ]
    actual = collect_batched_trajectories(teacher, draft, prompts, 19, set(), 1)
    for a, b in zip(actual.traces, expected, strict=True):
        assert a.tokens == b.tokens
        assert a.positions == b.positions
        assert a.accepted == b.accepted == 15
        assert a.committed == b.committed == 18
        assert all(not m.valid.any() for m in a.memories)
        torch.testing.assert_close(a.context, b.context)


def test_batched_bfloat16_with_fp32_trainable_parameters(tmp_path):
    target, draft = models(tmp_path)
    target.bfloat16()
    draft.bfloat16()
    for p in draft.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    teacher = LocalTarget(target, draft.target_layer_ids)
    prompts = [[1, 3, 4], [21, 22, 23, 24, 25]]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        padded = torch.tensor([[1, 3, 4, 0, 0], prompts[1]])
        batch = BatchedTarget(teacher, 32)
        actual = batch.states(padded, [3, 5], prefill=True)
        for i, prompt in enumerate(prompts):
            teacher.reset()
            expected = teacher.states(prompt)
            torch.testing.assert_close(
                actual.scoring[i, : len(prompt)],
                expected.scoring[0],
                atol=0.04,
                rtol=0.03,
            )
        result = collect_batched_trajectories(teacher, draft, prompts, 19, set(), 1)
    draft.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        values = [
            backward_trajectory(draft, trace, 1, blocks_per_forward=8, optimized=True)
            for trace in result.traces
        ]
    assert all(v["labels"] > 0 for v in values)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in draft.parameters()
    )


def test_resume_into_batched_mode_preserves_schedule_optimizer_and_epoch_saves(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    import sys

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from speculators.models.retrace import train_pretrained
    from speculators.models.retrace.pretrained import prepare

    from .test_retrace_performance import fixture

    _, source, _ = fixture(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"t{i}": i for i in range(48)}, unk_token="t0")
        ),
        unk_token="t0",
    )
    tokenizer.save_pretrained(tmp_path / "target")
    prepare(source, str(tmp_path / "target"), tmp_path / "initial")
    pool = tmp_path / "prompts.jsonl"
    prompts = [[1, 2, 3], [4, 5, 6, 7], [31, 32], [17, 16, 15]]
    pool.write_text(
        "".join(
            json.dumps(
                {
                    "input_ids": p,
                    "sha256": hashlib.sha256(
                        json.dumps(p, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            )
            + "\n"
            for p in prompts
        )
    )
    common = [
        "train",
        "--target",
        str(tmp_path / "target"),
        "--draft",
        str(tmp_path / "initial/retrace"),
        "--pool",
        str(pool),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--epochs",
        "2",
        "--global-prompt-batch",
        "2",
        "--response-length",
        "19",
        "--performance-mode",
        "cached",
        "--trace-storage",
        "device",
        "--blocks-per-forward",
        "8",
        "--profile-performance",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        common + ["--output", str(tmp_path / "serial"), "--stop-after", "1"],
    )
    train_pretrained.main()
    checkpoint = tmp_path / "serial/step_000001"
    state_path = checkpoint / "training_state.json"
    state = json.loads(state_path.read_text())
    state["signature"].pop("rollout_batch_size")
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(
        sys,
        "argv",
        common
        + [
            "--output",
            str(tmp_path / "batched"),
            "--resume",
            str(checkpoint),
            "--rollout-batch-size",
            "2",
            "--allow-performance-change",
        ],
    )
    train_pretrained.main()
    rows = [
        json.loads(line)
        for line in (tmp_path / "batched/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in rows] == [2, 3, 4]
    assert all(r["global_prompts"] == 2 and r["rollout_batch_size"] == 2 for r in rows)
    assert all(r["target_forwards"] < r["target_calls"] for r in rows)
    assert [r["lr"] for r in rows] == [
        train_pretrained.learning_rate(s, 4, 5e-5, 0.05) for s in (1, 2, 3)
    ]
    assert all(
        r["gradient_sync_seconds_max"] >= 0 and r["optimizer_seconds_max"] >= 0
        for r in rows
    )
    assert (tmp_path / "batched/step_000002/optimizer.pt").is_file()
    assert (tmp_path / "batched/step_000004/master_parameters.pt").is_file()
    optimizer = torch.load(
        tmp_path / "batched/step_000004/optimizer.pt", weights_only=True
    )
    assert all(v["step"] == 4 for v in optimizer["state"].values())
    changes = json.loads((tmp_path / "batched/resume_command.json").read_text())[
        "resume_execution_changes"
    ]
    assert changes == {"rollout_batch_size": {"before": 1, "after": 2}}


def test_batched_benchmark_uses_same_checkpoint_pool_and_updates(tmp_path):
    from speculators.models.retrace.batched_benchmark import commands_for
    from speculators.models.retrace.performance import validate_resume_signature

    signature = dict(
        world_size=8,
        total_steps=10000,
        target="/target",
        prompt_length=512,
        response_length=1024,
        global_prompt_batch=32,
        epochs=8,
        lr=5e-5,
        lr_warmup_ratio=0.05,
        weight_decay=0.01,
        gamma=4.0,
        seed=42,
        dtype="bfloat16",
        disable_conditioning=False,
        performance_mode="cached",
        trace_storage="device",
        blocks_per_forward=16,
    )
    commands = commands_for(
        signature, "/checkpoint/step_000006", "/pool", tmp_path, 12, 4, 16
    )
    for name, size in (("serial", 1), ("batched", 4)):
        command = commands[name]
        assert command[command.index("--resume") + 1] == "/checkpoint/step_000006"
        assert command[command.index("--pool") + 1] == "/pool"
        assert command[command.index("--max-steps") + 1] == "10000"
        assert command[command.index("--stop-after") + 1] == "12"
        assert command[command.index("--rollout-batch-size") + 1] == str(size)
    with pytest.raises(ValueError, match="allow-performance-change"):
        validate_resume_signature(signature, signature | {"rollout_batch_size": 4})
    assert set(
        validate_resume_signature(
            signature, signature | {"rollout_batch_size": 4}, True
        )
    ) == {"rollout_batch_size"}
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(
            signature,
            signature | {"global_prompt_batch": 64, "rollout_batch_size": 4},
            True,
        )
