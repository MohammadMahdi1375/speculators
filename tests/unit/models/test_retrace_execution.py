import copy
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from speculators.models.retrace.clean_training import (
    backward_trajectory,
    collect_trajectory,
)
from speculators.models.retrace.execution import (
    PromptQueue,
    set_draft_attention,
    static_prompt_groups,
    validate_prompt_coverage,
)
from speculators.models.retrace.execution_check import attention_case
from speculators.models.retrace.target import LocalTarget

from .test_retrace_performance import fixture, models


@pytest.mark.parametrize("case", ["prefill", "rollback", "draft", "packed_blocks"])
def test_sdpa_operator_masks_and_gradients(case):
    assert attention_case(case, torch.device("cpu"), torch.float32)["passed"]


def test_sdpa_complete_trace_and_packed_gradient_match_eager(tmp_path):
    target, draft = models(tmp_path)
    fast_target, fast_draft = copy.deepcopy(target), copy.deepcopy(draft)
    fast_target.set_attn_implementation("sdpa")
    set_draft_attention(fast_draft, "sdpa")
    traces = [
        collect_trajectory(
            LocalTarget(t, d.target_layer_ids),
            d,
            [1, 3, 7],
            27,
            set(),
            0.8,
            performance_mode="cached",
            trace_storage="device",
        )
        for t, d in ((target, draft), (fast_target, fast_draft))
    ]
    a, b = traces
    assert (a.tokens, a.positions, a.accepted, a.target_calls) == (
        b.tokens,
        b.positions,
        b.accepted,
        b.target_calls,
    )
    for x, y in zip(a.memories, b.memories, strict=True):
        assert torch.equal(x.valid, y.valid)
        torch.testing.assert_close(x.draft, y.draft, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(x.target, y.target, atol=2e-3, rtol=2e-3)
    # Identical captured states isolate attention/gradient changes from FP16 storage rounding.
    expected = backward_trajectory(
        draft.train(), a, 0.8, blocks_per_forward=3, optimized=True
    )
    actual = backward_trajectory(
        fast_draft.train(), a, 0.8, blocks_per_forward=7, optimized=True
    )
    assert actual["labels"] == expected["labels"]
    assert abs(actual["loss"] - expected["loss"]) < 3e-6
    for (name, p), (_, q) in zip(
        draft.named_parameters(), fast_draft.named_parameters(), strict=True
    ):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=5e-4, msg=name)


def test_sdpa_batched_rollback_and_bfloat16_masters(tmp_path, monkeypatch):
    # Reuse the actual padding/crop/row-removal and mixed-precision regression cases.
    from . import test_retrace_batched_training as original

    def sdpa_models(path):
        t, d = models(path)
        t.set_attn_implementation("sdpa")
        set_draft_attention(d, "sdpa")
        return t, d

    monkeypatch.setattr(original, "models", sdpa_models)
    for name, test in (
        ("cache", original.test_batched_target_padding_rollback_and_row_removal),
        ("bf16", original.test_batched_bfloat16_with_fp32_trainable_parameters),
    ):
        directory = tmp_path / name
        directory.mkdir()
        test(directory)


def _queue_worker(store_path, output, rank, world):
    store = dist.FileStore(store_path, world + 1)
    store.set(f"ready_{rank}", "1")
    store.wait([f"ready_{r}" for r in range(world)])
    result = []
    for step, microbatch in ((13, 1), (14, 3)):
        jobs = list(PromptQueue(store).groups(list(range(101, 138)), step, microbatch))
        result.append({"step": step, "jobs": jobs})
    Path(output).write_text(json.dumps(result))


def test_filestore_multiprocess_claims_exactly_once_without_network(tmp_path):
    path = str(tmp_path / "queue.store")
    keeper = dist.FileStore(path, 4)
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_queue_worker, args=(path, str(tmp_path / f"rank_{r}.json"), r, 3)
        )
        for r in range(3)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        outputs = [
            json.loads((tmp_path / f"rank_{r}.json").read_text()) for r in range(3)
        ]
        for step_index in range(2):
            positions, prompts = [], []
            for output in outputs:
                for indices, ids in output[step_index]["jobs"]:
                    positions.extend(indices)
                    prompts.extend(ids)
                    assert ids == [p + 101 for p in indices]
            assert sorted(positions) == list(range(37))
            assert sorted(prompts) == list(range(101, 138))
            validate_prompt_coverage(positions, 37, torch.device("cpu"), False)
        with pytest.raises(RuntimeError, match="exactly once"):
            validate_prompt_coverage([0, 0], 2, torch.device("cpu"), False)
        # A fresh store after resume does not inherit old claims.
        assert len(list(PromptQueue(dist.HashStore()).groups(range(37), 13, 1))) == 37
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        del keeper


def test_unequal_worker_counts_keep_global_prompt_gradient_mean(monkeypatch):
    from speculators.models.retrace.train import average_gradients

    # Include an idle rank. Summing local prompt gradients then dividing by the
    # global count must never average the differently-sized worker means.
    for count, gradient in ((0, None), (1, 2.0), (3, 22.0)):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        if gradient is not None:
            p.grad = torch.tensor([gradient])
        returns = iter([torch.tensor(4.0), torch.tensor([24.0])])
        monkeypatch.setattr(
            dist,
            "all_reduce",
            lambda tensor, values=returns: tensor.copy_(next(values)),
        )
        assert average_gradients([p], count, torch.device("cpu"), True) == 4
        assert p.grad.item() == 6.0
    assert list(static_prompt_groups(range(7), 1, 3, 2)) == [([1, 4], [1, 4])]


def test_real_resume_switches_execution_and_preserves_optimizer_epoch_schedule(
    tmp_path, monkeypatch
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from speculators.models.retrace import train_pretrained
    from speculators.models.retrace.pretrained import prepare

    _, source, _ = fixture(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"t{i}": i for i in range(48)}, unk_token="t0")
        ),
        unk_token="t0",
    )
    tokenizer.save_pretrained(tmp_path / "target")
    prepare(source, str(tmp_path / "target"), tmp_path / "initial")
    prompts = [[1, 2, 3], [4, 7, 9], [12, 14], [17, 18, 19, 20]]
    pool = tmp_path / "prompts.jsonl"
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
        "13",
        "--performance-mode",
        "cached",
        "--trace-storage",
        "device",
        "--profile-performance",
    ]
    monkeypatch.setattr(
        sys, "argv", common + ["--output", str(tmp_path / "old"), "--stop-after", "1"]
    )
    train_pretrained.main()
    checkpoint = tmp_path / "old/step_000001"
    path = checkpoint / "training_state.json"
    state = json.loads(path.read_text())
    for key in ("attention_backend", "target_norm", "work_distribution"):
        state["signature"].pop(key)
    path.write_text(json.dumps(state))
    monkeypatch.setattr(
        sys,
        "argv",
        common
        + [
            "--output",
            str(tmp_path / "fast"),
            "--resume",
            str(checkpoint),
            "--attention-backend",
            "sdpa",
            "--work-distribution",
            "dynamic",
            "--allow-performance-change",
        ],
    )
    train_pretrained.main()
    rows = [
        json.loads(line)
        for line in (tmp_path / "fast/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in rows] == [2, 3, 4]
    assert all(
        r["global_prompts"] == 2 and r["rank_performance"][0]["prompts"] == 2
        for r in rows
    )
    assert all(
        r["attention_backend"] == "sdpa" and r["work_distribution"] == "dynamic"
        for r in rows
    )
    assert [r["lr"] for r in rows] == [
        train_pretrained.learning_rate(s, 4, 5e-5, 0.05) for s in (1, 2, 3)
    ]
    assert (tmp_path / "fast/step_000002/optimizer.pt").is_file()
    assert (tmp_path / "fast/step_000004/master_parameters.pt").is_file()
    optimizer = torch.load(
        tmp_path / "fast/step_000004/optimizer.pt", weights_only=True
    )
    assert all(v["step"] == 4 for v in optimizer["state"].values())
    changes = json.loads((tmp_path / "fast/resume_command.json").read_text())[
        "resume_execution_changes"
    ]
    assert set(changes) == {"attention_backend", "work_distribution"}


def test_benchmark_preserves_recipe_and_guarded_checkpoint(tmp_path):
    from types import SimpleNamespace

    from speculators.models.retrace.execution_benchmark import (
        commands_for,
        validate_inputs,
    )
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
        target_backend="local",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    pool = run / "prompts.jsonl"
    pool.write_text('{"input_ids":[1,2]}\n')
    signature["pool_sha256"] = hashlib.sha256(pool.read_bytes()).hexdigest()
    (checkpoint / "training_state.json").write_text(
        json.dumps({"signature": signature, "step": 12})
    )
    (checkpoint / "config.json").write_text(
        json.dumps(
            {"speculators_model_type": "retrace", "pretrained_fingerprint": "test"}
        )
    )
    for file in ["master_parameters.pt", "optimizer.pt"] + [
        f"rng_rank_{r}.pt" for r in range(8)
    ]:
        (checkpoint / file).touch()
    args = SimpleNamespace(
        run=str(run),
        checkpoint=str(checkpoint),
        output=str(tmp_path / "output"),
        steps=4,
        warmup_updates=1,
        blocks=16,
        npus="0,1,2,3,4,5,6,7",
    )
    assert validate_inputs(args)[-2:] == (12, 16)
    commands = commands_for(
        signature, checkpoint, pool, tmp_path / "output", 16, 16, "npu"
    )
    for name, command in commands.items():
        fields = {
            flag: command[command.index(flag) + 1]
            for flag in (
                "--resume",
                "--max-steps",
                "--stop-after",
                "--epochs",
                "--lr",
                "--response-length",
                "--global-prompt-batch",
                "--save-every",
            )
        }
        assert fields == {
            "--resume": str(checkpoint),
            "--max-steps": "10000",
            "--stop-after": "16",
            "--epochs": "8",
            "--lr": "5e-05",
            "--response-length": "1024",
            "--global-prompt-batch": "32",
            "--save-every": "0",
        }
        assert command[command.index("--attention-backend") + 1] == (
            "sdpa" if name == "fast" else "eager"
        )
    altered = signature | {
        "attention_backend": "sdpa",
        "target_norm": "npu",
        "work_distribution": "dynamic",
    }
    assert len(validate_resume_signature(signature, altered, True)) == 3
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(signature, altered | {"response_length": 256}, True)
    (checkpoint / "rng_rank_7.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        validate_inputs(args)
