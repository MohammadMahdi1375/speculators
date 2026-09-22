import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.retrace.clean_training import (
    backward_trajectory,
    collect_trajectory,
)
from speculators.models.retrace.config import ReTraceSpeculatorConfig
from speculators.models.retrace.core import ReTraceDraftModel
from speculators.models.retrace.memory import ReTraceMemory
from speculators.models.retrace.performance import (
    DraftContextCache,
    validate_resume_signature,
)
from speculators.models.retrace.pretrained import SHARED, load_dflash
from speculators.models.retrace.target import LocalTarget
from speculators.proposals.greedy import GreedyTokenProposalConfig

torch.set_num_threads(1)


def fixture(tmp_path):
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["full_attention"] * 4,
        eos_token_id=None,
        pad_token_id=0,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    target.save_pretrained(tmp_path / "target")
    tl = copy.deepcopy(config)
    tl.num_hidden_layers, tl.layer_types = 2, ["full_attention"] * 2
    draft = ReTraceDraftModel(
        ReTraceSpeculatorConfig(
            transformer_layer_config=tl,
            draft_vocab_size=48,
            block_size=5,
            mask_token_id=47,
            aux_hidden_state_layer_ids=[1, 3],
            speculators_config=SpeculatorsConfig(
                algorithm="retrace",
                default_proposal_method="greedy",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=4)],
                verifier=VerifierConfig.from_config(config, name_or_path=None),
            ),
        )
    )
    source = tmp_path / "source"
    source.mkdir()
    raw = tl.to_dict()
    raw.update(
        architectures=["DFlashDraftModel"],
        block_size=5,
        num_target_layers=4,
        dflash_config={"target_layer_ids": [0, 2], "mask_token_id": 47},
    )
    (source / "config.json").write_text(json.dumps(raw))
    body = {
        name: value.clone().contiguous()
        for name, value in draft.state_dict().items()
        if name not in SHARED and not name.startswith("retrace.")
    }
    save_file(body, str(source / "model.safetensors"))
    return target, source, body


def models(tmp_path):
    target, source, _ = fixture(tmp_path)
    draft, _ = load_dflash(source, str(tmp_path / "target"), dtype=torch.float32)
    with torch.no_grad():
        draft.retrace.value.weight.normal_(std=0.02)
    return target, draft.eval()


def test_cached_proposals_match_with_nonzero_conditioning_and_project_only_new_context(
    tmp_path,
):
    target, draft = models(tmp_path)
    context = (
        LocalTarget(target, draft.target_layer_ids).states(list(range(1, 17))).auxiliary
    )
    cache = DraftContextCache(draft)
    calls = []
    hook = draft.fc.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[1])
    )
    memory = ReTraceMemory(
        torch.randn(1, 4, 32),
        torch.randn(1, 4, 32),
        torch.tensor([[True, True, False, False]]),
    )
    with torch.no_grad():
        for length in (3, 7, 11, 11, 15):
            before = len(calls)
            actual = cache.propose(context[:, :length], torch.tensor([2]), memory, 0.4)
            cache_calls = calls[before:]
            expected = draft(context[:, :length], torch.tensor([2]), memory, 0.4)
            torch.testing.assert_close(
                actual.logits, expected.logits, atol=2e-6, rtol=1e-5
            )
            assert torch.equal(actual.token_ids, expected.token_ids)
            assert sum(cache_calls) <= length
    hook.remove()
    assert cache.projected_context_tokens == 15
    assert calls == [3, 3, 4, 7, 4, 11, 11, 4, 15]
    with torch.no_grad():
        with pytest.raises(ValueError, match="monotonically"):
            cache.propose(context[:, :3], torch.tensor([2]), memory, 1)
        draft.fc.weight.add_(0.01)
        with pytest.raises(RuntimeError, match="parameter update"):
            cache.propose(context, torch.tensor([2]), memory, 1)


def test_cache_never_runs_in_a_gradient_forward(tmp_path):
    _, draft = models(tmp_path)
    cache = DraftContextCache(draft)
    with pytest.raises(RuntimeError, match="no-grad"):
        cache.propose(torch.randn(1, 3, 64), torch.tensor([2]), None, 1)


def test_complete_rollout_memory_labels_and_gradients_match(tmp_path):
    target, draft = models(tmp_path)
    teacher = LocalTarget(target, draft.target_layer_ids)
    first = collect_trajectory(teacher, draft, [1, 2, 3], 28, set(), 1)
    second = collect_trajectory(
        teacher,
        draft,
        [1, 2, 3],
        28,
        set(),
        1,
        performance_mode="cached",
        trace_storage="device",
    )
    assert first.tokens == second.tokens
    assert first.positions == second.positions
    assert first.accepted == second.accepted
    assert first.target_calls == second.target_calls == len(first.positions) + 1
    torch.testing.assert_close(first.context, second.context)
    for a, b in zip(first.memories, second.memories, strict=True):
        assert not b.draft.requires_grad
        assert b.draft.dtype == b.target.dtype == torch.float16
        assert torch.equal(a.valid, b.valid)
        torch.testing.assert_close(a.draft, b.draft, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(a.target, b.target)
    # Use identical recorded states: isolate replay/chunk changes from BF16 cache rounding.
    optimized = copy.deepcopy(draft)
    loss_a = backward_trajectory(draft.train(), first, 1, blocks_per_forward=2)
    loss_b = backward_trajectory(
        optimized.train(), first, 1, blocks_per_forward=9, optimized=True
    )
    assert loss_a["labels"] == loss_b["labels"]
    assert loss_a["conditioned"] == loss_b["conditioned"]
    assert abs(loss_a["loss"] - loss_b["loss"]) < 2e-6
    for (name, a), (_, b) in zip(
        draft.named_parameters(), optimized.named_parameters(), strict=True
    ):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4, msg=name)


def test_resume_only_allows_explicit_execution_changes():
    old = {"epochs": 8, "lr": 5e-5, "blocks_per_forward": 4, "pool_sha256": "abc"}
    assert validate_resume_signature(old, old) == {}
    new = old | {
        "performance_mode": "cached",
        "trace_storage": "device",
        "blocks_per_forward": 16,
    }
    with pytest.raises(ValueError, match="allow-performance-change"):
        validate_resume_signature(old, new)
    assert len(validate_resume_signature(old, new, True)) == 3
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(old, new | {"lr": 1e-4}, True)


def test_cached_cpu_resume_keeps_optimizer_schedule_and_saves_epoch(
    tmp_path, monkeypatch
):
    # Run the real training entry point, then switch execution settings at a saved update.
    import hashlib
    import sys

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from speculators.models.retrace import train_pretrained
    from speculators.models.retrace.pretrained import prepare

    _, source, _ = fixture(tmp_path)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"t{i}": i for i in range(48)}, unk_token="t0")
        ),
        unk_token="t0",
    )
    tok.save_pretrained(tmp_path / "target")
    prepare(source, str(tmp_path / "target"), tmp_path / "initial")
    pool = tmp_path / "prompts.jsonl"
    rows = []
    for ids in ([1, 2, 3], [2, 3, 4]):
        key = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        rows.append({"input_ids": ids, "sha256": key})
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    common = [
        "trainer",
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
        "1",
        "--response-length",
        "10",
        "--profile-performance",
    ]
    monkeypatch.setattr(
        sys, "argv", common + ["--output", str(tmp_path / "old"), "--stop-after", "1"]
    )
    train_pretrained.main()
    checkpoint = tmp_path / "old/step_000001"
    # Strip added defaults to emulate an actual checkpoint from the pre-patch trainer.
    path = checkpoint / "training_state.json"
    state = json.loads(path.read_text())
    for name in ("performance_mode", "trace_storage"):
        state["signature"].pop(name)
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
            "--performance-mode",
            "cached",
            "--trace-storage",
            "device",
            "--blocks-per-forward",
            "8",
            "--allow-performance-change",
        ],
    )
    train_pretrained.main()
    metrics = [
        json.loads(s)
        for s in (tmp_path / "fast/metrics.jsonl").read_text().splitlines()
    ]
    assert [x["step"] for x in metrics] == [2, 3, 4]
    assert all(x["step_seconds"] > 0 for x in metrics)
    assert [x["lr"] for x in metrics] == [
        train_pretrained.learning_rate(s, 4, 5e-5, 0.05) for s in (1, 2, 3)
    ]
    assert (tmp_path / "fast/step_000002/master_parameters.pt").is_file()
    assert (tmp_path / "fast/step_000004/optimizer.pt").is_file()
    change = json.loads((tmp_path / "fast/resume_command.json").read_text())[
        "resume_execution_changes"
    ]
    assert set(change) == {"performance_mode", "trace_storage", "blocks_per_forward"}


def test_cached_bfloat16_projections_remain_close(tmp_path):
    target, draft = models(tmp_path)
    context = (
        LocalTarget(target, draft.target_layer_ids).states(list(range(1, 17))).auxiliary
    )
    draft.to(dtype=torch.bfloat16)
    for p in draft.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    context = context.bfloat16()
    cache = DraftContextCache(draft)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        for length in (3, 7, 13):
            expected = draft(context[:, :length], torch.tensor([2]), None, 1).logits
            actual = cache.propose(
                context[:, :length], torch.tensor([2]), None, 1
            ).logits
            torch.testing.assert_close(expected, actual, atol=0.01, rtol=0.03)


def test_benchmark_command_preserves_training_recipe_and_measures_post_warmup():
    from speculators.models.retrace.performance_benchmark import make_command, summarize

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
    )
    command = make_command(
        signature,
        "/checkpoint",
        "/checkpoint",
        "/pool",
        "/output",
        "cached",
        16,
        1256,
        True,
    )
    assert command[command.index("--nproc_per_node") + 1] == "8"
    assert command[command.index("--resume") + 1] == "/checkpoint"
    assert command[command.index("--response-length") + 1] == "1024"
    assert command[command.index("--max-steps") + 1] == "10000"
    assert command[command.index("--stop-after") + 1] == "1256"
    assert "--allow-performance-change" in command
    rows = [
        dict(
            step=1251 + i,
            step_seconds=t,
            rollout_seconds_max=t * 0.6,
            backward_seconds_max=t * 0.3,
            optimizer_seconds_max=t * 0.1,
            peak_memory_gib_max=30,
            rounds=100,
            target_calls=132,
            global_prompts=32,
        )
        for i, t in enumerate((100, 70, 60, 50))
    ]
    report = summarize(rows, 2)
    assert report["mean_step_seconds"] == 55
    assert report["steps"] == [1253, 1254]
    assert report["global_prompts_per_step"] == [32, 32]
