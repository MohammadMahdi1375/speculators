"""Protect pretrained tensors, clean targets, causal blocks and epoch accounting."""

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.retrace.clean_training import (
    backward_trajectory,
    block_ce_sum,
    clean_labels,
    collect_trajectory,
)
from speculators.models.retrace.config import ReTraceSpeculatorConfig
from speculators.models.retrace.core import ReTraceDraftModel
from speculators.models.retrace.memory import ReTraceMemory
from speculators.models.retrace.pretrained import SHARED, load_dflash, prepare
from speculators.models.retrace.prompt_pool import disable_thinking, extract_prompt
from speculators.models.retrace.target import LocalTarget
from speculators.models.retrace.train_pretrained import epoch_batches
from speculators.proposals.greedy import GreedyTokenProposalConfig


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


def test_import_preserves_all_pretrained_weights_and_shifts_layers(tmp_path):
    target, source, body = fixture(tmp_path)
    draft, report = load_dflash(source, str(tmp_path / "target"), dtype=torch.float32)
    assert draft.target_layer_ids == [1, 3]
    assert report["imported_tensors"] == len(body)
    assert draft.config.training_objective == "clean_block_ce"
    for name, value in body.items():
        assert torch.equal(value, draft.state_dict()[name]), name
    assert torch.count_nonzero(draft.retrace.value.weight) == 0
    assert not draft.embed_tokens.weight.requires_grad
    assert not draft.lm_head.weight.requires_grad
    assert torch.equal(draft.embed_tokens.weight, target.model.embed_tokens.weight)
    prepare(source, str(tmp_path / "target"), tmp_path / "initial")
    loaded = ReTraceDraftModel.from_pretrained(tmp_path / "initial/retrace")
    assert loaded.target_layer_ids == [1, 3]
    baseline = json.loads((tmp_path / "initial/dflash/config.json").read_text())
    assert baseline["speculators_model_type"] == "dflash"


def test_missing_backbone_weight_is_not_silently_initialized(tmp_path):
    _, source, body = fixture(tmp_path)
    body.pop("fc.weight")
    save_file(body, str(source / "model.safetensors"))
    with pytest.raises(ValueError, match="Feature fusion"):
        load_dflash(source, str(tmp_path / "target"), dtype=torch.float32)


def test_chunked_blocks_match_independent_forwards_and_hide_future_context(tmp_path):
    target, source, _ = fixture(tmp_path)
    draft, _ = load_dflash(source, str(tmp_path / "target"), dtype=torch.float32)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    context = LocalTarget(target, draft.target_layer_ids).states(ids).auxiliary
    positions = torch.tensor([3, 5])
    anchors = torch.tensor([4, 6])
    m = torch.randn(2, 4, 32)
    memory = ReTraceMemory(m, m * 0.5, torch.ones(2, 4, dtype=torch.bool))
    with torch.no_grad():
        draft.retrace.value.weight.normal_(std=0.02)
    batched = draft.training_blocks(context, anchors, positions, memory).logits
    for i, position in enumerate(positions):
        mem = ReTraceMemory(
            memory.draft[i : i + 1], memory.target[i : i + 1], memory.valid[i : i + 1]
        )
        expected = draft(context[:, :position], anchors[i : i + 1], mem).logits
        torch.testing.assert_close(batched[i : i + 1], expected, atol=1e-6, rtol=1e-5)
    changed = context.clone()
    changed[:, 5:] += 100
    altered = draft.training_blocks(changed, anchors, positions, memory).logits
    torch.testing.assert_close(batched, altered, atol=1e-6, rtol=1e-5)


def test_clean_continuation_labels_and_original_ce_reduction():
    labels, valid = clean_labels([4, 5, 6, 7], torch.tensor([1, 2]), 3, "cpu")
    assert labels.tolist() == [[6, 7, -100], [7, -100, -100]]
    logits = torch.zeros(2, 3, 8, requires_grad=True)
    numerator = block_ce_sum(logits, labels)
    expected = torch.log(torch.tensor(8.0)) * (2 + torch.exp(torch.tensor(-0.25)))
    torch.testing.assert_close(numerator, expected)
    (numerator / valid.sum()).backward()
    assert torch.count_nonzero(logits.grad[~valid]) == 0


def test_online_collection_detaches_memory_and_all_modules_learn(tmp_path):
    target, source, _ = fixture(tmp_path)
    draft, _ = load_dflash(source, str(tmp_path / "target"), dtype=torch.float32)
    target_before = {name: value.clone() for name, value in target.state_dict().items()}
    before = {name: value.clone() for name, value in draft.retrace.state_dict().items()}
    teacher = LocalTarget(target, draft.target_layer_ids)
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad], lr=1e-3
    )
    for step in range(3):
        trace = collect_trajectory(
            teacher, draft.eval(), [1, 2, 3], 12, set(), min(step / 2, 1)
        )
        assert trace.target_calls == 1 + len(trace.positions)
        assert all(
            m.draft.dtype == m.target.dtype == torch.float16 for m in trace.memories
        )
        assert all(
            not m.draft.requires_grad and not m.target.requires_grad
            for m in trace.memories
        )
        optimizer.zero_grad()
        loss = backward_trajectory(
            draft.train(), trace, min(step / 2, 1), blocks_per_forward=3
        )
        assert loss["labels"] > 0
        optimizer.step()
    for name, value in before.items():
        assert not torch.equal(value, draft.retrace.state_dict()[name]), name
    for name, value in target_before.items():
        assert torch.equal(value, target.state_dict()[name]), name


def test_exact_prompt_batch_and_eight_epochs_including_partial_batch():
    batches = list(epoch_batches(35, 8, 32, 42))
    assert len(batches) == 16
    for epoch in range(8):
        assert sorted(i for e, batch in batches if e == epoch for i in batch) == list(
            range(35)
        )
    assert len(list(epoch_batches(40000, 8, 32, 42))) == 10000


def test_zero_residual_preserves_pretrained_predictions_with_memory(tmp_path):
    target, source, _ = fixture(tmp_path)
    draft, _ = load_dflash(source, str(tmp_path / "target"), dtype=torch.float32)
    context = LocalTarget(target, draft.target_layer_ids).states([1, 2, 3]).auxiliary
    memory = ReTraceMemory(
        torch.randn(1, 4, 32), torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool)
    )
    expected = draft(context, torch.tensor([4]), None).logits
    actual = draft(context, torch.tensor([4]), memory, 1.0).logits
    assert torch.equal(expected, actual)


def test_block_chunk_size_preserves_gradients(tmp_path):
    target, source, _ = fixture(tmp_path)
    draft, _ = load_dflash(source, str(tmp_path / "target"), dtype=torch.float32)
    with torch.no_grad():
        draft.retrace.value.weight.normal_(std=0.02)
    trace = collect_trajectory(
        LocalTarget(target, draft.target_layer_ids), draft, [1, 2, 3], 12, set(), 1.0
    )
    copy_draft = copy.deepcopy(draft)
    backward_trajectory(draft, trace, 1.0, blocks_per_forward=1)
    backward_trajectory(copy_draft, trace, 1.0, blocks_per_forward=4)
    for (name, a), (_, b) in zip(
        draft.named_parameters(), copy_draft.named_parameters(), strict=True
    ):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_prompt_mask_accepts_unscored_eos_and_never_copies_response():
    assert extract_prompt(
        {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [False, False, True, False, False],
            "seq_len": 5,
        },
        6,
    ) == [1, 2]

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [10, 11] if text.startswith("<|im_start|>") else [12, 13, 14, 15]

    tokenizer = Tokenizer()
    assert disable_thinking([1, 10, 11], tokenizer) == [1, 10, 11, 12, 13, 14, 15]
    assert disable_thinking([1, 10, 11, 12, 13, 14, 15], tokenizer) == [
        1,
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    with pytest.raises(ValueError, match="header"):
        disable_thinking([1, 2, 3], tokenizer)


def test_cpu_training_resume_preserves_fp32_master_and_schedule(tmp_path):
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
    with pool.open("w") as handle:
        for ids in ([1, 2, 3], [2, 3, 4], [3, 4, 5]):
            key = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            handle.write(json.dumps({"input_ids": ids, "sha256": key}) + "\n")
    command = [
        sys.executable,
        "-m",
        "speculators.models.retrace.train_pretrained",
        "--target",
        str(tmp_path / "target"),
        "--draft",
        str(tmp_path / "initial/retrace"),
        "--pool",
        str(pool),
        "--device",
        "cpu",
        "--prompt-length",
        "16",
        "--response-length",
        "6",
        "--global-prompt-batch",
        "2",
        "--epochs",
        "2",
        "--save-every",
        "0",
        "--blocks-per-forward",
        "2",
    ]

    def run(output, extra=()):
        result = subprocess.run(
            [*command, "--output", str(tmp_path / output), *extra],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run("continuous")
    # Three prompts and batch two include an incomplete last batch each epoch.
    assert sorted(p.name for p in (tmp_path / "continuous").glob("step_*")) == [
        "step_000002",
        "step_000004",
    ]
    # A pause halfway through epoch two must save without resetting epoch timing.
    run("resumed", ["--stop-after", "3"])
    run("resumed", ["--resume", str(tmp_path / "resumed/step_000003")])
    assert sorted(p.name for p in (tmp_path / "resumed").glob("step_*")) == [
        "step_000002",
        "step_000003",
        "step_000004",
    ]
    a = torch.load(
        tmp_path / "continuous/step_000004/master_parameters.pt", weights_only=True
    )
    b = torch.load(
        tmp_path / "resumed/step_000004/master_parameters.pt", weights_only=True
    )
    for name in a:
        assert torch.equal(a[name], b[name]), name
    metrics_a = [
        json.loads(line)
        for line in (tmp_path / "continuous/metrics.jsonl").read_text().splitlines()
    ]
    metrics_b = [
        json.loads(line)
        for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["lr"] for row in metrics_a] == [row["lr"] for row in metrics_b]
    assert len(metrics_b) == 4
    # Completed resume is a read-only no-op, not an attempt to overwrite weights.
    run("resumed", ["--resume", str(tmp_path / "resumed/step_000004")])
    # Positive step intervals remain available as an explicit override.
    run("step_interval", ["--save-every", "1", "--stop-after", "2"])
    assert sorted(p.name for p in (tmp_path / "step_interval").glob("step_*")) == [
        "step_000001",
        "step_000002",
    ]
