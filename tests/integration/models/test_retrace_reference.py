"""CPU checks with real tiny Qwen3 and isolated ReTrace models, no downloaded weights."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators import ReTraceDraftModel, ReTraceSpeculatorConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.retrace.rollout import (
    generate,
    rollout,
    soft_ce_loss,
    target_greedy,
)
from speculators.models.retrace.target import LocalTarget
from speculators.proposals.greedy import GreedyTokenProposalConfig


def models():
    torch.manual_seed(4)
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        # Qwen3-4B also has attention width larger than hidden_size.
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 3,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    draft_config = Qwen3Config(**config.to_dict())
    draft_config.num_hidden_layers = 2
    draft_config.layer_types = ["full_attention"] * 2
    draft_config._attn_implementation = "eager"
    draft = ReTraceDraftModel(
        ReTraceSpeculatorConfig(
            transformer_layer_config=draft_config,
            aux_hidden_state_layer_ids=[1, 2],
            draft_vocab_size=48,
            block_size=5,
            mask_token_id=47,
            retrace_enabled=True,
            speculators_config=SpeculatorsConfig(
                algorithm="retrace",
                default_proposal_method="greedy",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=4)],
                verifier=VerifierConfig.from_config(config, name_or_path=None),
            ),
        )
    )
    draft.embed_tokens = target.model.embed_tokens
    draft.lm_head = target.lm_head
    draft.verifier_lm_head = target.lm_head
    draft.verifier_norm = target.model.norm
    return target, draft


def test_conditioned_speculation_matches_target_tokens_across_rounds():
    model, draft = models()
    target = LocalTarget(model, draft.target_layer_ids)
    with torch.no_grad():
        draft.retrace.value.weight.normal_(std=0.03)
    expected = target_greedy(target, draft, [1, 2, 3, 4], 19, set())
    actual, rounds = generate(target, draft.eval(), [1, 2, 3, 4], 19, set())
    assert actual == expected
    assert len(rounds) > 2
    assert sum(x["conditioned"] for x in rounds) > 0


def test_all_fusion_matrices_learn_and_entire_target_stays_frozen():
    model, draft = models()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    old_fusion = {k: v.clone() for k, v in draft.retrace.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad], lr=1e-3
    )
    target = LocalTarget(model, draft.target_layer_ids)
    for event in rollout(target, draft, [1, 2, 3], 12, set(), train=True):
        loss, _ = soft_ce_loss(event)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for name, value in old_fusion.items():
        assert not torch.equal(value, draft.retrace.state_dict()[name]), name
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name]), name
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("budget", [0, 1, 2, 9])
def test_budget_and_eos_match_target(budget):
    model, draft = models()
    target = LocalTarget(model, draft.target_layer_ids)
    first = target_greedy(target, draft, [1, 2], 1, set())[0]
    assert generate(target, draft, [1, 2], budget, {first})[0] == target_greedy(
        target, draft, [1, 2], budget, {first}
    )


def test_serialized_conditioner_reload_preserves_output(tmp_path):
    model, draft = models()
    with torch.no_grad():
        draft.retrace.value.weight.normal_(std=0.02)
    draft.save_pretrained(tmp_path)
    loaded = ReTraceDraftModel.from_pretrained(tmp_path)
    loaded.embed_tokens = model.model.embed_tokens
    loaded.lm_head = model.lm_head
    loaded.verifier_lm_head = model.lm_head
    loaded.verifier_norm = model.model.norm
    torch.testing.assert_close(loaded.retrace.value.weight, draft.retrace.value.weight)
    target = LocalTarget(model, draft.target_layer_ids)
    assert loaded.config.speculators_config.algorithm == "retrace"
    assert not any(
        "markov" in name or "confidence_head" in name
        for name, _ in loaded.named_parameters()
    )
    assert (
        generate(target, loaded, [1, 2, 3], 8, set())[0]
        == generate(target, draft, [1, 2, 3], 8, set())[0]
    )


def test_hidden_states_match_unmodified_dflash_backbone():
    model, draft = models()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        output = model.model(ids, output_hidden_states=True)
        context = torch.cat(
            [output.hidden_states[i] for i in draft.target_layer_ids], -1
        )
    anchor = torch.tensor([3])
    documents = torch.zeros_like(ids)
    mask = draft._create_attention_mask(documents, ids.shape[1], anchor, ids.device)
    draft._build_attention_mask = lambda *args: (
        mask,
        None,
        anchor,
        torch.tensor([True]),
    )
    stock_hidden, *_ = draft._backbone_forward(
        context,
        ids,
        torch.ones_like(ids),
        output.last_hidden_state,
        documents,
        max_anchors=1,
    )
    actual = draft.propose(context[:, :3], ids[:, 3])
    torch.testing.assert_close(actual.hidden, stock_hidden[:, 1:], atol=1e-6, rtol=1e-5)


def test_all_accepted_uses_bonus_and_never_reuses_suffix():
    model, draft = models()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    target = LocalTarget(model, draft.target_layer_ids)
    tokens, rounds = generate(target, draft, [1, 2], 14, set())
    assert tokens == [0] * 14
    assert [r["accepted"] for r in rounds] == [4, 4, 3]
    assert all(r["conditioned"] == 0 for r in rounds)


def test_eos_inside_accepted_block_masks_later_training_positions():
    from types import SimpleNamespace

    from speculators.models.retrace.core import DraftBlock
    from speculators.models.retrace.target import TargetStates

    class Target:
        def reset(self):
            pass

        def states(self, tokens):
            scores = torch.full((1, len(tokens), 5), -100.0)
            for position in range(len(tokens)):
                scores[0, position, [1, 2, 3, 4, 0][position % 5]] = 100.0
            return TargetStates(scores, scores)

    class Draft:
        config = SimpleNamespace(retrace_enabled=True)
        lm_head = torch.nn.Identity()

        def __call__(self, *args):
            return DraftBlock(
                torch.zeros(1, 4, 5), torch.zeros(1, 4, 5), torch.tensor([[2, 3, 4, 0]])
            )

    event = next(rollout(Target(), Draft(), [0], 10, {3}))
    assert event.valid.tolist() == [[True, True, False, False]]
    assert event.accepted == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tokenwise_control_keeps_prefill_and_single_token_target_shapes(dtype):
    from speculators.models.retrace.reference import TokenwiseLocalTarget

    model, draft = models()
    model.to(dtype=dtype)
    draft.to(dtype=dtype).eval()
    prompt = [1, 2, 3, 4]
    expected = target_greedy(
        LocalTarget(model, draft.target_layer_ids), draft, prompt, 15, set()
    )
    shapes = []
    hook = model.model.register_forward_pre_hook(
        lambda _, args: shapes.append(args[0].shape[1])
    )
    try:
        actual, rounds = generate(
            TokenwiseLocalTarget(model, draft.target_layer_ids),
            draft,
            prompt,
            15,
            set(),
        )
    finally:
        hook.remove()
    assert expected == actual
    assert shapes[0] == len(prompt)
    assert all(size == 1 for size in shapes[1:])
    assert sum(r["conditioned"] for r in rounds) > 0


def test_tokenwise_control_crops_rejected_paths_and_shorter_requests():
    from speculators.models.retrace.reference import TokenwiseLocalTarget

    model, draft = models()
    target = TokenwiseLocalTarget(model, draft.target_layer_ids)
    paths = [
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 9, 10],
        [1, 2, 3, 4, 9],
        [1, 2, 3, 4, 9, 11],
    ]
    for tokens in paths:
        actual = target.states(tokens)
        reference = LocalTarget(model, draft.target_layer_ids)
        reference.states(tokens[:3])
        for end in range(4, len(tokens) + 1):
            reference.states(tokens[:end])
        assert torch.equal(actual.scoring, reference.last.scoring)
        assert torch.equal(actual.auxiliary, reference.last.auxiliary)
        assert target.cache.get_seq_length() == len(tokens)
        calls = target.calls
        assert target.states(tokens) is actual
        assert target.calls == calls


def test_tokenwise_control_projects_each_verification_row_like_greedy():
    from speculators.models.retrace.reference import TokenwiseLocalTarget

    class ShapeHead(torch.nn.Module):
        def forward(self, x):
            return x + (100 if x.ndim == 3 else 0)

    hidden = torch.randn(1, 8, 5)
    actual = TokenwiseLocalTarget.project(ShapeHead(), hidden)
    assert torch.equal(actual, hidden)
    with pytest.raises(ValueError):
        TokenwiseLocalTarget.project(ShapeHead(), hidden.expand(2, -1, -1))


def test_shape_sensitive_target_failure_remains_visible_in_default_mode():
    from types import SimpleNamespace

    from speculators.models.retrace.core import DraftBlock
    from speculators.models.retrace.reference import TokenwiseLocalTarget

    class Cache:
        length = 0

        def crop(self, length):
            self.length = length

    class Backbone(torch.nn.Module):
        def forward(self, ids, past_key_values=None, **kwargs):
            hidden = torch.zeros(1, ids.shape[1], 5)
            hidden[..., 1] = 1
            if past_key_values is not None and ids.shape[1] > 1:
                hidden[..., 2] = 2  # Inject a shape-dependent target difference.
            cache = past_key_values if past_key_values is not None else Cache()
            cache.length += ids.shape[1]
            return SimpleNamespace(
                hidden_states=(hidden,), last_hidden_state=hidden, past_key_values=cache
            )

    class Model(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.model = Backbone()

    class Draft:
        config = SimpleNamespace(retrace_enabled=True)
        lm_head = torch.nn.Identity()

        def __call__(self, *args):
            return DraftBlock(
                torch.zeros(1, 4, 5), torch.zeros(1, 4, 5), torch.tensor([[3, 3, 3, 3]])
            )

    model, draft = Model(), Draft()
    expected = target_greedy(LocalTarget(model, [0]), draft, [0], 6, set())
    batched, _ = generate(LocalTarget(model, [0]), draft, [0], 6, set())
    control, _ = generate(TokenwiseLocalTarget(model, [0]), draft, [0], 6, set())
    assert expected == control == [1] * 6
    assert batched != expected


def test_evaluation_failure_keeps_expected_tokens_and_fails(tmp_path, monkeypatch):
    import json
    import sys

    from speculators.models.retrace import evaluate

    model, draft = models()
    target = LocalTarget(model, draft.target_layer_ids)

    class Tokenizer:
        def __len__(self):
            return 48

    monkeypatch.setattr(
        evaluate.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(evaluate, "read_prompts", lambda *a: ([[1, 2]], {"rows": [51]}))
    monkeypatch.setattr(evaluate, "setup", lambda *a: (draft, target, set()))
    monkeypatch.setattr(evaluate, "target_greedy", lambda *a: [1, 2])
    monkeypatch.setattr(evaluate, "generate", lambda *a: ([1, 3], [{"accepted": 0}]))
    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--target",
            "fixture",
            "--draft",
            "fixture",
            "--data",
            "fixture",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="Greedy parity failed"):
        evaluate.main()
    report = json.loads(output.read_text())
    assert report["summary"]["parity"] is False
    assert report["summary"]["target_execution"] == "batched"
    assert report["summary"]["native_serving_validated"] is False
    assert report["results"][0]["expected_tokens"] == [1, 2]
    assert report["results"][0]["first_difference"] == {
        "generated_index": 1,
        "expected_token": 2,
        "actual_token": 3,
    }
