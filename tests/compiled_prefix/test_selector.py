"""CPU tests of the opt-in selector; no NPU throughput claims."""

import unittest
from unittest.mock import patch

import torch

from speculators.models.dflash_prefix.compiled_selector import CompiledPrefixSelector
from speculators.models.dflash_prefix.model_definitions import PrefixAttentionSelector
from speculators.models.dflash_prefix.selector_v2 import (
    LocalPrefixSelector,
    make_prefix_head,
)


class CompiledSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, proposals=4, candidates=3, rank=8, heads=2):
        torch.manual_seed(100)
        model = CompiledPrefixSelector(32, 16, rank, candidates, proposals, heads)
        with torch.no_grad():
            model.head_weights.fill_(1 / heads)
            model.successor_projection.weight.normal_(std=0.1)
            model.distance_bias.normal_(std=0.05)
        embedding = torch.randn(32, 16, requires_grad=True)
        hidden = torch.randn(2, proposals, 16)
        ids = torch.stack(
            [torch.randperm(32)[:candidates] for _ in range(2 * proposals)]
        ).reshape(2, proposals, candidates)
        unary = torch.randn(2, proposals, candidates)
        anchor = torch.randint(32, (2,))
        forced = torch.randint(candidates, (2, proposals))
        prefix = ids.gather(-1, forced[..., None]).squeeze(-1)
        return model, (hidden, ids, unary, anchor), embedding, forced, prefix

    def test_teacher_compiled_scores_and_greedy_match(self):
        for p, k, heads in ((1, 3, 2), (7, 16, 4), (15, 16, 4)):
            with self.subTest(proposals=p):
                model, inputs, embedding, forced, prefix = self.fixture(p, k, 16, heads)
                prepared = model.prepare_tables(*inputs, embedding_weight=embedding)
                expected = model.score_teacher_prefix(
                    *inputs, prefix, embedding_weight=embedding
                )
                _, actual = model.walk(
                    prepared, forced_indices=forced, return_scores=True
                )
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                sequential_prefix = torch.zeros_like(prefix)
                for i in range(p):
                    row = model.score_teacher_prefix(
                        *inputs, sequential_prefix, embedding_weight=embedding
                    )[:, i]
                    sequential_prefix[:, i] = inputs[1][:, i].gather(
                        -1, row.argmax(-1)[:, None]
                    ).squeeze(-1)
                self.assertTrue(
                    torch.equal(model.greedy_walk(prepared), sequential_prefix)
                )

    def test_causality_older_dependency_and_anchor_eligibility(self):
        model, inputs, embedding, _, prefix = self.fixture()
        expected = model.score_teacher_prefix(
            *inputs, prefix, embedding_weight=embedding
        )
        changed = prefix.clone()
        changed[:, 2:] = (changed[:, 2:] + 1) % 32
        actual = model.score_teacher_prefix(
            *inputs, changed, embedding_weight=embedding
        )
        torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=0, rtol=0)
        changed = prefix.clone()
        changed[:, 0] = (changed[:, 0] + 1) % 32
        actual = model.score_teacher_prefix(
            *inputs, changed, embedding_weight=embedding
        )
        self.assertTrue(torch.equal(prefix[:, 1], changed[:, 1]))
        self.assertGreater((actual[:, 2] - expected[:, 2]).abs().max().item(), 1e-6)
        model.history_scale = 0
        local = model.score_teacher_prefix(*inputs, prefix, embedding_weight=embedding)
        torch.testing.assert_close(local[:, 0], expected[:, 0], atol=0, rtol=0)
        self.assertGreater((local[:, 1] - expected[:, 1]).abs().max().item(), 1e-6)

    def test_backward_all_parameters_finite_and_target_embedding_frozen(self):
        model, inputs, embedding, _, prefix = self.fixture()
        model.score_teacher_prefix(
            *inputs, prefix, embedding_weight=embedding
        ).square().mean().backward()
        self.assertIsNone(embedding.grad)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_bfloat16_autocast_keeps_statistics_fp32(self):
        model, inputs, embedding, forced, prefix = self.fixture()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = model.score_teacher_prefix(
                *inputs, prefix, embedding_weight=embedding
            )
            prepared = model.prepare_tables(*inputs, embedding_weight=embedding)
            _, actual = model.walk(prepared, forced_indices=forced, return_scores=True)
            expected.square().mean().backward()
        self.assertEqual(actual.dtype, torch.float32)
        for value in prepared[1:8]:
            self.assertEqual(value.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_walk_has_no_projections_or_exponentials(self):
        model, inputs, embedding, forced, _ = self.fixture()
        prepared = model.prepare_tables(*inputs, embedding_weight=embedding)
        with (
            patch.object(
                torch.nn.Linear,
                "forward",
                side_effect=AssertionError("projection during walk"),
            ),
            patch.object(
                torch.Tensor, "exp", side_effect=AssertionError("exp during walk")
            ),
        ):
            model.walk(prepared, forced_indices=forced, return_scores=True)

    def test_initialize_copies_and_freezes_only_local_teacher_parameters(self):
        model, inputs, embedding, _, prefix = self.fixture()
        teacher = LocalPrefixSelector(32, 16, 8, 3, 4, 2, 2, history_scale=0)
        with torch.no_grad():
            teacher.successor_projection.weight.normal_(std=0.1)
        model.initialize_from_teacher(teacher)
        model.history_scale = 0
        actual = model.score_teacher_prefix(
            *inputs, prefix, embedding_weight=embedding
        )
        expected = teacher.score_teacher_prefix(
            *inputs, prefix, embedding_weight=embedding
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name in (
            "predecessor_projection", "successor_projection", "hidden_projection"
        ):
            self.assertFalse(getattr(model, name).weight.requires_grad)
        self.assertTrue(model.candidate_projection.weight.requires_grad)
        self.assertTrue(model.memory_projection.weight.requires_grad)
        self.assertFalse(hasattr(model, "prefix_layers"))
        with self.assertRaisesRegex(ValueError, "rank differs"):
            model.initialize_from_teacher(LocalPrefixSelector(32, 16, 16, 3, 4, 2, 2))

    def test_old_factory_modes_keep_identical_state_and_defaults(self):
        torch.manual_seed(123)
        expected = LocalPrefixSelector(32, 16, 8, 3, 4, 2, 2)
        torch.manual_seed(123)
        actual = make_prefix_head(
            32, 16, 8, 3, 4, kind="local_prefix_v2", heads=2, layers=2
        )
        self.assertEqual(expected.state_dict().keys(), actual.state_dict().keys())
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, atol=0, rtol=0)
        self.assertIsInstance(
            make_prefix_head(32, 16, 8, 3, 4), PrefixAttentionSelector
        )
        self.assertIsInstance(
            make_prefix_head(32, 16, 8, 3, 4, kind="compiled_prefix_v3", heads=2),
            CompiledPrefixSelector,
        )


if __name__ == "__main__":
    unittest.main()
