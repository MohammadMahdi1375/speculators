"""CPU integration tests; run unittest discovery without repository conftest."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash_prefix.compiled_training import (
    CompiledPrefixPilot, build_compiled_pilot, older_history_intervention,
    pilot_loss_terms,
)
from speculators.models.dflash_prefix.config import DFlashPrefixSpeculatorConfig
from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.checkpointer import SingleGPUCheckpointer


def make_checkpoint(root):
    torch.manual_seed(401)
    target = root / "relocated_target"
    target.mkdir()
    target_config = Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        architectures=["Qwen3ForCausalLM"], _attn_implementation="eager",
    )
    target_config.save_pretrained(target)
    embedding, head = torch.randn(16, 16) * .2, torch.randn(16, 16) * .2
    save_file({"model.embed_tokens.weight": embedding, "lm_head.weight": head,
               "model.norm.weight": torch.ones(16)}, str(target / "model.safetensors"))
    layer = Qwen3Config(**{**target_config.to_dict(), "num_hidden_layers": 1,
                          "layer_types": ["full_attention"],
                          "_attn_implementation": "eager"})
    config = DFlashPrefixSpeculatorConfig(
        transformer_layer_config=layer, draft_vocab_size=16, block_size=4,
        aux_hidden_state_layer_ids=[0, 1], mask_token_id=0, sample_from_anchor=False,
        prefix_selector_kind="local_prefix_v2", prefix_rank=8, prefix_top_k=16,
        prefix_attention_heads=2, prefix_attention_layers=2,
        prefix_loss_kind="target_ce", prefix_loss_scope="reachable_greedy_prefix",
        speculators_config=SpeculatorsConfig(
            algorithm="dflash_prefix",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=["Qwen3ForCausalLM"]),
        ),
    )
    source = DFlashPrefixDraftModel(config)
    with torch.no_grad():
        source.embed_tokens.weight.copy_(embedding)
        source.lm_head.weight.copy_(head)
        source.verifier_lm_head.weight.copy_(head)
        source.verifier_norm.weight.fill_(1)
        source.prefix_head.successor_projection.weight.normal_(std=.1)
        source.prefix_head.prefix_output.weight.normal_(std=.1)
    # The embedded path genuinely does not exist on this machine. Loading the
    # pilot must use the supplied new target before reconstruction of weights.
    source.config.speculators_config.verifier.name_or_path = str(root / "old_server_missing_target")
    checkpoint = root / "teacher"
    source.save_pretrained(checkpoint)
    return source, checkpoint, target


def tiny_batch(model, *, valid=True):
    torch.manual_seed(402)
    length = 20
    last = torch.randn(1, length, 16)
    with torch.no_grad():
        reference = model.verifier_lm_head(model.verifier_norm(last)).argmax(-1)
    tokens = torch.randint(0, 16, (1, length))
    tokens[:, 1:] = reference[:, :-1]
    return dict(hidden_states=torch.randn(1, length, 32), input_ids=tokens,
                loss_mask=torch.ones(1, length) if valid else torch.zeros(1, length),
                verifier_last_hidden_states=last,
                document_ids=torch.zeros(1, length, dtype=torch.long), max_anchors=3)


class CompiledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_checkpoint_relocation_frozen_gradients_and_serving_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint, target = make_checkpoint(Path(directory))
            model = build_compiled_pilot(checkpoint, target, {"intervention_fraction": 1.0})
            self.assertIsInstance(model, CompiledPrefixPilot)
            SpeculatorModel.verify_training_compatible(model)
            self.assertEqual(model.config.speculators_config.verifier.name_or_path, str(target))
            self.assertEqual(model.config.prefix_selector_kind, "compiled_prefix_v3")
            for name, value in source.prefix_head.state_dict().items():
                torch.testing.assert_close(model.teacher_head.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(model.fc.weight, source.fc.weight, rtol=0, atol=0)
            model.train()
            self.assertFalse(model.teacher_head.training)
            with torch.no_grad():
                model.prefix_head.head_weights.fill_(.2)
            _, loss, metrics = model(**tiny_batch(model))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(metrics["target_ce_total"].item(), 0)
            self.assertGreater(metrics["history_delta_loss_total"].item(), 0)
            live = {name: p for name, p in model.named_parameters() if p.requires_grad}
            self.assertTrue(live)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertTrue(name.startswith("prefix_head."), name)
                    self.assertEqual(parameter.dtype, torch.float32, name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            self.assertGreater(model.prefix_head.history_value.weight.grad.abs().sum().item(), 0)
            self.assertNotIn("prefix_head.predecessor_projection.weight", live)
            exported = Path(directory) / "student"
            model.save_pretrained(exported, state_dict=model.state_dict())
            saved = load_file(str(exported / "model.safetensors"))
            raw = json.loads((exported / "config.json").read_text())
            self.assertFalse(any(name.startswith("teacher_head.") for name in saved))
            self.assertEqual(raw["architectures"], ["DFlashPrefixDraftModel"])
            self.assertEqual(raw["prefix_attention_layers"], 1)
            restored = DFlashPrefixDraftModel.from_pretrained(exported, local_files_only=True)
            self.assertFalse(hasattr(restored, "teacher_head"))
            self.assertEqual(restored.config.prefix_selector_kind, "compiled_prefix_v3")
            for name, value in saved.items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            model.eval()
            _, _, metrics = model(**tiny_batch(model))
            for name in ("student_reference_eal", "teacher_reference_eal", "direct_prefix_reference_proxy"):
                self.assertGreater(metrics[name + "_total"].item(), 0)

    def test_no_valid_tokens_gives_connected_zero_for_all_trainable_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            _, checkpoint, target = make_checkpoint(Path(directory))
            model = build_compiled_pilot(checkpoint, target, {"history_weight": 0})
            _, loss, metrics = model(**tiny_batch(model, valid=False))
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(metrics["target_ce_total"].item(), 0)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_trainer_bfloat16_supplied_state_exports_standard_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            _, checkpoint, target = make_checkpoint(Path(directory))
            model = build_compiled_pilot(checkpoint, target)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
            checkpointer = SingleGPUCheckpointer(Path(directory) / "checkpoints")
            checkpointer.save_checkpoint(model, optimizer, 0, float_dtype=torch.bfloat16)
            exported = Path(directory) / "checkpoints" / "0"
            saved = load_file(str(exported / "model.safetensors"))
            raw = json.loads((exported / "config.json").read_text())
            self.assertEqual(raw["architectures"], ["DFlashPrefixDraftModel"])
            self.assertTrue(all(value.dtype == torch.bfloat16 for value in saved.values()))
            self.assertFalse(any(name.startswith("teacher_head.") for name in saved))
            restored = DFlashPrefixDraftModel.from_pretrained(exported, local_files_only=True, dtype="auto")
            self.assertEqual(restored.prefix_head.head_weights.dtype, torch.bfloat16)
            self.assertFalse(hasattr(restored, "teacher_head"))
            for name, value in saved.items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            # Checkpoint dtype conversion must not round the training master weights.
            self.assertEqual(model.prefix_head.head_weights.dtype, torch.float32)

    def test_bfloat16_teacher_with_float32_history_trains_under_autocast(self):
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint, target = make_checkpoint(Path(directory))
            source.to(torch.bfloat16).save_pretrained(checkpoint)
            model = build_compiled_pilot(checkpoint, target, {"intervention_fraction": 1.0})
            self.assertEqual(model.fc.weight.dtype, torch.bfloat16)
            self.assertEqual(model.teacher_head.prefix_output.weight.dtype, torch.bfloat16)
            self.assertEqual(model.prefix_head.history_value.weight.dtype, torch.float32)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                batch = tiny_batch(model)
                _, loss, _ = model(**batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_incomplete_teacher_and_incompatible_target_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, checkpoint, target = make_checkpoint(Path(directory))
            state = load_file(str(checkpoint / "model.safetensors"))
            state.pop("fc.weight")
            save_file(state, str(checkpoint / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Incomplete/incompatible"):
                build_compiled_pilot(checkpoint, target)
            config_path = target / "config.json"
            raw = json.loads(config_path.read_text())
            raw["vocab_size"] = 17
            config_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "vocab_size"):
                build_compiled_pilot(checkpoint, target)

    def test_interventions_preserve_anchor_and_immediate_predecessor(self):
        reference = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        candidates = torch.stack((reference, reference + 6), -1)
        valid = torch.ones_like(reference, dtype=torch.bool)
        edited, mask = older_history_intervention(reference, candidates, valid, 1.0)
        for row in range(2):
            changes = torch.where(edited[row] != reference[row])[0]
            self.assertEqual(changes.numel(), 1)
            j = changes.item()
            self.assertGreater(mask[row].sum().item(), 0)
            self.assertTrue(candidates[row, j].eq(edited[row, j]).any())
            for i in torch.where(mask[row])[0].tolist():
                self.assertGreaterEqual(i - j, 2)
                self.assertEqual(edited[row, i-1].item(), reference[row, i-1].item())
        two = reference[:, :2]
        unchanged, mask = older_history_intervention(two, candidates[:, :2], valid[:, :2], 1.0)
        torch.testing.assert_close(unchanged, two)
        self.assertFalse(mask.any())

    def test_loss_masks_target_coverage_and_detaches_teacher(self):
        student = torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]], requires_grad=True)
        teacher = torch.zeros_like(student, requires_grad=True)
        edited_student = student + torch.tensor([[[.1, .0], [.0, .1], [.3, -.1]]])
        edited_teacher = teacher + .2
        ids = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        terms = pilot_loss_terms(
            student, teacher, ids, torch.tensor([[1, 99, 5]]), valid,
            torch.tensor([[True, True, False]]), student_edited=edited_student,
            teacher_edited=edited_teacher, older_intervention_mask=torch.tensor([[False, False, True]]),
        )
        self.assertEqual(terms["target_ce"][1].item(), 1)
        self.assertEqual(terms["history_delta_loss"][1].item(), 1)
        sum(total / count.clamp_min(1) for total, count in terms.values()).backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        shifted = pilot_loss_terms(
            student + 2, teacher + 5, ids, torch.tensor([[1, 99, 5]]), valid,
            torch.tensor([[True, True, False]]), student_edited=edited_student + 3,
            teacher_edited=edited_teacher + 7, older_intervention_mask=torch.tensor([[False, False, True]]),
        )
        for name in terms:
            torch.testing.assert_close(terms[name][0], shifted[name][0], atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
