#!/usr/bin/env python3
"""Tiny full-model scratch/backward/export check; not a tau or speed benchmark."""
import argparse
import json
import logging
import tempfile
from pathlib import Path


def make_fixture(directory, settings, device="cpu"):
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig
    from speculators.models.dflash_prefix.config import DFlashPrefixSpeculatorConfig
    from speculators.models.dflash_prefix.joint_training import JointCompiledScratch
    torch.manual_seed(901)
    target = Path(directory) / "tiny_target"
    target.mkdir()
    cfg = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        architectures=["Qwen3ForCausalLM"], _attn_implementation="sdpa")
    cfg.save_pretrained(target)
    save_file({"model.embed_tokens.weight": torch.randn(64, 64).bfloat16() * .2,
               "lm_head.weight": torch.randn(64, 64).bfloat16() * .2,
               "model.norm.weight": torch.ones(64).bfloat16()}, str(target / "model.safetensors"))
    layer = Qwen3Config(**{**cfg.to_dict(), "num_hidden_layers": 1, "layer_types": ["full_attention"], "_attn_implementation": "sdpa"})
    config = DFlashPrefixSpeculatorConfig(transformer_layer_config=layer, draft_vocab_size=64,
        block_size=settings["block_size"], aux_hidden_state_layer_ids=[0, 1], mask_token_id=0,
        sample_from_anchor=False, prefix_selector_kind="compiled_prefix_v3", prefix_rank=settings["rank"],
        prefix_top_k=16, prefix_attention_heads=settings["heads"], prefix_attention_layers=1,
        prefix_loss_kind="target_ce", prefix_compiled_logit_cap=settings["compiled_logit_cap"],
        speculators_config=SpeculatorsConfig(algorithm="dflash_prefix",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=settings["block_size"]-1)],
            default_proposal_method="greedy", verifier=VerifierConfig(name_or_path=str(target), architectures=["Qwen3ForCausalLM"])))
    model = JointCompiledScratch(config)
    model.load_verifier_weights()
    model.configure_joint(settings)
    for p in model.parameters():
        p.data = p.data.float() if p.requires_grad else p.data.bfloat16()
    model.to(device)
    if settings.get("backbone_checkpointing", True):
        model.gradient_checkpointing_enable()
    length = max(48, 3 * settings["block_size"])
    last = torch.randn(1, length, 64, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        reference = model.verifier_lm_head(model.verifier_norm(last)).argmax(-1)
    tokens = torch.randint(0, 64, (1, length), device=device)
    tokens[:, 1:] = reference[:, :-1]
    batch = dict(hidden_states=torch.randn(1, length, 128, device=device, dtype=torch.bfloat16),
        input_ids=tokens, loss_mask=torch.ones(1, length, device=device),
        verifier_last_hidden_states=last, document_ids=torch.zeros(1, length, device=device, dtype=torch.long),
        max_anchors=4)
    return model, batch


def check(settings, device_name):
    import torch
    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(torch.device(device_name))
    from speculators.losses import resolve_loss_config
    from speculators.model import SpeculatorModel
    from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
    from speculators.models.dflash_prefix.joint_validation import checkpoint_parameters
    from speculators.models.dflash_prefix.joint_guard import bf16_replay_copy
    from speculators.train.checkpointer import SingleGPUCheckpointer
    from safetensors.torch import load_file
    device = torch.device(device_name)
    s = dict(settings, delay_steps=0, head_warmup_steps=1, guard_start_steps=0,
             guard_mode="project", guard_replay_anchors=4, guard_add_anchors=4,
             guard_chunk_size=settings.get("guard_chunk_size", 2),
             anchor_chunk_size=settings.get("anchor_chunk_size", 2))
    with tempfile.TemporaryDirectory() as directory:
        model, batch = make_fixture(directory, s, device)
        SpeculatorModel.verify_training_compatible(model)
        batch["loss_config"] = resolve_loss_config("ce", "eager")
        head = torch.optim.AdamW(model.prefix_head.parameters(), lr=3e-4)
        backbone = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("prefix_head.")], lr=1e-4)
        losses, reports = [], []
        for step in range(1, 4):
            model.train()
            model.joint_step = step
            model.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                _, loss, _ = model(**batch)
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
                    raise RuntimeError(f"Missing/nonfinite gradient: {name}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            backbone.step()
            reports.append(model.selector_guard.step(model.prefix_head, head, model.embed_tokens.weight))
            losses.append(float(loss.detach().cpu()))
        # A random tiny drafter may have zero target-correct first positions.
        # Exercise the nonempty guard path explicitly on synthetic labels too.
        replay = model.selector_guard.replay.data
        if replay is None:
            raise RuntimeError("Empty smoke replay")
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            initial = bf16_replay_copy(model.prefix_head).greedy(replay["hidden"], replay["candidates"], replay["unary"],
                                              replay["anchors"], embedding_weight=model.embed_tokens.weight)
        replay["reference"], replay["target"] = initial.clone(), initial.clone()
        replay["valid"].fill_(True)
        for parameter in model.prefix_head.parameters():
            parameter.grad = torch.randn_like(parameter) * .01
        synthetic_guard = model.selector_guard.step(model.prefix_head, head, model.embed_tokens.weight)
        if synthetic_guard["protected_positions"] <= 0 or synthetic_guard["accepted_lost_tokens"] != 0:
            raise RuntimeError("Nonempty synthetic guard check failed")
        root = Path(directory) / "checkpoints"
        SingleGPUCheckpointer(root).save_checkpoint(model, [backbone, head], 0)
        config = model.config.__class__.from_pretrained(root / "0")
        config.transformer_layer_config._attn_implementation = "sdpa"
        restored = DFlashPrefixDraftModel.from_pretrained(root / "0", config=config, local_files_only=True, dtype="auto").to(device)
        state = load_file(str(root / "0/model.safetensors"))
        for name, value in state.items():
            torch.testing.assert_close(restored.state_dict()[name].cpu(), value, atol=0, rtol=0)
        model.eval()
        with checkpoint_parameters(model, root / "0") as (replay_model, _):
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                torch.manual_seed(910)
                _, _, metrics = replay_model(**batch)
                if not torch.isfinite(metrics["rollout_reference_eal_sum"]):
                    raise RuntimeError("Nonfinite saved-weight validation")
        graph_checked = False
        if device.type == "npu":
            from speculators.models.dflash_prefix.compiled_inference import CompiledPrefixGraph
            replay = model.selector_guard.replay.data
            if replay is None:
                raise RuntimeError("Empty smoke replay")
            f = {k: v[:1].clone() for k, v in replay.items()}
            serving = restored.prefix_head.eval()
            graph = CompiledPrefixGraph(serving, logging.getLogger("joint_smoke"))
            with torch.inference_mode():
                for _ in range(3):
                    f["hidden"] = torch.randn_like(f["hidden"])
                    inputs = (f["hidden"], f["candidates"], f["unary"], f["anchors"])
                    expected, scores = serving.walk(serving.prepare_tables(*inputs, embedding_weight=restored.embed_tokens.weight), return_scores=True)
                    actual, actual_scores = graph.select(*inputs, embedding_weight=restored.embed_tokens.weight, return_scores=True)
                    production = graph.select(*inputs, embedding_weight=restored.embed_tokens.weight)
                    torch.testing.assert_close(scores, actual_scores, atol=1e-5, rtol=1e-5)
                    if not torch.equal(expected, actual) or not torch.equal(expected, production):
                        raise RuntimeError("Compiled graph changed selected tokens")
            graph_checked = True
        report = dict(device=str(device), backbone_and_head_backward=True, optimizer_steps=3,
            backbone_checkpointing=s.get("backbone_checkpointing", True),
            selector_checkpointing=s.get("selector_checkpointing", True),
            anchor_chunk_size=s["anchor_chunk_size"], guard_chunk_size=s["guard_chunk_size"],
            saved_checkpoint_loads_as="DFlashPrefixDraftModel", selector_kind="compiled_prefix_v3",
            guard_reports=reports, synthetic_guard_check=synthetic_guard, losses=losses, npu_graph_checked=graph_checked,
            note="Tiny synthetic correctness check; no claim about full-model memory, tau, speed or training convergence")
        print("JOINT_SCRATCH_SMOKE " + json.dumps(report), flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    check(json.loads(args.settings.read_text()), args.device)
