#!/usr/bin/env python3
"""Small differentiability and NPU graph check before the training pilot."""
import argparse
import copy
import json
import logging
from pathlib import Path


def check(device_name, block_size, teacher=None):
    import torch
    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(torch.device(device_name))
    from speculators.models.dflash_prefix.compiled_selector import CompiledPrefixSelector
    from speculators.models.dflash_prefix.selector_v2 import unary_topk
    device = torch.device(device_name)
    cfg = json.loads((Path(teacher) / "config.json").read_text()) if teacher else {}
    rank, heads = cfg.get("prefix_rank", 256), cfg.get("prefix_attention_heads", 4)
    hidden_size = cfg.get("transformer_layer_config", {}).get("hidden_size", 64)
    if cfg and (cfg.get("prefix_selector_kind") != "compiled_prefix_v3" or cfg.get("block_size") != block_size):
        raise ValueError("Teacher kind/block size differs from the prepared pilot")
    torch.manual_seed(42)
    head = CompiledPrefixSelector(97, hidden_size, rank=rank, top_k=16, max_proposals=block_size - 1, heads=heads).to(device)
    with torch.no_grad():
        head.successor_projection.weight.normal_(std=.03)
        head.head_weights.fill_(.2)
    embedding = torch.randn(97, hidden_size, device=device)
    hidden = torch.randn(1, block_size - 1, hidden_size, device=device)
    full_logits = torch.randn(1, block_size - 1, 97, device=device)
    unary, ids = unary_topk(full_logits, 16)
    anchors = torch.tensor([1], device=device)
    forced = torch.randint(16, (1, block_size - 1), device=device)
    prefix = ids.gather(-1, forced.unsqueeze(-1)).squeeze(-1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        scores = head.score_teacher_prefix(hidden, ids, unary, anchors, prefix, embedding_weight=embedding)
        _, walked = head.walk(head.prepare_tables(hidden, ids, unary, anchors, embedding_weight=embedding), forced_indices=forced, return_scores=True)
        loss = -scores.log_softmax(-1).gather(-1, forced.unsqueeze(-1)).mean()
    # Direct teacher forcing has different matrix shapes than all-candidate preparation.
    torch.testing.assert_close(scores, walked, atol=.03, rtol=.02)
    loss.backward()
    for name, parameter in head.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"Missing or nonfinite gradient: {name}")
    graph_checked = False
    graph_error = None
    if device.type == "npu":
        from speculators.models.dflash_prefix.compiled_inference import CompiledPrefixGraph
        serving = copy.deepcopy(head).eval().to(torch.bfloat16)
        graph = CompiledPrefixGraph(serving, logging.getLogger("compiled_prefix_check"))
        inputs = [hidden.to(torch.bfloat16), ids, unary, anchors]
        serving_embedding = embedding.to(torch.bfloat16)
        with torch.inference_mode():
            for _ in range(3):
                inputs[0] = torch.randn_like(inputs[0])
                expected, expected_scores = serving.walk(serving.prepare_tables(*inputs, embedding_weight=serving_embedding), return_scores=True)
                actual, actual_scores = graph.select(*inputs, embedding_weight=serving_embedding, return_scores=True)
                production = graph.select(*inputs, embedding_weight=serving_embedding)
                torch.testing.assert_close(actual_scores, expected_scores, atol=1e-5, rtol=1e-5)
                if not torch.equal(expected, actual) or not torch.equal(expected, production):
                    raise RuntimeError("NPU graph altered selected tokens")
                graph_error = float((actual_scores - expected_scores).abs().max().cpu())
        graph_checked = True
    print(json.dumps({"device": str(device), "block_size": block_size, "rank": rank, "heads": heads,
                      "finite_gradients": True, "npu_graph_checked": graph_checked,
                      "last_graph_score_error": graph_error,
                      "teacher_walk_max_error": float((scores - walked).abs().max().detach().cpu()),
                      "note": "Synthetic small-head check; neither target-model parity nor a speed benchmark"}, indent=2))
    from speculators.models.dflash_prefix.boundary_objective import boundary_loss_terms, boundary_global_loss, dense_boundary_anchors
    anchors, _ = dense_boundary_anchors(torch.ones(1, 64, device=device), 8, block_size, training=False)
    if not torch.equal(anchors[1:] - anchors[:-1], torch.ones_like(anchors[1:])):
        raise RuntimeError("Teacher refresh requires contiguous anchors")
    s = torch.tensor([[[0., 1.], [0., 1.], [0., 1.]]], device=device, requires_grad=True)
    t = torch.tensor([[[2., 0.], [2., 0.], [2., 0.]]], device=device)
    ids = torch.tensor([[[0, 1], [0, 1], [0, 1]]], device=device)
    valid = torch.ones(1, 3, device=device, dtype=torch.bool)
    terms, _ = boundary_loss_terms(s, s.detach(), t, ids,
        torch.zeros(1, 3, device=device, dtype=torch.long), valid, valid, valid, {})
    terms["full_ce"] = (s.sum() * 0, s.new_tensor(1.))
    loss = boundary_global_loss(terms, {})
    loss.backward()
    if terms["boundary_kl"][1] != 2 or not torch.isfinite(s.grad).all():
        raise RuntimeError("Boundary distillation gradient/causality smoke check failed")
    print("BOUNDARY_TRAINING_SMOKE_PASS: first slot excluded; fresh teacher detached; finite gradients")
    from speculators.models.dflash_prefix.boundary_support import support_transfer_terms
    # Target 3 is outside the natural top-2, but must still get support gradient.
    full = torch.tensor([[[2., 1., 0., -2.]] * 3], device=device, requires_grad=True)
    refresh = torch.tensor([[[0., 0., 0., 4.]] * 3], device=device)
    target = torch.full((1, 3), 3, device=device, dtype=torch.long)
    support, _ = support_transfer_terms(full, full.detach(), refresh, target,
                                        valid, ~valid)
    support["support_kl"][0].backward()
    if (support["support_kl"][1] != 2 or not torch.isfinite(full.grad).all()
            or not bool((full.grad[:, 1:, 3] < 0).all()) or bool(full.grad[:, 0].any())):
        raise RuntimeError("Full-vocabulary missing-candidate support gradient check failed")
    print("BOUNDARY_SUPPORT_SMOKE_PASS: omitted target receives gradient; first slot excluded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--teacher")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    check(args.device, args.block_size, args.teacher)
