#!/usr/bin/env python3
"""Small CPU/NPU check before spending compute on the experiment."""
import argparse
import json

import torch
from speculators.models.dflash_prefix.selector_v2 import LocalPrefixSelector, unary_topk


def check(device, block_size=8, top_k=16):
    if block_size < 2 or not 2 <= top_k <= 97:
        raise ValueError("Smoke check needs block_size >= 2 and 2 <= top_k <= 97")
    proposals = block_size - 1
    torch.manual_seed(42)
    head = LocalPrefixSelector(97, 64, rank=32, top_k=top_k, max_proposals=proposals,
                               heads=4, layers=2).to(device)
    weight = torch.randn(97, 64, device=device)
    hidden = torch.randn(2, proposals, 64, device=device)
    full_logits = torch.randn(2, proposals, 97, device=device)
    unary, ids = unary_topk(full_logits, top_k)
    anchor = torch.tensor([1, 2], device=device)
    forced = torch.randint(top_k, (2, proposals), device=device)
    reference = ids.gather(-1, forced.unsqueeze(-1)).squeeze(-1)
    initial = head.greedy(hidden, ids, unary, anchor, embedding_weight=weight)
    torch.testing.assert_close(initial, full_logits.argmax(-1))
    with torch.no_grad():
        head.successor_projection.weight.normal_(std=.03)
        head.prefix_output.weight.normal_(std=.03)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        teacher = head.score_teacher_prefix(hidden, ids, unary, anchor, reference,
                                            embedding_weight=weight)
        table = head.prepare_tables(hidden, ids, unary, anchor, embedding_weight=weight)
        _, walk_scores = head.walk(table, forced_indices=forced, return_scores=True)
        loss = -teacher.log_softmax(-1).gather(-1, forced.unsqueeze(-1)).mean()
    # Different matrix shapes can produce BF16 rounding differences on the NPU.
    torch.testing.assert_close(teacher, walk_scores, rtol=.015, atol=.015)
    loss.backward()
    for name, parameter in head.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing/non-finite selector gradient: {name}")
    print(json.dumps({"device": str(device), "block_size": block_size,
                      "num_speculative_tokens": proposals, "top_k": top_k,
                      "zero_init_matches_unary": True,
                      "max_teacher_walk_score_difference": float((teacher-walk_scores).abs().max().detach().cpu()),
                      "finite_gradients": True, "note": "Correctness smoke test; no tau claim."}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    check(torch.device(args.device), args.block_size, args.top_k)
