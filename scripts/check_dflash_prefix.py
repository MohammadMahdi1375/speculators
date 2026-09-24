#!/usr/bin/env python3
"""Validate causal prefix attention, BF16 arithmetic, gradients and greedy walks."""

import argparse
import copy
import json

import torch
from speculators.models.dflash_prefix.model_definitions import PrefixAttentionSelector


def run(device, rank=64, top_k=16, steps=15, backend="torch"):
    if device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(torch.device(device))
    if steps < 3 or top_k < 2 or rank < 8:
        raise ValueError("Use steps>=3, top_k>=2, rank>=8")
    if backend not in {"torch", "triton", "auto"}:
        raise ValueError("Unknown walk backend")
    torch.manual_seed(123)
    vocab = steps * top_k + 8
    head = PrefixAttentionSelector(vocab, 128, rank, top_k, steps).to(device)
    hidden = torch.randn(2, steps, 128, device=device)
    candidates = torch.arange(steps * top_k, device=device).reshape(1, steps, top_k)
    candidates = candidates.expand(2, -1, -1).clone()
    unary = torch.randn(2, steps, top_k, device=device)
    anchors = torch.tensor([vocab - 1, vocab - 2], device=device)
    path = torch.randint(top_k, (2, steps), device=device)
    reference = candidates.gather(-1, path.unsqueeze(-1)).squeeze(-1)
    results = []
    fused_status = "not_requested"
    fused_walk = None
    if backend != "torch":
        try:
            from vllm_ascend.ops.triton.spec_decode.dflash_prefix import (
                greedy_select_prefix,
            )

            fused_walk = greedy_select_prefix
        except Exception as exc:
            if backend == "triton":
                raise
            fused_status = f"unavailable: {type(exc).__name__}: {exc}"

    for dtype in (torch.float32, torch.bfloat16):
        tested = copy.deepcopy(head).to(dtype=dtype)
        h = hidden.to(dtype=dtype)
        u = unary.to(dtype=dtype)
        teacher = tested.score_teacher_prefix(h, candidates, u, anchors, reference)
        tables = tested.prepare_tables(h, candidates, u, anchors)
        _, _, walked = tested.walk(tables, forced_indices=path)
        tol = 3e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(teacher, walked, rtol=tol, atol=tol)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            autocast_teacher = tested.score_teacher_prefix(
                h, candidates, u, anchors, reference
            )
        if dtype == torch.bfloat16:
            torch.testing.assert_close(teacher, autocast_teacher, rtol=tol, atol=tol)
        changed = reference.clone()
        changed[:, 2:] = vocab - 3
        changed_logits = tested.score_teacher_prefix(h, candidates, u, anchors, changed)
        torch.testing.assert_close(
            teacher[:, :3], changed_logits[:, :3], rtol=0, atol=0
        )
        older = reference.clone()
        older[:, 0] = vocab - 4
        older_logits = tested.score_teacher_prefix(h, candidates, u, anchors, older)
        if torch.allclose(teacher[:, -1], older_logits[:, -1]):
            raise AssertionError("Older prefix token failed to affect the head")
        loss = -teacher.log_softmax(-1)[:, :, 0].mean()
        loss.backward()
        for name, parameter in tested.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Missing/nonfinite gradient: {name}")
        expected, probabilities, _ = tested.walk(tables)
        actual = tested.greedy_walk(tables)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(
            probabilities.sum(-1), torch.ones_like(probabilities[..., 0])
        )
        if not ((probabilities == 0) | (probabilities == 1)).all():
            raise AssertionError("Greedy proposal must be a point mass")
        if fused_walk is not None:
            try:
                fused = fused_walk(tables)
                torch.testing.assert_close(expected, fused, rtol=0, atol=0)
                # Flat score ties must resolve to the first candidate. Random
                # inputs alone would rarely exercise this boundary condition.
                tied = tables._replace(
                    unary_logits=torch.zeros_like(tables.unary_logits),
                    gate=torch.zeros_like(tables.gate),
                )
                torch.testing.assert_close(
                    fused_walk(tied), candidates[:, :, 0], rtol=0, atol=0
                )
                fused_status = "passed"
            except Exception as exc:
                if backend == "triton":
                    raise
                fused_status = f"unavailable: {type(exc).__name__}: {exc}"
                fused_walk = None
        results.append({"dtype": str(dtype), "status": "passed"})
    report = {
        "device": device,
        "rank": rank,
        "top_k": top_k,
        "steps": steps,
        "checks": results,
        "fused_walk": fused_status,
    }
    print(json.dumps(report, indent=2))
    print(
        "PASS: prefix causality, older-token dependence, teacher/table parity, finite gradients, native greedy parity"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument(
        "--backend", choices=("torch", "triton", "auto"), default="torch"
    )
    run(**vars(parser.parse_args()))
