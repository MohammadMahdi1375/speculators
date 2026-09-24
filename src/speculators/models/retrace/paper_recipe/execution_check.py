"""Check SDPA masks/gradients and optional NPU RMSNorm on the selected device.

This is an operator check, not a greedy-generation or Table 2 validation.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from .execution import FrozenNPURMSNorm, draft_attention
from .runtime import device_for


def attention_case(name, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(1729)
    heads, kv_heads, dim = 32, 8, 128
    qlen, klen = (33, 33) if name == "prefill" else (16, 127)
    if name == "packed_blocks":
        qlen, klen = 48, 147
    inputs = [
        torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
        for shape in (
            (2, heads, qlen, dim),
            (2, kv_heads, klen, dim),
            (2, kv_heads, klen, dim),
        )
    ]
    key = torch.arange(klen, device=device)
    query = torch.arange(qlen, device=device)
    if name == "prefill":
        visible = key[None] <= query[:, None]
        visible = visible[None].expand(2, -1, -1)
    elif name == "rollback":
        positions = torch.tensor([17, 111], device=device)[:, None] + query[None]
        visible = key[None, None] <= positions[:, :, None]
    elif name == "draft":
        visible = (
            (key[None] < torch.tensor([17, 111], device=device)[:, None])
            | (key[None] >= 111)
        )[:, None].expand(-1, qlen, -1)
    elif name == "packed_blocks":
        blocks = query // 16
        anchors = torch.tensor([8, 44, 87], device=device)
        visible = torch.cat(
            (key[:99][None] < anchors[blocks, None], blocks[:, None] == blocks[None]),
            dim=1,
        )
        visible = visible[None].expand(2, -1, -1)
    else:
        raise ValueError(name)
    mask = torch.zeros(2, 1, qlen, klen, device=device, dtype=dtype)
    mask.masked_fill_(~visible[:, None], float("-inf"))
    probe = torch.randn((2, qlen, heads, dim), generator=generator).to(
        device=device, dtype=dtype
    )
    outputs, gradients = [], []
    for backend in ("eager", "sdpa"):
        module = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation=backend),
            num_key_value_groups=heads // kv_heads,
            is_causal=False,
            training=True,
        )
        q, k, v = [x.detach().clone().requires_grad_() for x in inputs]
        out, _ = draft_attention(module, q, k, v, mask, dropout=0.0, scaling=dim**-0.5)
        grads = torch.autograd.grad((out * probe).sum(), (q, k, v))
        outputs.append(out.detach())
        gradients.append([g.detach() for g in grads])
    tolerance = 0.035 if dtype == torch.bfloat16 else 2e-5
    for actual, expected in [
        (outputs[1], outputs[0]),
        *zip(gradients[1], gradients[0], strict=True),
    ]:
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def relative(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))

    errors = [relative(outputs[1], outputs[0])] + [
        relative(a, b) for a, b in zip(gradients[1], gradients[0], strict=True)
    ]
    if not all(math.isfinite(e) and e < tolerance for e in errors):
        raise RuntimeError(f"Attention relative error too large: {name}: {errors}")
    return {
        "case": name,
        "forward_relative_l2": errors[0],
        "gradient_relative_l2": errors[1:],
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--target-norm", choices=("reference", "npu"), default="npu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output file")
    device, dtype = device_for(args.device), getattr(torch, args.dtype)
    report = {
        "device": str(device),
        "dtype": args.dtype,
        "checks": [],
        "greedy_parity_validated": False,
        "limit": "Synthetic operator/mask/gradient check; no model-level equivalence or speed claim",
    }
    try:
        for name in ("prefill", "rollback", "draft", "packed_blocks"):
            report["checks"].append(attention_case(name, device, dtype))
        if args.target_norm == "npu":
            for width in (128, 2560):
                norm = (
                    Qwen3RMSNorm(width)
                    .to(device=device, dtype=dtype)
                    .requires_grad_(False)
                )
                x = torch.linspace(
                    -2, 2, 6 * width, device=device, dtype=dtype
                ).reshape(2, 3, width)
                with torch.no_grad():
                    actual = FrozenNPURMSNorm(norm)(x)
                    tolerance = 0.035 if dtype == torch.bfloat16 else 1e-5
                    torch.testing.assert_close(
                        actual, norm(x), atol=tolerance, rtol=tolerance
                    )
                report["checks"].append(
                    {"case": f"target_rmsnorm_{width}", "passed": True}
                )
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        with output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
