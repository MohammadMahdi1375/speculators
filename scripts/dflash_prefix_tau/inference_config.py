#!/usr/bin/env python3
"""Check target/draft roles and create serving JSON without loading torch."""
import argparse
import json
from pathlib import Path
import sys


def serving_config(target, draft):
    target, draft = Path(target).resolve(strict=True), Path(draft).resolve(strict=True)
    target_cfg = json.loads((target / "config.json").read_text())
    cfg = json.loads((draft / "config.json").read_text())
    if target == draft or target_cfg.get("model_type") != "qwen3" or target_cfg.get("speculators_model_type"):
        raise ValueError("TARGET must be the original Qwen3-4B target, not a draft checkpoint")
    if cfg.get("speculators_model_type") != "dflash_prefix" or cfg.get("prefix_selector_kind") != "local_prefix_v2":
        raise ValueError("DRAFT must be the trained local_prefix_v2 DFlash-prefix checkpoint")
    if cfg.get("prefix_disable_selector", False) or cfg.get("prefix_inference_gate_scale", 1.0) != 1.0:
        raise ValueError("Choose the unmodified prefix checkpoint (selector enabled, inference scale one)")
    if cfg.get("prefix_history_scale", 1.0) != 1.0 or cfg.get("prefix_walk_backend", "torch") != "torch":
        raise ValueError("This comparison requires history_scale=1 and prefix_walk_backend=torch")
    block = cfg.get("block_size")
    if block not in (8, 16):
        raise ValueError(f"This launcher expects a trained block-8 or block-16 checkpoint, got {block}")
    layer_cfg = cfg.get("transformer_layer_config", {})
    for key in ("hidden_size", "vocab_size"):
        if key in layer_cfg and key in target_cfg and layer_cfg[key] != target_cfg[key]:
            raise ValueError(f"Target/draft {key} mismatch")
    print(f"TARGET: {target}\nDRAFT:  {draft}", file=sys.stderr)
    print(f"selector={cfg['prefix_selector_kind']}, rank={cfg.get('prefix_rank')}, "
          f"heads={cfg.get('prefix_attention_heads')}, layers={cfg.get('prefix_attention_layers')}, "
          f"top-K={cfg.get('prefix_top_k')}", file=sys.stderr)
    print(f"trained block={block}; proposals={block - 1}; KV-cache block=128", file=sys.stderr)
    return {"method": "dflash", "model": str(draft), "num_speculative_tokens": block - 1,
            "draft_sample_method": "greedy", "disable_padded_drafter_batch": False,
            "enable_adaptive_verification": False, "enforce_eager": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(serving_config(args.target, args.draft)))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
