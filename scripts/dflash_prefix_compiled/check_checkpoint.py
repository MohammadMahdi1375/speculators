#!/usr/bin/env python3
"""Validate compiled checkpoint metadata without importing Torch."""
import argparse
import json
from pathlib import Path
import sys


def speculative_config(draft, target, expected_block=None):
    path, target = Path(draft).expanduser().resolve(strict=True), Path(target).expanduser().resolve(strict=True)
    cfg = json.loads((path / "config.json").read_text())
    target_cfg = json.loads((target / "config.json").read_text())
    target_cfg = target_cfg.get("text_config", target_cfg)
    if cfg.get("speculators_model_type") != "dflash_prefix" or cfg.get("prefix_selector_kind") != "compiled_prefix_v3":
        raise ValueError("DRAFT must be a trained compiled_prefix_v3 checkpoint, not the old Prefix teacher")
    block = cfg.get("block_size")
    if block not in (8, 16) or cfg.get("prefix_top_k") != 16:
        raise ValueError("Expected block size 8/16 and shortlist top-K=16")
    if expected_block is not None and expected_block != block:
        raise ValueError("BLOCK_SIZE differs from the checkpoint; do not modify checkpoint config")
    for field in ("vocab_size", "hidden_size"):
        if cfg["transformer_layer_config"].get(field) != target_cfg.get(field):
            raise ValueError(f"Target/draft {field} mismatch")
    if cfg.get("prefix_disable_selector", False) or cfg.get("prefix_inference_gate_scale", 1) != 1:
        raise ValueError("Compiled selector must be enabled with scale one")
    if not list(path.glob("*.safetensors")):
        raise ValueError("No safetensors checkpoint weights found")
    print(f"COMPILED_PREFIX_CHECKPOINT draft={path}; block={block}; proposals={block - 1}; target={target}", file=sys.stderr)
    return {"method": "dflash", "model": str(path), "num_speculative_tokens": block - 1,
            "draft_sample_method": "greedy", "disable_padded_drafter_batch": False,
            "enable_adaptive_verification": False, "enforce_eager": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--block-size", type=int)
    args = parser.parse_args()
    print(json.dumps(speculative_config(args.draft, args.target, args.block_size)))
