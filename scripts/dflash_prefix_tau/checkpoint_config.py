#!/usr/bin/env python3
"""Read checkpoint metadata without importing torch; emit serving JSON."""
import argparse
import json
from pathlib import Path
import sys


def speculative_config(checkpoint, expected_block_size):
    path = Path(checkpoint).resolve(strict=True)
    cfg = json.loads((path / "config.json").read_text())
    if (cfg.get("speculators_model_type") != "dflash_prefix"
            or cfg.get("prefix_selector_kind") != "local_prefix_v2"):
        raise ValueError("Choose a local_prefix_v2 DFlash-prefix checkpoint")
    block = cfg.get("block_size")
    if type(block) is not int or block < 2:
        raise ValueError("Checkpoint block_size must include an anchor and a proposal")
    if block != expected_block_size:
        raise ValueError(
            f"Checkpoint block_size={block}, but BLOCK_SIZE={expected_block_size}. "
            "Select the correct run; do not edit checkpoint config.json."
        )
    if cfg.get("prefix_history_scale") != 1:
        raise ValueError("Expected the trained checkpoint with prefix history enabled")
    if cfg.get("prefix_walk_backend", "torch") != "torch":
        raise ValueError("local_prefix_v2 requires the torch walk backend")
    print(f"Serving checkpoint: {path}", file=sys.stderr)
    print(f"block_size={block}; num_speculative_tokens={block - 1}; "
          f"prefix_top_k={cfg.get('prefix_top_k')}", file=sys.stderr)
    return {
        "method": "dflash", "model": str(path),
        "num_speculative_tokens": block - 1,
        "draft_sample_method": "greedy", "disable_padded_drafter_batch": False,
        "enable_adaptive_verification": False, "enforce_eager": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--expected-block-size", type=int, default=8)
    args = parser.parse_args()
    try:
        print(json.dumps(speculative_config(args.draft, args.expected_block_size)))
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
