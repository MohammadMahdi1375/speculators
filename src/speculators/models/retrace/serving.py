"""Checkpoint identity and supported native inference configuration."""

import json
from pathlib import Path


def speculative_config(path):
    config = json.loads((Path(path) / "config.json").read_text())
    kind = config.get("speculators_model_type")
    if kind == "retrace":
        if config.get("retrace_format_version") != 1 or config.get("architectures") != [
            "ReTraceDraftModel"
        ]:
            raise ValueError("Unrecognized ReTrace checkpoint format")
        if config.get("sample_from_anchor", False):
            raise ValueError("ReTrace must preserve the anchor slot")
        method = "dflash"
        k = config["block_size"] - 1
    elif kind == "dspark":
        if config.get("retrace_enabled", False):
            raise ValueError(
                "Legacy combined DSpARK+ReTrace checkpoints require the original runtime"
            )
        method = "dspark"
        k = config["block_size"] - (0 if config.get("sample_from_anchor", True) else 1)
    elif kind == "dflash":
        method = "dflash"
        k = config["block_size"] - 1
    else:
        raise ValueError(f"Unsupported checkpoint identity: {kind}")
    if k < 1:
        raise ValueError("Need at least one speculative token")
    return {
        "method": method,
        "model": str(Path(path).resolve()),
        "num_speculative_tokens": k,
        "draft_sample_method": "greedy",
        "disable_padded_drafter_batch": False,
    }
