"""Export trained native DSpark ``mtp.*`` weights into a HF target checkpoint.

Run on rank 0 after training, for example:

python -m speculators.models.dspark_dsv4_native.export_hf \
  --target-model /path/DeepSeek-V4-Flash-bf16 \
  --checkpoint /path/checkpoints/latest/pytorch_model.bin \
  --output /path/DeepSeek-V4-Flash-DSpark-native
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _strip_fsdp_prefix(key: str) -> str:
    prefixes = (
        "module.",
        "_fsdp_wrapped_module.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        # Common Speculators/HF trainer layouts.
        for candidate in (
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "model.pt",
            path / "model_state.pt",
            path / "state_dict.pt",
        ):
            if candidate.exists():
                path = candidate
                break
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise TypeError(f"Unsupported checkpoint object from {path}: {type(obj)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dspark-block-size", type=int, default=5)
    parser.add_argument("--dspark-noise-token-id", type=int, default=128799)
    parser.add_argument("--dspark-target-layer-ids", type=int, nargs="+", default=[40, 41, 42])
    parser.add_argument("--dspark-markov-rank", type=int, default=256)
    parser.add_argument("--num-mtp-layers", type=int, default=3)
    args = parser.parse_args()

    src = Path(args.target_model)
    out = Path(args.output)
    ckpt = Path(args.checkpoint)

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)

    raw_state = _load_state(ckpt)
    mtp_state: dict[str, torch.Tensor] = {}
    for key, tensor in raw_state.items():
        key = _strip_fsdp_prefix(key)
        if key.startswith("mtp.") and isinstance(tensor, torch.Tensor):
            mtp_state[key] = tensor.detach().cpu().contiguous()
    if not mtp_state:
        raise RuntimeError(f"No mtp.* tensors found in checkpoint {ckpt}")

    mtp_file = "model-mtp-00001-of-00001.safetensors"
    save_file(mtp_state, out / mtp_file)

    config_path = out / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["n_mtp_layers"] = args.num_mtp_layers
    cfg["dspark_block_size"] = args.dspark_block_size
    cfg["dspark_noise_token_id"] = args.dspark_noise_token_id
    cfg["dspark_target_layer_ids"] = args.dspark_target_layer_ids
    cfg["dspark_markov_rank"] = args.dspark_markov_rank

    # Native DSpark stages are appended after n_layers and use uncompressed
    # sliding-window DSpark attention, so the final ratios must be zeros.
    n_layers = int(cfg.get("n_layers", cfg.get("num_hidden_layers", 43)))
    expected_len = n_layers + args.num_mtp_layers
    ratios = list(cfg.get("compress_ratios", []))
    if ratios:
        if len(ratios) < expected_len:
            ratios.extend([0] * (expected_len - len(ratios)))
        ratios[-args.num_mtp_layers :] = [0] * args.num_mtp_layers
        cfg["compress_ratios"] = ratios

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.setdefault("weight_map", {})
        for key in mtp_state:
            weight_map[key] = mtp_file
        meta = index.setdefault("metadata", {})
        old_size = int(meta.get("total_size", 0))
        add_size = sum(t.numel() * t.element_size() for t in mtp_state.values())
        meta["total_size"] = old_size + add_size
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    else:
        # If the target directory was unindexed, create a tiny mtp-only index.
        with open(out / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "total_size": sum(t.numel() * t.element_size() for t in mtp_state.values())
                    },
                    "weight_map": {key: mtp_file for key in mtp_state},
                },
                f,
                indent=2,
            )

    print(f"Exported {len(mtp_state)} native DSpark tensors to {out}")


if __name__ == "__main__":
    main()
