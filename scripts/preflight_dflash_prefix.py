#!/usr/bin/env python3
"""Validate local model/data/device allocation before starting expensive jobs."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-layers", type=int, nargs="+", required=True)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--subset-data-path", type=Path)
    parser.add_argument("--subset-rows", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import speculators
    from speculators.models.dflash_prefix import DFlashPrefixDraftModel
    from speculators.train.config.schema import TrainConfig

    cfg = json.loads((args.model / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    if cfg["vocab_size"] != 151936:
        raise SystemExit("Expected the requested Qwen3-4B vocabulary, 151936")
    if not 0 <= args.mask_token_id < cfg["vocab_size"]:
        raise SystemExit("Mask token is outside the vocabulary")
    if any(i < 1 or i > cfg["num_hidden_layers"] for i in args.target_layers):
        raise SystemExit("Requested target layer is outside the verifier")
    if "dflash_prefix" not in TrainConfig.model_fields:
        raise SystemExit("Training schema is not patched")
    from datasets import Dataset, load_from_disk

    data = load_from_disk(str(args.dataset))
    if not isinstance(data, Dataset):
        raise SystemExit("Expected a prepared Dataset, not DatasetDict")
    if not {"input_ids", "loss_mask"}.issubset(data.column_names):
        raise SystemExit("Prepared dataset needs input_ids and loss_mask")
    if len(data) < 24:
        raise SystemExit("Dataset is too small for this six-rank smoke/pilot setup")
    for row in data.select(range(min(4, len(data)))):
        if len(row["input_ids"]) != len(row["loss_mask"]):
            raise SystemExit("input_ids/loss_mask lengths differ")
        if min(row["input_ids"]) < 0 or max(row["input_ids"]) >= cfg["vocab_size"]:
            raise SystemExit("Dataset contains out-of-vocabulary token IDs")
    server = os.environ["VLLM_NPUS"].split(",")
    train = os.environ["TRAIN_NPUS"].split(",")
    if (
        len(set(server)) != len(server)
        or len(set(train)) != len(train)
        or set(server) & set(train)
    ):
        raise SystemExit("NPU assignments overlap or contain duplicate IDs")
    if len(train) != int(os.environ["NUM_TRAIN_NPUS"]):
        raise SystemExit("NUM_TRAIN_NPUS does not match TRAIN_NPUS")
    if len(server) != int(os.environ["VLLM_DP"]):
        raise SystemExit("This launcher uses TP=1: one server NPU per DP rank")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.subset_data_path:
        if args.subset_data_path.exists():
            raise SystemExit("Subset output already exists; choose a fresh OUTPUT_DIR")
        if args.subset_rows < 24:
            raise SystemExit("Use at least 24 subset rows for six ranks")
        subset = data.shuffle(seed=args.seed).select(
            range(min(args.subset_rows, len(data)))
        )
        subset.save_to_disk(str(args.subset_data_path))
        print("Pilot/smoke subset rows:", len(subset))
    print("Speculators:", speculators.__file__)
    print("New model:", DFlashPrefixDraftModel.__name__)
    print("Prepared rows:", len(data), "Target layers:", args.target_layers)
    print("NPU allocation: server", server, "trainer", train)


if __name__ == "__main__":
    main()
