#!/usr/bin/env python3
"""Copy the experiment's source files, including untracked additions."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patterns = {
        "speculators": [
            "src/speculators/models/dflash_prefix/*.py",
            "src/speculators/models/__init__.py",
            "src/speculators/train/config/schema.py",
            "src/speculators/train/config/resolution.py",
            "src/speculators/train/optimizers.py",
            "scripts/*dflash_prefix*.py",
            "examples/train/dflash_prefix_qwen3_4b_ascend.sh",
        ],
        "vllm": [
            "vllm/model_executor/models/qwen3_dflash_prefix.py",
            "vllm/model_executor/models/registry.py",
            "vllm/transformers_utils/configs/speculators/*.py",
        ],
        "vllm-ascend": [
            "vllm_ascend/spec_decode/dflash_prefix_proposer.py",
            "vllm_ascend/spec_decode/dflash_proposer.py",
            "vllm_ascend/ops/triton/spec_decode/dflash_prefix.py",
            "vllm_ascend/spec_decode/__init__.py",
        ],
    }
    hashes = {}
    for repo, globs in patterns.items():
        root = args.spec_main / repo
        for pattern in globs:
            for source in root.glob(pattern):
                relative = Path(repo) / source.relative_to(root)
                dest = args.output / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
