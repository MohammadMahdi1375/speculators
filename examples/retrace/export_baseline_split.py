#!/usr/bin/env python
"""Save the same training rows, including recorded responses, for DSpARK."""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from speculators.models.retrace.data import read_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int, default=40000)
    parser.add_argument("--prompt-length", type=int, default=2048)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Use a new output directory")
    if not Path(args.data).is_dir():
        parser.error("This export needs the original Arrow dataset with responses")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    _, manifest = read_prompts(
        args.data, len(tokenizer), args.max_prompts, args.prompt_length, "train"
    )
    dataset = load_from_disk(args.data)
    dataset.select(manifest["rows"]).save_to_disk(str(output))
    (output / "retrace_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Saved {manifest['count']} original training rows to {output}; responses and masks preserved"
    )


if __name__ == "__main__":
    main()
