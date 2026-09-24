"""Build a deterministic, deduplicated prompt-only pool from local Arrow rows."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from .pretrained import sha256


def prompt_key(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def extract_prompt(row, vocabulary):
    ids, mask = row["input_ids"], row["loss_mask"]
    length = int(row.get("seq_len", len(ids)))
    if (
        len(ids) != len(mask)
        or not 0 < length <= len(ids)
        or any(v not in (0, 1, False, True) for v in mask)
    ):
        raise ValueError("Invalid Arrow input_ids/loss_mask/seq_len")
    boundary = next((i for i, value in enumerate(mask[:length]) if value), None)
    if boundary is None:
        return None
    prompt = ids[:boundary]
    if any(type(x) is not int or not 0 <= x < vocabulary for x in prompt):
        raise ValueError("Prompt contains an invalid token ID")
    return prompt


def disable_thinking(ids, tokenizer):
    open_assistant = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    suffix = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    if ids[-len(open_assistant) :] == open_assistant:
        return ids + suffix
    complete = open_assistant + suffix
    if ids[-len(complete) :] == complete:
        return ids
    raise ValueError(
        "Prompt must end in Qwen's assistant header or its empty-think suffix"
    )


def prepare_pool(source, target, output, count=40000, prompt_length=512, seed=42):
    from datasets import Dataset, load_from_disk

    output = Path(output)
    if output.exists():
        raise ValueError(f"Prompt pool already exists: {output}")
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    vocabulary = len(tokenizer)
    dataset = load_from_disk(source)
    if not isinstance(dataset, Dataset) or not {"input_ids", "loss_mask"}.issubset(
        dataset.column_names
    ):
        raise ValueError(
            "Pass one save_to_disk Arrow Dataset with input_ids and loss_mask"
        )
    dataset = dataset.with_format(None)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    counts = dict(scanned=0, no_response=0, overlength=0, duplicate=0, exported=0)
    keys, rows = set(), []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        with temporary.open("x") as handle:
            for index in indices:
                counts["scanned"] += 1
                ids = extract_prompt(dataset[index], vocabulary)
                if not ids:
                    counts["no_response"] += 1
                    continue
                try:
                    ids = disable_thinking(ids, tokenizer)
                except ValueError as exc:
                    raise ValueError(f"Dataset row {index}: {exc}") from exc
                if len(ids) > prompt_length:
                    counts["overlength"] += 1
                    continue
                key = prompt_key(ids)
                if key in keys:
                    counts["duplicate"] += 1
                    continue
                keys.add(key)
                rows.append(index)
                handle.write(
                    json.dumps({"input_ids": ids, "source_row": index, "sha256": key})
                    + "\n"
                )
                counts["exported"] += 1
                if counts["exported"] % 1000 == 0:
                    print(
                        f"Selected {counts['exported']:,}/{count:,} distinct prompts",
                        flush=True,
                    )
                if counts["exported"] == count:
                    break
        if counts["exported"] != count:
            raise ValueError(
                f"Only {counts['exported']} eligible distinct prompts; requested {count}"
            )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "source": str(Path(source).absolute()),
        "source_fingerprint": dataset._fingerprint,
        "source_rows": len(dataset),
        "selected_rows": rows,
        "counts": counts,
        "seed": seed,
        "prompt_length": prompt_length,
        "thinking": False,
        "selection": "seeded row permutation, unique prompt tokens, skip overlength",
        "output_sha256": sha256(output),
        "author_prompt_ids_available": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int, default=40000)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.max_prompts, args.prompt_length) < 1:
        parser.error("Counts and lengths must be positive")
    result = prepare_pool(
        args.data,
        args.target,
        args.output,
        args.max_prompts,
        args.prompt_length,
        args.seed,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "selected_rows"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
