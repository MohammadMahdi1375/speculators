"""Read prompt prefixes directly from existing Arrow data or exported JSONL."""

import hashlib
import json
from pathlib import Path


def read_prompts(
    path, vocabulary, max_prompts=40000, max_length=2048, split="train", seed=42
):
    if split not in {"train", "validation", "all"}:
        raise ValueError("split must be train, validation, or all")
    if max_prompts < 0 or max_length <= 0:
        raise ValueError("max_prompts must be nonnegative and max_length positive")
    path = Path(path)
    if path.is_dir():
        from datasets import Dataset, load_from_disk

        dataset = load_from_disk(str(path))
        if not isinstance(dataset, Dataset):
            raise ValueError("Pass a single save_to_disk Dataset directory")
        dataset = dataset.with_format(None)
        if not {"input_ids", "loss_mask"}.issubset(dataset.column_names):
            raise ValueError("Arrow data needs input_ids and loss_mask")
        rows = iter(
            dataset.select_columns(
                [
                    x
                    for x in ("input_ids", "loss_mask", "seq_len")
                    if x in dataset.column_names
                ]
            )
        )
        fingerprint = dataset._fingerprint
    else:

        def json_rows():
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

        rows = json_rows()
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1048576), b""):
                hasher.update(block)
        fingerprint = hasher.hexdigest()
    prompts, row_ids = [], []
    skipped = 0
    for index, row in enumerate(rows):
        # Same prefix always lands in the same partition, even across duplicate
        # rollout rows or Arrow/JSONL representations of the same examples.
        if not isinstance(row, dict) or "input_ids" not in row:
            raise ValueError(
                f"Row {index}: expected an object with exact input_ids; export prompt tokens first"
            )
        ids = row["input_ids"]
        if "loss_mask" in row:
            mask = row["loss_mask"]
            length = int(row.get("seq_len", len(ids)))
            if len(mask) != len(ids) or not 0 < length <= len(ids):
                raise ValueError(f"Row {index}: invalid sequence/mask length")
            if any(value not in (0, 1, False, True) for value in mask):
                raise ValueError(f"Row {index}: loss_mask is not binary")
            start = next((j for j, value in enumerate(mask[:length]) if value), None)
            if start is None:
                skipped += 1
                continue
            ids = ids[:start]
        if not isinstance(ids, list) or any(
            type(token) is not int or not 0 <= token < vocabulary for token in ids
        ):
            raise ValueError(f"Row {index}: invalid token IDs")
        if not ids or len(ids) > max_length:
            skipped += 1
            continue
        key = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).digest()
        validation = int.from_bytes(key[:8], "big") % 20 == 0
        if split != "all" and validation != (split == "validation"):
            continue
        prompts.append(ids)
        row_ids.append(index)
        if len(prompts) % 5000 == 0:
            print(f"Loaded {len(prompts):,} {split} prompts", flush=True)
        if max_prompts and len(prompts) >= max_prompts:
            break
    if not prompts:
        raise ValueError(f"No valid prompts in {split} split")
    return prompts, {
        "source": str(path.resolve()),
        "fingerprint": fingerprint,
        "split": split,
        "rows": row_ids,
        "skipped": skipped,
        "count": len(prompts),
    }
