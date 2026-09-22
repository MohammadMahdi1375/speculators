"""Select full stored prompt/response rows; never generate replacement responses."""

import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset, Features, List, Value, load_from_disk

from .prompt_pool import prompt_key


def stored_row(row, vocabulary, prompt_limit, response_limit, minimum_response):
    ids, mask = list(row["input_ids"]), list(row["loss_mask"])
    length = int(row.get("seq_len", len(ids)))
    if (
        len(ids) != len(mask)
        or not 0 < length <= len(ids)
        or any(type(t) is not int or not 0 <= t < vocabulary for t in ids[:length])
        or any(v not in (0, 1, False, True) for v in mask[:length])
    ):
        raise ValueError("Invalid stored token IDs, mask, or seq_len")
    start = next((i for i, v in enumerate(mask[:length]) if v), None)
    if start is None or not 0 < start <= prompt_limit:
        return None
    # Use the first contiguous response only. Do not join assistant turns,
    # supervise masked terminators, or carry memory across document boundaries.
    end = next((i for i in range(start, length) if not mask[i]), length)
    end = min(end, start + response_limit)
    if end - start < minimum_response:
        return None
    return {
        "input_ids": ids[:end],
        "loss_mask": [bool(x) for x in mask[:end]],
        "seq_len": end,
    }, start


def prepare(
    source,
    output,
    *,
    count,
    vocabulary,
    block_size,
    prompt_limit=512,
    response_limit=1024,
    seed=42,
    prompt_pool=None,
):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Use a new stored dataset directory: {output}")
    dataset = load_from_disk(source)
    if not isinstance(dataset, Dataset):
        raise ValueError("Expected one Arrow Dataset, not DatasetDict")
    dataset = dataset.with_format(None)
    pool = None
    if prompt_pool:
        with Path(prompt_pool).open() as handle:
            pool = [json.loads(line) for line in handle if line.strip()]
        indices = [int(x["source_row"]) for x in pool]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate source rows in prompt pool")
        pool = dict(zip(indices, pool, strict=True))
    else:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
    rows, source_rows, seen = [], [], set()
    skipped = 0
    for index in indices:
        raw = dataset[index]
        if pool is not None:
            # Legacy pool may include an empty-think prefix taken from the
            # response. Only reuse it if it is already in the stored sequence.
            prefix = pool[index]["input_ids"]
            if prompt_key(prefix) != pool[index]["sha256"]:
                raise ValueError(f"Prompt pool checksum mismatch at row {index}")
            if raw["input_ids"][: len(prefix)] != prefix:
                raise ValueError(f"Stored sequence does not match pool row {index}")
            raw = dict(raw)
            raw["loss_mask"] = [False] * len(prefix) + raw["loss_mask"][len(prefix) :]
        selected = stored_row(
            raw,
            vocabulary,
            prompt_limit,
            response_limit,
            minimum_response=block_size + 1,
        )
        if selected is None:
            skipped += 1
            continue
        row, boundary = selected
        key = prompt_key(row["input_ids"][:boundary])
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        rows.append(row)
        source_rows.append(index)
        if len(rows) % 1000 == 0:
            print(f"Selected {len(rows):,}/{count:,} stored responses", flush=True)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(
            f"Only {len(rows)} eligible stored rows for requested {count}; "
            "reduce --max-records or omit --prompt-pool"
        )
    features = Features(
        {
            "input_ids": List(Value("int32")),
            "loss_mask": List(Value("bool")),
            "seq_len": Value("int64"),
        }
    )
    selected = Dataset.from_list(rows, features=features)
    # ArrowDataset's stock text path expects tensors; persist the same format
    # metadata as the normal DSpark preprocessing pipeline.
    selected.set_format("torch")
    selected.save_to_disk(str(output))
    record_hash = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "source": str(Path(source).resolve()),
        "source_fingerprint": dataset._fingerprint,
        "selected_source_rows": source_rows,
        "records": len(rows),
        "skipped": skipped,
        "content_sha256": record_hash,
        "total_tokens": sum(r["seq_len"] for r in rows),
        "seed": seed,
        "prompt_limit": prompt_limit,
        "response_limit": response_limit,
        "prompt_pool": prompt_pool,
        "tokens_preserved": True,
        "selection": "first response span; unique prompts; bounded lengths",
        "author_prompt_ids_available": False,
    }
    (output / "stored_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
