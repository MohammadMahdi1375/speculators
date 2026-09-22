import json

import pytest
from datasets import Dataset

from speculators.models.retrace.data import read_prompts


def test_arrow_suffix_and_split_match_jsonl(tmp_path):
    prefixes = [[1, 2, i, j] for i in range(3, 15) for j in range(3, 15)]
    arrow = tmp_path / "data"
    Dataset.from_dict(
        {
            "input_ids": [p + [20, 21, 22, 23] for p in prefixes],
            "loss_mask": [[False] * 4 + [True, True, False, False]] * len(prefixes),
            "seq_len": [8] * len(prefixes),
        }
    ).save_to_disk(arrow)
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text("".join(json.dumps({"input_ids": p}) + "\n" for p in prefixes))
    splits = {}
    for split in ("train", "validation"):
        a, _ = read_prompts(arrow, 32, 0, 16, split)
        b, _ = read_prompts(jsonl, 32, 0, 16, split)
        assert a == b
        splits[split] = set(map(tuple, a))
    assert not splits["train"] & splits["validation"]
    assert len(splits["train"] | splits["validation"]) == len(prefixes)


def test_multiple_loss_spans_stop_at_first_response_and_keep_assistant_header(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "input_ids": [1, 2, 3, 4, 5, 6],
                "loss_mask": [0, 0, 1, 0, 1, 0],
                "seq_len": 6,
            }
        )
    )
    assert read_prompts(path, 32, split="all")[0] == [[1, 2]]


@pytest.mark.parametrize("ids", [[True], [-1], [32], [1.0]])
def test_bad_tokens_rejected(tmp_path, ids):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"input_ids": ids}))
    with pytest.raises(ValueError, match="invalid token"):
        read_prompts(path, 32, split="all")
