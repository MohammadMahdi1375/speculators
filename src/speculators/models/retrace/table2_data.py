"""Freeze seven benchmark prompt sets for paired AR/DFlash/ReTrace evaluation.

The paper does not publish its exact row IDs, LCB release or every prompt
template. These are explicit, versioned choices, not a claim of exact recovery.
Local JSONL overrides accept {"id": ..., "prompt": "..."} per row.
"""

import argparse
import json
import random
from pathlib import Path

from .pretrained import sha256
from .prompt_pool import prompt_key

COUNTS = {
    "gsm8k": 128,
    "math500": 128,
    "humaneval": 164,
    "lcb": 128,
    "aime25": 30,
    "mtbench": 80,
    "alpaca": 128,
}
DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "default", "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "aime25": ("MathArena/aime_2025", "default", "train"),
    "mtbench": ("HuggingFaceH4/mt_bench_prompts", "default", "train"),
    "alpaca": ("tatsu-lab/alpaca", "default", "train"),
}


def format_prompt(name, row):
    if name in ("gsm8k", "math500", "aime25"):
        question = row["question"] if name == "gsm8k" else row["problem"]
        return (
            question
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    if name == "humaneval":
        return (
            "Write a solution to the following problem and make sure that it passes the tests:\n```python\n"
            + row["prompt"]
            + "\n```"
        )
    if name == "mtbench":
        # First user turn, matching the public DFlash OpenAI benchmark path.
        # Keep this choice visible: the ReTrace paper does not specify turn policy.
        return row["prompt"][0]
    if name == "alpaca":
        return row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
    if name == "lcb":
        starter = row.get("starter_code", "")
        instruction = (
            "Complete the following Python starter code."
            if starter
            else "Write a Python program that reads from standard input and writes to standard output."
        )
        return (
            row["question_content"]
            + "\n\n"
            + instruction
            + ("\n```python\n" + starter + "\n```" if starter else "")
            + "\nReturn your solution in a Python code block."
        )
    raise ValueError(name)


def remote_rows(name, lcb_release):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    if name == "lcb":
        repo = "livecodebench/code_generation_lite"
        revision = HfApi().dataset_info(repo).sha
        number = int(lcb_release.removeprefix("release_v"))
        files = ["test.jsonl", *[f"test{i}.jsonl" for i in range(2, number + 1)]]
        rows, fingerprints = [], {}
        for filename in files:
            print(
                f"Downloading LCB {filename}; source files include tests and can be large",
                flush=True,
            )
            path = hf_hub_download(
                repo, filename, repo_type="dataset", revision=revision
            )
            fingerprints[filename] = sha256(path)
            # Only retain prompt fields. Never unpickle or execute benchmark code.
            with Path(path).open() as handle:
                for line in handle:
                    row = json.loads(line)
                    rows.append(
                        {
                            k: row.get(k, "")
                            for k in (
                                "question_content",
                                "starter_code",
                                "question_id",
                                "contest_date",
                            )
                        }
                    )
        return rows, {
            "repo": repo,
            "revision": revision,
            "release": lcb_release,
            "source_sha256": fingerprints,
        }
    repo, config, split = DATASETS[name]
    revision = HfApi().dataset_info(repo).sha
    data = load_dataset(repo, config, split=split, revision=revision)
    return data, {
        "repo": repo,
        "revision": revision,
        "config": config,
        "split": split,
        "fingerprint": data._fingerprint,
    }


def make_manifest(
    target, output, names, *, sources=None, lcb_release="release_v5", seed=42, limit=0
):
    from transformers import AutoTokenizer

    output = Path(output)
    if output.exists():
        raise ValueError("Use a new evaluation manifest filename")
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    result = {
        "format": 1,
        "target": str(Path(target).absolute()),
        "seed": seed,
        "thinking": False,
        "datasets": {},
        "author_row_ids_available": False,
        "choices": {
            "lcb_release": lcb_release,
            "mtbench": "all 80 first turns",
            "templates": "table2_data.py; override with --source NAME=JSONL",
        },
        "template_sha256": sha256(__file__),
    }
    sources = sources or {}
    for name in names:
        if name in sources:
            path = Path(sources[name])
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            info = {"local_jsonl": str(path.absolute()), "sha256": sha256(path)}
        else:
            rows, info = remote_rows(name, lcb_release)
        desired = min(COUNTS[name], limit) if limit else COUNTS[name]
        if len(rows) < desired:
            raise ValueError(
                f"{name} has {len(rows)} rows; expected at least {desired}"
            )
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        chosen = indices[:desired]
        prompts = []
        for index in chosen:
            row = rows[index]
            text = row["prompt"] if name in sources else format_prompt(name, row)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{name} row {index}: expected a nonempty prompt string"
                )
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
            return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not ids:
                raise ValueError("Empty tokenized evaluation prompt")
            prompts.append(
                {
                    "source_row": index,
                    "source_id": str(
                        row.get("id", row.get("task_id", row.get("question_id", index)))
                    ),
                    "prompt_token_ids": ids,
                    "sha256": prompt_key(ids),
                }
            )
        result["datasets"][name] = {
            "source": info,
            "source_rows": len(rows),
            "requests": len(prompts),
            "results": prompts,
        }
        print(f"{name}: frozen {len(prompts)} prompts", flush=True)
    result["complete_table2_counts"] = set(names) == set(COUNTS) and all(
        result["datasets"][name]["requests"] == COUNTS[name] for name in names
    )
    result["tokenizer_sha256"] = {
        p.name: sha256(p) for p in Path(target).glob("*token*json")
    }
    result["target_config_sha256"] = sha256(Path(target) / "config.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Frozen evaluation manifest: {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(COUNTS), default=list(COUNTS)
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="NAME=/path/to/prompts.jsonl; raw user prompt strings, before chat templating",
    )
    parser.add_argument(
        "--lcb-release",
        choices=[f"release_v{i}" for i in range(1, 7)],
        default="release_v5",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=0, help="Smoke subset; 0 uses the paper's counts"
    )
    args = parser.parse_args()
    sources = {}
    for entry in args.source:
        name, separator, path = entry.partition("=")
        if not separator or name not in COUNTS or name in sources:
            parser.error("Use one NAME=JSONL override per benchmark")
        sources[name] = path
    if args.limit < 0 or len(set(args.datasets)) != len(args.datasets):
        parser.error("Invalid limit or duplicate dataset")
    make_manifest(
        args.target,
        args.output,
        args.datasets,
        sources=sources,
        lcb_release=args.lcb_release,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
