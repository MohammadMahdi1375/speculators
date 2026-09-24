#!/usr/bin/env python3
"""Compare greedy output text from target-only and prefix-serving endpoints.

This is a plumbing/quality smoke test, not a paper-quality latency benchmark.
Use identical model, tokenizer, backend, dtype and request settings on servers.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def request(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="http://127.0.0.1:8093")
    parser.add_argument("--candidate", default="http://127.0.0.1:8094")
    parser.add_argument("--output", type=Path, default=Path("prefix_output_smoke.json"))
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    prompts = [
        "Write a Python function that returns the first n Fibonacci numbers.\n",
        "Solve step by step: A train travels 180 km in 2.5 hours. What is its average speed?\n",
        "Explain why the sky appears blue in three concise sentences.\n",
        "List five practical ways to reduce memory use in a Python data pipeline.\n",
    ]
    bases = (args.baseline, args.candidate)
    model_ids = [request(base, "/v1/models")["data"][0]["id"] for base in bases]
    rows = []
    for prompt in prompts:
        row = {"prompt": prompt, "outputs": []}
        for base, model in zip(bases, model_ids):
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": 0,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "seed": 42,
            }
            started = time.perf_counter()
            result = request(base, "/v1/completions", payload)
            row["outputs"].append(
                {
                    "endpoint": base,
                    "text": result["choices"][0]["text"],
                    "seconds_including_http": time.perf_counter() - started,
                    "usage": result.get("usage"),
                }
            )
        row["text_equal"] = row["outputs"][0]["text"] == row["outputs"][1]["text"]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        f"Exact text match: {sum(r['text_equal'] for r in rows)}/{len(rows)}; report: {args.output}"
    )
    if not all(row["text_equal"] for row in rows):
        raise SystemExit(
            "Output differs. Investigate settings/numerics/verification before benchmarking."
        )


if __name__ == "__main__":
    main()
