#!/usr/bin/env python3
"""Warm up parity checks and record deterministic HTTP probes (not a benchmark)."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

PROMPTS = [
    "Explain how to add the integers from 1 to 100 and calculate their sum.",
    "Write a Python function that returns the first n Fibonacci numbers and explain it.",
    "A shop has 18 boxes with 24 pencils each. It sells 137 pencils. How many remain?",
]


def completion(base_url, model, prompt, max_tokens, ignore_eos=False):
    body = {"model": model, "prompt": prompt, "temperature": 0,
            "top_p": 1.0, "top_k": -1, "max_tokens": max_tokens,
            "seed": 42, "ignore_eos": ignore_eos}
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    request = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')[:4000]}") from error
    if not result.get("choices"):
        raise RuntimeError(f"Completion failed: {result}")
    return {"prompt": prompt, "text": result["choices"][0].get("text"),
            "finish_reason": result["choices"][0].get("finish_reason"), "usage": result.get("usage")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8211")
    parser.add_argument("--model", default="qwen3-4b-dflash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Choose a fresh output path: {args.output}")
    print("Warmup: generating 512 tokens to exercise the initial 32 selector checks...", flush=True)
    warmup = completion(args.base_url, args.model,
                        "Write an extended tutorial about elementary number theory with examples.", 512, True)
    print("Warmup completed. In fast mode, confirm PREFIX_PARITY_PASS in the SERVER log.", flush=True)
    records = []
    for index, prompt in enumerate(PROMPTS):
        print(f"HTTP output probe {index + 1}/{len(PROMPTS)}...", flush=True)
        records.append(completion(args.base_url, args.model, prompt, 128))
    report = {"base_url": args.base_url, "model": args.model, "timestamp": time.time(),
              "warmup_usage": warmup.get("usage"), "records": records,
              "note": "Output probes only. Inspect server selector-parity log. This is not a speed benchmark."}
    passed = True
    if args.compare:
        previous = json.loads(args.compare.read_text())
        old_records = previous.get("records", [])
        passed = len(old_records) == len(records) and all(
            old["prompt"] == new["prompt"] and old["text"] == new["text"]
            and old["finish_reason"] == new["finish_reason"]
            for old, new in zip(old_records, records))
        report["compared_with"] = str(args.compare)
        report["exact_probe_text_match"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved probes: {args.output}")
    if not passed:
        raise SystemExit("HTTP output probes differ. Inspect server parity logs and model paths before benchmarking.")
    if args.compare:
        print("All three HTTP output probes match the reference run.")


if __name__ == "__main__":
    main()
