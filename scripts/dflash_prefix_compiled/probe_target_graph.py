#!/usr/bin/env python3
"""Compare eager/graph outputs for one checkpoint without timing the requests.

Use an already running, idle server. No server is started or stopped here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(opener, url, payload=None):
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=600) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read(8192).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data


def prompts(manifest, launch, count):
    document = json.loads(manifest.read_text())
    if document.get("thinking") is not False:
        raise ValueError("Use the frozen no-thinking evaluation manifest")
    if document.get("target_config_sha256") != launch["target_config_sha256"]:
        raise ValueError("Manifest target config does not match launch.json")
    target = Path(launch["target"])
    for name, expected in document.get("tokenizer_sha256", {}).items():
        if Path(name).name != name or sha((target / name).read_bytes()) != expected:
            raise ValueError("Manifest tokenizer mismatch")
    rows = document["datasets"]["gsm8k"]["results"]
    if len(rows) < count:
        raise ValueError("Manifest contains fewer requested GSM8K prompts")
    result = []
    for row in rows[:count]:
        ids = row["prompt_token_ids"]
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Invalid prompt token IDs")
        result.append(ids)
    return result


def validate_response(data, proposals):
    choice = data["choices"][0]
    ids = choice.get("token_ids")
    if (not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids)
            or len(ids) != data["usage"]["completion_tokens"]):
        raise ValueError("Server did not return complete output token IDs")
    spec = data.get("metrics", {}).get("speculative_decoding", {})
    counts = {}
    for key in ("num_spec_steps", "num_accepted_draft_tokens", "num_draft_tokens", "num_spec_tokens"):
        value = spec.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Missing/invalid speculative counter: {key}")
        counts[key] = value
    if counts["num_spec_steps"] < 1 or counts["num_spec_tokens"] != proposals:
        raise ValueError("Server proposal count does not match launch.json")
    if counts["num_accepted_draft_tokens"] > counts["num_draft_tokens"]:
        raise ValueError("Accepted counter exceeds drafted counter")
    return {"output_token_ids": ids, "finish_reason": choice["finish_reason"], "speculative": counts}


def compare(reference, current):
    keys = ("draft", "target", "draft_config_sha256", "target_config_sha256", "source_sha256",
            "selector_environment", "block_size", "proposals", "npu", "additional_config",
            "speculative_config", "max_model_len", "max_batched_tokens", "memory_utilization")
    issues = []
    if not reference.get("complete") or not current.get("complete"):
        issues.append("incomplete probe")
    if any(reference["launch"].get(k) != current["launch"].get(k) for k in keys):
        issues.append("checkpoint/runtime/device/selector configuration differs")
    if reference["manifest_sha256"] != current["manifest_sha256"] or reference["tokens"] != current["tokens"]:
        issues.append("manifest or generation limit differs")
    old, new = reference["rows"], current["rows"]
    if len(old) != len(new) or [r["prompt_sha256"] for r in old] != [r["prompt_sha256"] for r in new]:
        issues.append("ordered prompts differ")
    token_mismatches, counter_mismatches = [], []
    if not issues:
        for i, (a, b) in enumerate(zip(old, new)):
            if a["output_token_ids"] != b["output_token_ids"] or a["finish_reason"] != b["finish_reason"]:
                token_mismatches.append(i)
            if a["speculative"] != b["speculative"]:
                counter_mismatches.append(i)
    return {
        "comparable": not issues, "issues": issues,
        "output_tokens_equal": not issues and not token_mismatches,
        "acceptance_counters_equal": not issues and not counter_mismatches,
        "token_mismatch_indices": token_mismatches, "counter_mismatch_indices": counter_mismatches,
        "pass": not issues and not token_mismatches and not counter_mismatches,
        "scope": "These observed prompts only; not a throughput benchmark or universal parity proof.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.num_prompts <= 128 or not 64 <= args.tokens <= 2048:
        parser.error("Use 1..128 prompts and 64..2048 tokens")
    if args.output.exists():
        parser.error("Output already exists; use a fresh path")
    launch = json.loads(args.launch_record.read_text())
    if launch["check_only"]:
        parser.error("launch.json describes CHECK_ONLY; use the running server record")
    selected = prompts(args.manifest, launch, args.num_prompts)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{launch['port']}"
    params = {"model": launch["served_model_name"], "temperature": 0, "top_p": 1.0,
              "top_k": -1, "seed": 42, "stream": False, "return_token_ids": True}
    record = {"complete": False, "launch": launch, "manifest_sha256": sha(args.manifest.read_bytes()),
              "tokens": args.tokens, "rows": [], "diagnostic_only": True,
              "stage": "models", "warmup_rounds": 0}
    error = None
    try:
        models = read_json(opener, url + "/v1/models")
        if launch["served_model_name"] not in [row["id"] for row in models["data"]]:
            raise ValueError("Unexpected served model name")
        record["stage"] = "warmup"
        for attempt in range(3):
            record["warmup_request"] = attempt + 1
            data = read_json(opener, url + "/v1/completions", dict(
                params, prompt=selected[0], max_tokens=512, min_tokens=512, ignore_eos=True,
            ))
            row = validate_response(data, launch["proposals"])
            record["warmup_rounds"] += row["speculative"]["num_spec_steps"]
            if record["warmup_rounds"] >= 32:
                break
        if record["warmup_rounds"] < 32:
            raise ValueError("Warmup did not complete 32 verification rounds")
        print(f"Warmup: {record['warmup_rounds']} verification rounds; "
              "inspect server selector parity and graph dispatch logs.", flush=True)
        record["stage"] = "prompts"
        for i, ids in enumerate(selected):
            record["prompt_index_zero_based"] = i
            print(f"Checking GSM8K prompt {i + 1}/{len(selected)}", flush=True)
            data = read_json(opener, url + "/v1/completions", dict(params, prompt=ids, max_tokens=args.tokens))
            row = validate_response(data, launch["proposals"])
            row["prompt_sha256"] = sha(json.dumps(ids).encode())
            record["rows"].append(row)
        record["complete"] = True
        record["stage"] = "completed"
        if args.reference:
            record["comparison"] = compare(json.loads(args.reference.read_text()), record)
            print(json.dumps(record["comparison"], indent=2), flush=True)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        error = exc
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(f"Probe report: {args.output}", flush=True)
    if error:
        raise error
    if args.reference and not record["comparison"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
