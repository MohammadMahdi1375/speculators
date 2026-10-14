#!/usr/bin/env python3
"""Collect an EXISTING target-graph failure into one ZIP below 900,000 bytes.

Standard library only. Reads saved logs and launch records; never imports Torch,
contacts a server, starts/stops a process, or modifies model/checkpoint files.
Usage: python collect_target_graph_failure.py --run-dir output/target_graph_check_...
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile


LIMIT = 900_000
ERROR = re.compile(rb"Traceback \(most recent call last\)|\bERROR\b|\b(?:Runtime|Value|Assertion|Type|Attribute|NotImplemented)Error\b|COMPILED_PREFIX_PARITY_FAIL")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def log_excerpt(data, limit):
    if len(data) <= limit:
        return data, False
    # Preserve startup, the first error contexts, and shutdown separately.
    # Line/byte offsets refer to the original file, not this excerpt.
    contexts = []
    for match in ERROR.finditer(data):
        start, end = max(0, match.start() - 4000), min(len(data), match.end() + 22000)
        if contexts and start <= contexts[-1][1]:
            continue
        contexts.append((start, end))
        if len(contexts) == 6:
            break
    head = data[:min(48000, limit // 8)]
    pieces = [b"[TRUNCATED LOG: START]\n" + head]
    context_budget = limit // 2
    for start, end in contexts:
        part = f"\n[ORIGINAL BYTES {start}:{end}]\n".encode() + data[start:end]
        if len(part) > context_budget:
            part = part[:context_budget]
        pieces.append(part)
        context_budget -= len(part)
        if context_budget == 0:
            break
    tail_budget = limit - sum(map(len, pieces)) - 100
    pieces.append(b"\n[TRUNCATED LOG: END]\n" + data[-tail_budget:])
    return b"".join(pieces), True


def json_bytes(document):
    return (json.dumps(document, indent=2, ensure_ascii=True) + "\n").encode()


def collect(run_dir, output=None):
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    if not run_dir.is_dir() or not (run_dir / "graph.log").is_file():
        raise ValueError("Use the existing target_graph_check directory containing graph.log")
    report = {
        "collector_version": 1, "diagnostic_only": True, "run_dir": str(run_dir),
        "note": "Existing files only; no new server, request, evaluation, or parity claim.",
        "files": {}, "missing": [], "errors": [],
    }
    entries = {}
    for name, limit in (("graph.log", 500000), ("eager.log", 64000)):
        path = run_dir / name
        if not path.is_file():
            report["missing"].append(name)
            continue
        raw = path.read_bytes()
        contents, truncated = log_excerpt(raw, limit)
        entries[name] = contents
        report["files"][name] = {
            "original_bytes": len(raw), "original_sha256": sha(raw),
            "included_bytes": len(contents), "truncated": truncated,
        }
    for name in ("result.json", "graph/launch.json", "eager/launch.json"):
        path = run_dir / name
        if not path.is_file():
            report["missing"].append(name)
            continue
        raw = path.read_bytes()
        # Launch records contain settings and source hashes, never model weights.
        if len(raw) > 32000:
            report["errors"].append(f"{name}: exceeds 32000 bytes; omitted")
            continue
        entries[name] = raw
        report["files"][name] = {"original_bytes": len(raw), "original_sha256": sha(raw)}
    for phase in ("eager", "graph"):
        name = f"{phase}_probe.json"
        path = run_dir / name
        if not path.is_file():
            report["missing"].append(name)
            continue
        try:
            probe = json.loads(path.read_text())
            summary = {key: probe[key] for key in (
                "complete", "diagnostic_only", "error", "stage", "comparison",
                "manifest_sha256", "tokens", "warmup_rounds",
            ) if key in probe}
            summary["completed_prompt_count"] = len(probe.get("rows", []))
            # Retain evidence of completion without collecting generated text/IDs.
            raw = json_bytes(summary)
            if len(raw) > 32000:
                raise ValueError("probe summary exceeds 32000 bytes")
            entries[f"{phase}_probe_summary.json"] = raw
        except (ValueError, TypeError, OSError) as exc:
            report["errors"].append(f"{name}: {exc}")
    entries["collection.json"] = json_bytes(report)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(entries.items()):
            archive.writestr(name, raw)
    payload = buffer.getvalue()
    if len(payload) >= LIMIT:
        raise ValueError("Collection exceeds 900000 bytes; no archive was written")
    output = Path(output).expanduser().resolve() if output else (
        run_dir.parent / f"{run_dir.name}_failure_{time.time_ns()}.zip"
    )
    with output.open("xb") as stream:
        stream.write(payload)
    print(f"Existing logs collected; no experiment was rerun.\nUpload: {output}")
    print(f"Size: {len(payload)} bytes (below the 1 MB upload limit)")
    if "graph_probe.json" in report["missing"]:
        print("graph_probe.json is absent; graph.log and launch records are included.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        collect(args.run_dir, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Collection failed: {exc}\n")


if __name__ == "__main__":
    main()
