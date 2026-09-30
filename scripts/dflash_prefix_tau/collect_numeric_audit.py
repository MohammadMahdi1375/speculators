#!/usr/bin/env python3
"""Package bounded numeric-audit reports and server log. No torch/NPU imports."""
import argparse
import json
from pathlib import Path
import re
import time
import zipfile


ALLOWED = re.compile(r"(?:launch\.json|server\.log|summary_[0-9]+\.json|event_[0-9]+_[0-9]+\.json|profile_failure_[0-9]+\.json)\Z")


def collect(root, audit_dir=None):
    root = Path(root).resolve(strict=True)
    if audit_dir is None:
        candidates = [p for p in (root / "output").glob("prefix_numeric_audit_*")
                      if p.is_dir() and not p.is_symlink() and (p / "launch.json").is_file()]
        if not candidates:
            raise ValueError("No numeric audit directory found; pass --audit-dir explicitly")
        audit_dir = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    audit_dir = Path(audit_dir)
    if audit_dir.is_symlink():
        raise ValueError("Refusing a symlink audit directory")
    audit_dir = audit_dir.resolve(strict=True)
    files = sorted(p for p in audit_dir.iterdir() if ALLOWED.fullmatch(p.name))
    if not files:
        raise ValueError("Audit directory contains no recognized reports")
    snapshots = {}
    summaries = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular report file: {path}")
        if path.stat().st_size > 128 * 1024**2:
            raise ValueError(f"Report exceeds 128 MiB: {path}; collect the traceback separately")
        data = path.read_bytes()
        snapshots[path.name] = data
        if path.name.startswith("summary_"):
            summaries.append(json.loads(data))
    summary = {
        "audit_dir": str(audit_dir), "workers": len(summaries),
        "observed_blocks": sum(s["observed_blocks"] for s in summaries),
        "finite_token_equal_blocks": sum(s["finite_token_equal_blocks"] for s in summaries),
        "score_drift_blocks": sum(s["score_drift_blocks"] for s in summaries),
        "token_mismatch_blocks": sum(s["token_mismatch_blocks"] for s in summaries),
        "nonfinite_blocks": sum(s["nonfinite_blocks"] for s in summaries),
        "profiling_token_mismatch": any(s.get("profiling_token_mismatch", False) for s in summaries),
        "diagnostic_errors": sum(s["diagnostic_errors"] for s in summaries),
        "max_abs_score_error": max((s["max_abs_score_error"] for s in summaries), default=None),
        "note": "Observed inputs only. Confirm evaluator completion separately. Audit throughput must not be reported as optimized serving performance.",
    }
    output = root / "output" / f"prefix_numeric_audit_report_{time.time_ns()}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in snapshots.items():
            archive.writestr("prefix_numeric_audit_report/" + name, data)
        archive.writestr("prefix_numeric_audit_report/collection.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Upload this ZIP: {output}")
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-main", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()
    try:
        collect(args.spec_main, args.audit_dir)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
