#!/usr/bin/env python3
"""Select the DSpark job's C++ runtime before importing Torch or torch_npu.

This bootstrap uses only the Python standard library. Runtime libraries are
selected for child processes; no installed library or other process is changed.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

REQUIRED_ABI = b"CXXABI_1.3.15\0"

IMPORT_PROBE = r"""
import os
from pathlib import Path
import sqlite3
import torch
import torch_npu

chosen = Path(os.environ["RETRACE_DSPARK_CXX_RUNTIME"]).resolve()
mapped = set()
for line in Path("/proc/self/maps").read_text().splitlines():
    fields = line.split(None, 5)
    if len(fields) == 6 and "libstdc++.so" in fields[5]:
        mapped.add(Path(fields[5].removesuffix(" (deleted)")).resolve())
if chosen not in mapped:
    raise RuntimeError(f"Selected C++ runtime was not loaded: {chosen}; mapped={mapped}")
if any(path != chosen for path in mapped):
    raise RuntimeError(f"Multiple C++ runtimes were loaded: {sorted(map(str, mapped))}")
print(f"Runtime imports passed: sqlite3={sqlite3.sqlite_version}, "
      f"torch={torch.__version__}, torch_npu={getattr(torch_npu, '__version__', 'loaded')}", flush=True)
print(f"Loaded C++ runtime: {chosen}", flush=True)
print("Requested NPU visibility: " + os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"), flush=True)
"""


def runtime_environment(prefix, inherited):
    """Build a child environment, preserving CANN's search paths and other preloads."""
    env = dict(inherited)
    candidate = Path(
        env.get("RETRACE_DSPARK_CXX_RUNTIME") or Path(prefix) / "lib/libstdc++.so.6"
    ).expanduser()
    try:
        library = candidate.resolve(strict=True)
        body = library.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Cannot read the environment's C++ runtime: {candidate}. "
            "Set RETRACE_DSPARK_CXX_RUNTIME to a compatible local libstdc++.so.6 "
            "providing CXXABI_1.3.15. No packages have been changed."
        ) from exc
    if not body.startswith(b"\x7fELF") or REQUIRED_ABI not in body:
        raise ValueError(
            f"{library} is not an ELF C++ runtime containing CXXABI_1.3.15. "
            "The environment also needs a newer compatible C++ runtime; selecting "
            "this file cannot solve the reported ICU import failure. "
            "Set RETRACE_DSPARK_CXX_RUNTIME to a compatible local libstdc++.so.6. "
            "No packages have been changed."
        )
    # Check the marker cheaply; the child import probe verifies actual linkage.
    selected = []
    gcc = candidate.parent / "libgcc_s.so.1"
    if gcc.is_file():
        selected.append(str(gcc.resolve()))
    selected.append(str(library))
    if any(re.search(r"[:\s]", path) for path in selected):
        raise ValueError(
            "LD_PRELOAD cannot represent runtime paths with spaces or colons"
        )
    retained = []
    for item in re.split(r"[:\s]+", env.get("LD_PRELOAD", "")):
        if not item:
            continue
        # Avoid explicitly preloading the old C++/GCC runtime alongside the new one.
        if Path(item).name.startswith(("libstdc++.so", "libgcc_s.so")):
            continue
        if item not in retained:
            retained.append(item)
    env["LD_PRELOAD"] = ":".join(selected + retained)
    env["RETRACE_DSPARK_CXX_RUNTIME"] = str(library)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.check_only and not command:
        parser.error("Pass a training command after --, or use --check-only")
    if args.check_only and command:
        parser.error("--check-only does not take a training command")
    try:
        env = runtime_environment(sys.prefix, os.environ)
    except ValueError as exc:
        parser.exit(2, f"C++ runtime check failed: {exc}\n")
    print(
        "Checking SQLite, Torch and torch_npu imports with the selected runtime",
        flush=True,
    )
    probe = subprocess.run([sys.executable, "-c", IMPORT_PROBE], env=env, check=False)
    if probe.returncode:
        parser.exit(
            1, "Runtime import check failed; DSpark training was not started.\n"
        )
    if not args.check_only:
        os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
