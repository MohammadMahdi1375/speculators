#!/usr/bin/env python3
"""Install end-of-epoch saving for pretrained DFlash-based ReTrace training.

Run with --root /path/to/vLLM_NPU_spec_main --apply. Without --apply, review only.
Checks known source fingerprints, preserves backups, and synchronizes the bundle.
The default --save-every 0 selects epoch boundaries. Positive values select steps.
Running trainers retain their original schedule until restarted.
"""

import argparse
import ast
import base64
import hashlib
import json
import os
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

UPDATES = {
    "src/speculators/models/retrace/train_pretrained.py": {
        "before": "43372fef49302937bd31d005f64bb0f1d3dca93172a0b5b1716ba7222d540cd0",
        "after": "07fb31bea36d71d034795398cec9ce6cbb11d104a080b24301d1edb49a2883cf",
        "payload": "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",
    },
    "src/speculators/models/retrace/pretrained_launch.py": {
        "before": "b888242ac266bb6b4d8452fb53267d8ee95803ca2a9e7b7766a78efdfceab202",
        "after": "df15f4c0f4cf5773258b26d0410c6959df3da9a7a8efe9ad059d1a9d53059084",
        "payload": "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",
    },
    "examples/train/train_retrace_dflash_pretrained_full.sh": {
        "before": "730981e6879bfb481848028b7a2b1dbeabf9b7a376b36554594ff780c1f27500",
        "after": "e03fa074f9bf9450abfbad833a63760101a296b6c0b4966ddf3badcb277cd5de",
        "payload": "eNqNVm2P2jgQ/s6vmFLoi4oJBPYFTj1ddmFvT8cCl4VWVVtFJnEgwrFT22G3V91/v3GyQLatVkUCifH4mRnPM4/9/JmTa+WsEuEwsYMV1Zvac5grZhRNBItgdMXRRlbdU3gDPlsoGjKIcYmYXCRiDbFUYDYMcs3USw1ZIuw2T4dMRKANDbdtRByxmObc6CH0O1vIlEwzo1tw0nWdbsftg5FbJtBQrmAeJtxAz23BObBMhhtcmvhwwsiJRbvcsHCbyURYQBtHxsAo7ih84RUhmu4YYTumvkLnNcLyXEOI0JyZRAon47RIM6OYtkWcSlBURDIlkaKxYQpiyvnKZg9+LgARU0wRK1fatGzBwlYMFKbj9xDnnBOFblGiWGik+tquaWaAMIkHkmHtCa/V/PHC9y7HgT+bLd7WG9+q/4fE2ciUOeK83+8P3IHrpOf93tngZHDqlC1wdpPJTTCdLwOdsTBIsT3/1Q+Y8w+L69m0ilpansINpYios+Mci45tCJdY1IIM2VezkdUAC8//c/wo7dLyVIDrfL1GisRIGSeVEeOakH/umCh/e6R/4WhBM72RRjvdMKYDeuZ2zgddt+ue9k8GbrfPzledfs/tnnZ6q14Un3bCSk4X3u04GF1NvNvramIV869mt8+HHOleieP5/ux9NUJh+FVsmTERZEzFyIwVR7a2v5TRVkRJzmVuSkNbsy+9zplbCXzpTac/0OVgfCoB6+QM2t12p9pD3/trGsyWi/nycScr9iFpVHnpYHpZbpxCEEIWlEQJsoNABI1XETUM3jQ/NNNmFDSvmzfN29cY1c4pZoHzuUuUFCkTBnSoEhx8oIqBkAa/ucBBwXGNcQy1zBWqS73xQ6kOLfSEGCn5NjEO7gkQt6039ad2CUG5Q83qsb8dzLzG7jOpDJRjMvcWlkKPKrdjlnOK06wdrcLGt6Pr8M2wcfyHtT4CW04vlldXY388etvdr8xu5sF0eRMsrv2xN7q1p/+daUi6R6B3dtaXSON3bnAzG40ngb+cTsf+206tlsTw8SM8A3JfqbmMXIfPn38rxKkG+MkUKmQML+fFMAO7x4IMRQ5CmmiN/BxCU38SL3+C8/sLt4Bg94kBtxYntp0TGVIOVhSxF8NSce+k2qJabiSPsK0QK/kvSqOhas1QJ3MbBRLseCJCxSwHEOHvdwgW4m7WhlGijUpWuUE1XisaJeiCQKjcdM3wBoBLPKfJeDHeXxqAowR5ZklnhdvjKNaCmmTHwDBtYZCZQttjHKJsl5mQh6TBqp0Vc6YwAhEZXgudVhdx0LMg9IPRbfVa/dZJ67R1hhcASyVS/IALKPKYC+asAWGSOAmpvVYwRRYn90zbxJZiK+SdwMI36GLz2p+xbagz8r2rhXO7GM9vHe/ycnkDO8pzdhwNVGedpyxq12zfftIikkKFou1SYNsPg9quTCinucBOfSoaSlB0EP0x1+uHxfK4Ksulyh8d8IXAHq6LildFco+u2CJa8SlE87haCks1UkWFvk/o0D9eUPCwWG0Z9rFVadvBKaX3ZM+dfgc/h5XSSlCT12Zj3yLHM2LYaFSm/VrxRtkvrrlcUb7fvX+qHJbL9wo+XPYGroqHy/EIsYitJshjgo+nO6oiOIJXHy4Pxnrjj3rtf0uk/oE=",
    },
    "tests/unit/models/test_retrace_pretrained.py": {
        "before": "c5b663da190c6c1f05207c3ece8a6ddae33e893c2b1e7253ed3f2728bdc82e3c",
        "after": "75fea8e4e1ff68c7ec9538c67798d1f516979615c8dcfa2e2786d5ae16eebdbe",
        "payload": "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",
    },
}
README_UPDATE = {
    "before": "98d1b4edc457e134a443ae729d7cd8f7ad502e17d4d6805ae9f8b93401bba308",
    "after": "256d874fe872a38a145baae93cc135c6ba3b6a99fb36ce236463d27f71743076",
    "payload": "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",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def under(base, relative):
    path = base
    for part in Path(relative).parts:
        if part in ("..", "/"):
            raise ValueError("Invalid relative path")
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink: {path}")
    return path


def atomic_write(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def reviewed_change(path, spec, changes):
    old = path.read_bytes()
    if digest(old) not in (spec["before"], spec["after"]):
        raise ValueError(f"Unrecognized local edits; no files changed: {path}")
    new = zlib.decompress(base64.b64decode(spec["payload"]))
    if digest(new) != spec["after"]:
        raise ValueError(f"Update payload checksum failed: {path}")
    if path.suffix == ".py":
        ast.parse(new.decode("utf-8"), filename=str(path))
    if old != new:
        changes.append((path, old, new, path.stat().st_mode & 0o777))


def prepare(root, bundle):
    changes = []
    for relative, spec in UPDATES.items():
        reviewed_change(under(root, "speculators/" + relative), spec, changes)
    if bundle.is_dir():
        for relative, spec in UPDATES.items():
            reviewed_change(
                under(bundle, "payload/speculators/" + relative), spec, changes
            )
            if relative.startswith("examples/train/"):
                reviewed_change(
                    under(bundle, "scripts/" + Path(relative).name), spec, changes
                )
        path = under(bundle, "manifest.json")
        old = path.read_bytes()
        manifest = json.loads(old)
        for relative, spec in UPDATES.items():
            matches = [
                entry
                for entry in manifest["repositories"]["speculators"]["files"]
                if entry["path"] == relative
            ]
            if len(matches) != 1 or matches[0]["after_sha256"] not in (
                spec["before"],
                spec["after"],
            ):
                raise ValueError(f"Unrecognized bundle fingerprint: {relative}")
            entry = matches[0]
            entry["after_sha256"] = spec["after"]
            if spec["before"] not in entry["accepted_before_sha256"]:
                entry["accepted_before_sha256"].append(spec["before"])
        new = (json.dumps(manifest, indent=2) + "\n").encode()
        if new != old:
            changes.append((path, old, new, path.stat().st_mode & 0o777))
        reviewed_change(under(bundle, "README.md"), README_UPDATE, changes)
    return changes


def apply(root, changes):
    if not changes:
        print("End-of-epoch saving is already installed.")
        return
    for path, old, _, _ in changes:
        if path.read_bytes() != old:
            raise ValueError(f"File changed during review: {path}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = root / "retrace-epoch-checkpoint-backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    receipts = []
    for index, (path, old, new, mode) in enumerate(changes):
        atomic_write(backup / f"{index:02d}.before", old, mode)
        receipts.append(
            {
                "path": str(path),
                "backup": f"{index:02d}.before",
                "before_sha256": digest(old),
                "after_sha256": digest(new),
                "mode": mode,
            }
        )
    atomic_write(
        backup / "receipt.json", (json.dumps(receipts, indent=2) + "\n").encode(), 0o644
    )
    done = []
    try:
        for path, old, new, mode in changes:
            if path.read_bytes() != old:
                raise ValueError(f"File changed during installation: {path}")
            done.append((path, old, mode))
            atomic_write(path, new, mode)
        for path, _, new, _ in changes:
            if path.read_bytes() != new:
                raise RuntimeError(f"Post-write verification failed: {path}")
    except BaseException:
        for path, old, mode in reversed(done):
            atomic_write(path, old, mode)
        raise
    print(f"End-of-epoch saving installed. Backup: {backup}")
    print("Launch examples/train/train_retrace_dflash_pretrained_full.sh as usual.")
    print("Running trainers retain their original save schedule until restarted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--bundle", help="Default: ROOT/retrace-dflash-pretrained, when present"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    bundle = (
        Path(args.bundle).absolute()
        if args.bundle
        else root / "retrace-dflash-pretrained"
    )
    if bundle.is_symlink():
        parser.error("Bundle directory is a symlink")
    if args.bundle and not bundle.is_dir():
        parser.error("The specified bundle directory does not exist")
    changes = prepare(root, bundle)
    print(f"Reviewed {len(changes)} file updates")
    for path, _, _, _ in changes:
        print(f"  update {path}")
    if args.apply:
        apply(root, changes)
    else:
        print("Dry run only. Add --apply to install the reviewed update.")


if __name__ == "__main__":
    main()
