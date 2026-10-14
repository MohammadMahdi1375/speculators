"""Fixed validation inputs and BF16 export precision for the bounded pilot."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import torch


@contextmanager
def exported_precision(model, dtype=torch.bfloat16):
    """Restore the same Parameter objects AND master storages, even on failure.

Run only between optimization steps with no pending autograd graph. Includes
the frozen teacher so a tiny FP32 fixture also represents BF16 export behavior.
"""
    saved = [(p, p.data) for p in model.parameters() if p.is_floating_point() and p.dtype != dtype]
    try:
        for parameter, original in saved:
            parameter.data = original.to(dtype=dtype)
        yield
    finally:
        for parameter, original in saved:
            parameter.data = original


class _Sampler:
    def set_epoch(self, epoch):
        pass


class CachedValidationLoader:
    """Materialize the bounded initial cohort once per rank, then replay it.

Inputs include original target features; later windows never re-query them.
Disk use is reported. A fresh run directory is required, so there is no silent
reuse of an unrelated cohort. Only tensor fields consumed by the model survive.
"""
    FIELDS = ("hidden_states", "input_ids", "loss_mask", "verifier_last_hidden_states", "document_ids", "position_ids")

    def __init__(self, loader, batches, directory):
        self.loader, self.batches = loader, min(batches, len(loader))
        if self.batches < 1:
            raise ValueError("Empty validation cohort")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.batch_sampler = _Sampler()
        self.ready = False

    def __len__(self):
        return self.batches

    def __iter__(self):
        if not self.ready:
            if hasattr(getattr(self.loader, "batch_sampler", None), "set_epoch"):
                self.loader.batch_sampler.set_epoch(0)
            iterator = iter(self.loader)
            inventory = []
            for i in range(self.batches):
                batch = next(iterator)
                if batch.get("error_records"):
                    raise ValueError("Validation contains data recovery errors; refusing to freeze that cohort")
                tensors = {k: batch[k].detach().cpu().clone() for k in self.FIELDS if k in batch}
                if any(k not in tensors for k in self.FIELDS[:-1]):
                    raise ValueError("Validation batch lacks required model tensors")
                path = self.directory / f"batch_{i:04d}.pt"
                temporary = path.with_suffix(".partial")
                torch.save(tensors, temporary)
                os.replace(temporary, path)
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for data in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(data)
                inventory.append(dict(file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
            self.ready = True
            (self.directory / "manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
            print(f"BOUNDARY_VALIDATION_COHORT batches={self.batches} bytes={sum(x['bytes'] for x in inventory)} path={self.directory}", flush=True)
        for i in range(self.batches):
            yield torch.load(self.directory / f"batch_{i:04d}.pt", map_location="cpu", weights_only=True)
