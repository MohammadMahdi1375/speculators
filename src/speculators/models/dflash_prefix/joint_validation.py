"""Fixed validation cohort and actual on-disk checkpoint parameter replay."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


class FixedSampler:
    def set_epoch(self, epoch):
        pass


class CachedValidationLoader:
    FIELDS = ("hidden_states", "input_ids", "loss_mask", "verifier_last_hidden_states", "document_ids", "position_ids")

    def __init__(self, loader, batches, directory):
        self.loader, self.batches = loader, min(batches, len(loader))
        if self.batches < 1:
            raise ValueError("Empty validation cohort")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.batch_sampler, self.ready = FixedSampler(), False

    def __len__(self):
        return self.batches

    def __iter__(self):
        if not self.ready:
            sampler = getattr(self.loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(0)
            iterator, inventory = iter(self.loader), []
            for i in range(self.batches):
                batch = next(iterator)
                if batch.get("error_records"):
                    raise ValueError("Validation contains recovery errors; refusing to cache it")
                tensors = {k: batch[k].detach().cpu().clone() for k in self.FIELDS if k in batch}
                if any(k not in tensors for k in self.FIELDS[:-1]):
                    raise ValueError("Validation batch lacks required tensors")
                path = self.directory / f"batch_{i:04d}.pt"
                temporary = path.with_suffix(".partial")
                torch.save(tensors, temporary)
                os.replace(temporary, path)
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for data in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(data)
                inventory.append(dict(file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
            self.ready = True
            (self.directory / "manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
            print(f"JOINT_VALIDATION_COHORT batches={self.batches} bytes={sum(x['bytes'] for x in inventory)} path={self.directory}", flush=True)
        for i in range(self.batches):
            yield torch.load(self.directory / f"batch_{i:04d}.pt", map_location="cpu", weights_only=True)


@contextmanager
def checkpoint_parameters(model, directory):
    """Yield an isolated saved-weight model and its provenance record.

The live model is never cast or rebound: its DDP gradient accumulators must
survive validation. Only saved BF16 parameters and small buffers are copied;
omitted frozen target weights are shared read-only. No second FP32 master
model is allocated. The temporary model is for inference only.
"""
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    for key in ("block_size", "prefix_selector_kind", "prefix_rank", "prefix_top_k", "prefix_attention_heads"):
        if cfg.get(key) != getattr(model.config, key):
            raise ValueError(f"Exported checkpoint {key} differs from live model")
    if cfg.get("architectures") != ["DFlashPrefixDraftModel"]:
        raise ValueError("Exported checkpoint has a training-only architecture name")
    index = directory / "model.safetensors.index.json"
    filenames = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    parameters = dict(model.named_parameters())
    required = {name for name, p in parameters.items() if p.requires_grad}
    ignored = set(getattr(model, "_keys_to_ignore_on_save", []))
    expected = set(model.state_dict()) - ignored
    seen, memo = set(), {}
    # Frozen target weights omitted from the export are already correct and
    # have no training gradients. Sharing them avoids duplicating the target
    # embedding/LM-head on each validation rank.
    for name, p in parameters.items():
        if not p.requires_grad:
            memo[id(p)] = p
    for key in ("selector_guard", "joint_metrics"):
        value = getattr(model, key, None)
        if value is not None:
            memo[id(value)] = None
    try:
        for filename in filenames:
            file = (directory / filename).resolve()
            if not file.is_relative_to(directory.resolve()):
                raise ValueError("Checkpoint shard escapes checkpoint directory")
            with safe_open(str(file), framework="pt", device="cpu") as stream:
                for name in stream.keys():
                    if name in seen or name not in expected:
                        raise ValueError(f"Unexpected/duplicate checkpoint tensor: {name}")
                    seen.add(name)
                    value = stream.get_tensor(name)
                    if name not in parameters:
                        # This architecture should only persist parameters. Do
                        # not silently alter rotary buffer precision on reload.
                        raise ValueError(f"Unexpected persistent buffer: {name}")
                    p = parameters[name]
                    if value.shape != p.shape or (value.is_floating_point() and value.dtype != torch.bfloat16):
                        raise ValueError(f"Checkpoint shape/export dtype mismatch: {name}")
                    memo[id(p)] = torch.nn.Parameter(value.to(p.device), requires_grad=False)
        if required - seen or expected - seen:
            raise ValueError(f"Incomplete exported weights: {sorted((required | expected) - seen)[:8]}")
        replay_model = copy.deepcopy(model, memo).eval()
        info = dict(source=str(directory), parameter_tensors=len(seen),
                    precision="saved BF16 parameters; original FP32 rotary buffers", actual_disk_reload=True,
                    isolated_model=True, live_parameter_metadata_untouched=True)
        yield replay_model, info
    finally:
        # Drop memo references promptly; live master weights need no restoration.
        memo.clear()
