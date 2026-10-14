#!/usr/bin/env python3
"""Read-only replay of a completed boundary pilot's cached validation cohort.

No optimizer, target forward/server, checkpoint write, or serving patch is used.
Metrics concern recorded reference prefixes, not live speculative decoding tau.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_first_losses(student, baseline, candidates, reference, valid, preserve):
    """Partition all paired reference-token loss by the FIRST earlier mismatch.

    At that position the preceding student and baseline reference prefixes agree.
    'support' means the reference token is absent, 'selection' means present.
    Neither category identifies which optimizer/loss term caused the change.
    """
    if student.shape != baseline.shape or student.shape != reference.shape:
        raise ValueError("Mismatched proposal shapes")
    if valid.shape != student.shape or preserve.shape != student.shape:
        raise ValueError("Mismatched mask shapes")
    if candidates.shape[:2] != student.shape or candidates.shape[-1] < 1:
        raise ValueError("Mismatched candidate shape")
    alive = valid[:, 0].bool()
    accepted_s = (student.eq(reference) & valid).long().cumprod(-1).sum(-1)
    accepted_b = (baseline.eq(reference) & valid).long().cumprod(-1).sum(-1)
    delta = accepted_s - accepted_b
    lost = delta.lt(0) & alive
    position = accepted_s.clamp_max(student.shape[1] - 1)
    coverage = candidates.eq(reference[..., None]).any(-1)
    covered = coverage.gather(1, position[:, None]).squeeze(1)
    guarded = preserve.gather(1, position[:, None]).squeeze(1)
    support = lost & ~covered
    selection = lost & covered
    severity = (-delta).clamp_min(0)
    sum_if = lambda value, mask: (value * mask).sum()
    return {
        "anchors": alive.sum(),
        "student_reference_tokens_with_bonus": sum_if(accepted_s + 1, alive),
        "baseline_reference_tokens_with_bonus": sum_if(accepted_b + 1, alive),
        "gained_reference_tokens": sum_if(delta.clamp_min(0), alive),
        "lost_reference_tokens": sum_if(severity, alive),
        "losing_anchors": lost.sum(),
        "support_loss_anchors": support.sum(),
        "selection_loss_anchors": selection.sum(),
        "support_lost_reference_tokens": sum_if(severity, support),
        "selection_lost_reference_tokens": sum_if(severity, selection),
        "loss_anchors_guarded_by_v2": (lost & guarded).sum(),
        "loss_anchors_not_guarded_by_v2": (lost & ~guarded).sum(),
        "lost_reference_tokens_guarded_by_v2": sum_if(severity, lost & guarded),
        "lost_reference_tokens_not_guarded_by_v2": sum_if(severity, lost & ~guarded),
    }


@torch.no_grad()
def audit_batch(model, batch, active, chunk):
    from speculators.models.dflash_prefix.boundary_objective import dense_boundary_anchors
    from speculators.models.dflash_prefix.core import reachable_greedy_prefix
    from speculators.models.dflash_prefix.selector_v2 import unary_topk

    p = model.block_size - 1
    total = active + p - 1  # Keep the exact original backbone shapes.
    anchors = dense_boundary_anchors(batch["loss_mask"], total, model.block_size, training=False)
    model._boundary_anchors = model.teacher_model._boundary_anchors = anchors
    th, tu, target, aligned, indices = model.teacher_model._backbone_forward(**batch, max_anchors=total)
    h, u, _, _, _ = model._backbone_forward(**batch, max_anchors=total)
    block = model.block_size
    h, th = h.reshape(total, block, -1)[:, 1:], th.reshape(total, block, -1)[:, 1:]
    u, tu = u.reshape(total, block, -1)[:, 1:], tu.reshape(total, block, -1)[:, 1:]
    targets = target.reshape(total, block, -1)[:, 1:].argmax(-1)
    del target
    ids = batch["input_ids"][0, indices].reshape(total, block)
    docs = batch["document_ids"][0, indices].reshape(total, block)
    valid = aligned.reshape(total, block)[:, 1:].bool() & docs[:, 1:].eq(docs[:, :1])
    valid = valid.long().cumprod(-1).bool()
    reference, anchor = ids[:, 1:], ids[:, 0]
    logits, candidates = unary_topk(u[:active], model.config.prefix_top_k)
    old_logits, old_candidates = unary_topk(tu, model.config.prefix_top_k)
    coverage = candidates.eq(reference[:active, :, None]).any(-1)
    context = reachable_greedy_prefix(reference[:active], targets[:active], coverage)
    forced = candidates.eq(reference[:active, :, None]).long().argmax(-1)
    accumulated = defaultdict(int)
    for start in range(0, active, chunk):
        sl = slice(start, min(active, start + chunk))
        c = candidates[sl]
        tables = model.prefix_head.prepare_tables(h[sl], c, logits[sl], anchor[sl],
                                                  embedding_weight=model.embed_tokens.weight)
        student = model.prefix_head.walk(tables)
        native = model.teacher_model.prefix_head.prepare_tables(
            th[sl], old_candidates[sl], old_logits[sl], anchor[sl],
            embedding_weight=model.teacher_model.embed_tokens.weight)
        baseline = model.teacher_model.prefix_head.walk(native)
        # Reconstruct the EXISTING v2 preservation gate exactly, for comparison.
        common = model.teacher_model.prefix_head.prepare_tables(
            th[sl], c, tu[sl].gather(-1, c), anchor[sl],
            embedding_weight=model.teacher_model.embed_tokens.weight)
        _, scores = model.teacher_model.prefix_head.walk(common, forced_indices=forced[sl], return_scores=True)
        matched = c.eq(targets[sl, :, None])
        labels = matched.long().argmax(-1)
        preserve = valid[sl] & context[sl] & matched.any(-1) & scores.argmax(-1).eq(labels)
        stats = split_first_losses(student, baseline, c, reference[sl], valid[sl], preserve)
        for key, value in stats.items():
            accumulated[key] += value
    return accumulated


def cohort_inventory(run, expected_ranks, batches):
    result = []
    actual = {p.name for p in (run / "validation_inputs").glob("rank_*") if p.is_dir()}
    expected = {f"rank_{i}" for i in range(expected_ranks)}
    if actual != expected:
        raise ValueError(f"Cached rank directories differ: expected {sorted(expected)}, found {sorted(actual)}")
    for rank in range(expected_ranks):
        directory = run / "validation_inputs" / f"rank_{rank}"
        manifest = json.loads((directory / "manifest.json").read_text())
        if not manifest or len(manifest) > batches:
            raise ValueError(f"Unexpected cached batch count: {directory}")
        names = [entry["file"] for entry in manifest]
        if names != [f"batch_{i:04d}.pt" for i in range(len(manifest))]:
            raise ValueError(f"Invalid cached file inventory: {directory}")
        for entry in manifest:
            result.append((directory / entry["file"], entry))
    lengths = [sum(p.parent.name == f"rank_{rank}" for p, _ in result) for rank in range(expected_ranks)]
    if len(set(lengths)) != 1:
        raise ValueError("Original validation ranks have different cached batch counts")
    return result


def load_student(model, checkpoint, original):
    from safetensors import safe_open
    from speculators.models.dflash_prefix.compiled_training import _TARGET_OWNED
    cfg = json.loads((checkpoint / "config.json").read_text())
    before = json.loads((original / "config.json").read_text())
    # Compare all configuration except known training/serialization metadata.
    ignored = {"_name_or_path", "transformers_version", "dtype", "torch_dtype",
               "prefix_training_experiment", "prefix_freeze_backbone", "prefix_detach_backbone"}
    comparable = lambda data: {k: v for k, v in data.items() if k not in ignored}
    if comparable(cfg) != comparable(before):
        changed = sorted(k for k in set(cfg) | set(before)
                         if k not in ignored and cfg.get(k) != before.get(k))
        raise ValueError(f"Checkpoint configuration differs outside training metadata: {changed}")
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("teacher_model.")}
    index = checkpoint / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    seen, inventory = set(), {}
    for filename in files:
        if Path(filename).name != filename:
            raise ValueError("Checkpoint shard path must be a basename")
        path = checkpoint / filename
        inventory[filename] = sha256(path)
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key in seen or key not in state:
                    raise ValueError(f"Unexpected or duplicate checkpoint tensor: {key}")
                value = stream.get_tensor(key)
                if value.shape != state[key].shape:
                    raise ValueError(f"Checkpoint tensor shape mismatch: {key}")
                value = value.to(state[key].dtype)
                if key in _TARGET_OWNED and not torch.equal(value, state[key].cpu()):
                    raise ValueError(f"Frozen target-owned tensor changed: {key}")
                state[key].copy_(value)
                seen.add(key)
    if set(state) - seen - _TARGET_OWNED:
        raise ValueError(f"Missing trained tensors: {sorted(set(state) - seen - _TARGET_OWNED)}")
    return inventory


def summarize(counts):
    n = counts["anchors"]
    if n < 1:
        raise ValueError("No usable validation anchors")
    ratio = lambda key: counts[key] / n
    return {
        "student_reference_eal": ratio("student_reference_tokens_with_bonus"),
        "baseline_reference_eal": ratio("baseline_reference_tokens_with_bonus"),
        "paired_reference_gain": (counts["gained_reference_tokens"] - counts["lost_reference_tokens"]) / n,
        "reference_tokens_gained": ratio("gained_reference_tokens"),
        "reference_tokens_lost": ratio("lost_reference_tokens"),
        "support_reference_tokens_lost": ratio("support_lost_reference_tokens"),
        "selection_reference_tokens_lost": ratio("selection_lost_reference_tokens"),
        "reference_tokens_lost_guarded_by_v2": ratio("lost_reference_tokens_guarded_by_v2"),
        "reference_tokens_lost_not_guarded_by_v2": ratio("lost_reference_tokens_not_guarded_by_v2"),
    }


def compare_history(run, checkpoint, actual):
    rows = [json.loads(line) for line in (run / "checkpoints/boundary_history.jsonl").read_text().splitlines() if line.strip()]
    is_original = checkpoint == Path(json.loads((run / "settings.json").read_text())["init_checkpoint"]).resolve()
    window = -1 if is_original else int(checkpoint.name) if checkpoint.parent == (run / "checkpoints").resolve() and checkpoint.name.isdecimal() else None
    matches = [row for row in rows if row["window"] == window]
    if len(matches) != 1:
        return {"comparable": False, "pass": False, "reason": "Cannot identify one matching history window"}
    expected = matches[0]["metrics"]
    details = {}
    for name in ("student_reference_eal", "baseline_reference_eal", "paired_reference_gain", "reference_tokens_gained", "reference_tokens_lost"):
        value = expected.get(name + "_epoch")
        details[name] = {"history": value, "replay": actual[name], "pass": value is not None and math.isclose(value, actual[name], rel_tol=0., abs_tol=1e-5)}
    return {"comparable": True, "pass": all(x["pass"] for x in details.values()),
            "window": window, "step": matches[0]["step"], "metrics": details,
            "scope": "Aggregated cached reference metrics only, not serving parity"}


def compare_forward(metrics, replay):
    """Compare the diagnostic with the actual training validation forward."""
    names = ("student_reference_eal", "baseline_reference_eal", "paired_reference_gain",
             "reference_tokens_gained", "reference_tokens_lost")
    values = torch.stack([metrics[name + suffix] for name in names
                          for suffix in ("_sum", "_total")]).cpu().tolist()
    details = {}
    for i, name in enumerate(names):
        value = values[2 * i] / max(values[2 * i + 1], 1)
        details[name] = {"original_forward": value, "audit": replay[name],
                         "pass": math.isclose(value, replay[name], rel_tol=0., abs_tol=1e-5)}
    return {"pass": all(x["pass"] for x in details.values()), "metrics": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-forward-batches", type=int, default=2,
                        help="Run the original validation forward on this many cached batches as an internal cross-check (default 2)")
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve(strict=True)
    settings = json.loads((run / "settings.json").read_text())
    experiment = json.loads((run / "experiment.json").read_text())
    original = Path(settings["init_checkpoint"]).resolve(strict=True)
    checkpoint = (args.checkpoint or run / "checkpoints/3").expanduser().resolve(strict=True)
    output = (args.output or run / f"boundary_loss_audit_{time.time_ns()}.json").resolve()
    # Never overwrite an existing report, config or checkpoint file.
    with output.open("x") as stream:
        json.dump({"complete": False, "diagnostic_only": True}, stream)
    report = {"complete": False, "diagnostic_only": True, "audit_version": 2, "run_dir": str(run),
              "checkpoint": str(checkpoint), "teacher_checkpoint": str(original), "device": args.device,
              "note": "Cached offline reference-prefix audit. No training, target verification or throughput benchmark."}
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Run this diagnostic as one process, not torchrun")
        if args.reference_forward_batches < 1:
            raise ValueError("At least one original-forward comparison is required")
        torch.set_num_threads(1)
        if args.device.startswith("npu"):
            import torch_npu  # noqa: F401
            torch.npu.set_device(args.device)
        device = torch.device(args.device)
        if device.type not in {"cpu", "npu"}:
            raise ValueError("This diagnostic supports cpu or npu")
        from speculators.models.dflash_prefix.boundary_training import build_boundary_model
        from speculators.models.dflash_prefix.boundary_validation import CachedValidationLoader, exported_precision
        report["provenance"] = {
            "torch": torch.__version__,
            "physical_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "audit_script_sha256": sha256(__file__),
            "settings_sha256": sha256(run / "settings.json"),
            "experiment_sha256": sha256(run / "experiment.json"),
            "student_config_sha256": sha256(checkpoint / "config.json"),
        }
        for filename, expected in experiment["initial_weight_sha256"].items():
            if Path(filename).name != filename or sha256(original / filename) != expected:
                raise ValueError("Initial teacher weights differ from the recorded training run")
        if sha256(original / "config.json") != experiment["initial_config_sha256"]:
            raise ValueError("Initial teacher configuration changed since training")
        files = cohort_inventory(run, len(experiment["trainer_npus"]), settings["val_batches"])
        active = int(experiment["train_config"]["max_anchors"])
        model = build_boundary_model(original, settings["target_path"], settings)
        model = model.eval().requires_grad_(False)
        report["student_weight_sha256"] = load_student(model, checkpoint, original)
        model.to(device)
        totals = defaultdict(int)
        print(f"BOUNDARY_LOSS_AUDIT checkpoint={checkpoint} cached_batches={len(files)} device={device}; no server or optimizer", flush=True)
        forward_checks = []
        # Match train_entry.validate exactly: convert PARAMETERS only. Module.to
        # with dtype=bf16 also quantizes FP32 RoPE inv_freq buffers and changes
        # both teacher and student predictions, even for unchanged checkpoints.
        with exported_precision(model):
            report["precision"] = {
                "mode": "training_exported_precision_parameters_only",
                "floating_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
                "floating_buffer_dtypes": {name: str(value.dtype) for name, value in model.named_buffers() if value.is_floating_point()},
            }
            print("BOUNDARY_AUDIT_PRECISION parameters=BF16 buffers=preserved; using training exported_precision", flush=True)
            for i, (path, entry) in enumerate(files):
                if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Cached validation tensor file changed: {path}")
                batch = torch.load(path, map_location="cpu", weights_only=True)
                if set(batch) - set(CachedValidationLoader.FIELDS) or any(k not in batch for k in CachedValidationLoader.FIELDS[:-1]):
                    raise ValueError(f"Unexpected cached tensor fields: {path}")
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                    if i < args.reference_forward_batches:
                        _, _, original_metrics = model(**batch, max_anchors=active)
                    stats = audit_batch(model, batch, active, settings["anchor_chunk_size"])
                # One small device-to-host copy per batch. This is not timed serving.
                values = torch.stack(list(stats.values())).cpu().tolist()
                local = {key: int(value) for key, value in zip(stats, values)}
                for key, value in local.items():
                    totals[key] += value
                if i < args.reference_forward_batches:
                    if local["anchors"]:
                        check = compare_forward(original_metrics, summarize(local))
                        check["cache_file"] = str(path.relative_to(run))
                    else:
                        check = {"pass": True, "skipped": "No usable anchors", "cache_file": str(path.relative_to(run))}
                    forward_checks.append(check)
                if (i + 1) % 8 == 0 or i + 1 == len(files):
                    print(f"Audited cached batches {i + 1}/{len(files)}", flush=True)
        metrics = summarize(totals)
        comparison = compare_history(run, checkpoint, metrics)
        checked = [check for check in forward_checks if "skipped" not in check]
        original_forward = {"pass": bool(checked) and all(check["pass"] for check in checked),
                            "checked_batches": len(checked), "comparisons": forward_checks,
                            "scope": "Same-process comparison on these cached batches, not serving parity"}
        report.update(complete=True, cached_batches=len(files), counts=dict(totals), metrics=metrics,
                      history_comparison=comparison, original_forward_comparison=original_forward,
                      interpretation_ready=bool(comparison["pass"] and original_forward["pass"]))
        print(json.dumps({"metrics": metrics, "history_comparison": comparison,
                          "original_forward_comparison": original_forward}, indent=2), flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Small audit report: {output}", flush=True)


if __name__ == "__main__":
    main()
