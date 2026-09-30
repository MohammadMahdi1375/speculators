"""Failure-only diagnostics. Never changes scores, tokens, or parity tolerances.

The caller must still raise its original parity exception. No diagnostic work
is performed on successful blocks, including production fast-path blocks.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch


def cpu(tensor):
    return tensor.detach().to(device="cpu").clone()


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def score_comparison(expected, actual):
    """Use EXACTLY the orientation/tolerance of the runtime's allclose check."""
    a, b = cpu(expected).float(), cpu(actual).float()
    if a.shape != b.shape:
        return {"shape_equal": False, "expected_shape": list(a.shape),
                "actual_shape": list(b.shape), "close": False}
    finite = torch.isfinite(a) & torch.isfinite(b)
    diff = (a - b).abs()
    tolerance = 0.02 + 0.005 * b.abs()
    close = torch.isclose(a, b, rtol=0.005, atol=0.02)
    bad = (~close).nonzero()
    # Show actual failing elements, not just the largest absolute difference.
    examples = []
    for coordinate in bad[:16].tolist():
        ix = tuple(coordinate)
        examples.append({"index": coordinate, "expected": number(a[ix]),
                         "actual": number(b[ix]), "abs_error": number(diff[ix]),
                         "allowed_error": number(tolerance[ix])})
    result = {"shape_equal": True, "finite": bool(finite.all()),
              "close": bool(close.all()), "rtol": 0.005, "atol": 0.02,
              "tolerance_reference": "actual (second torch.allclose argument)",
              "max_abs_error": number(diff.max()), "failed_elements": len(bad),
              "failed_examples": examples}
    if a.ndim == 3 and a.shape[-1] >= 2:
        result["max_abs_error_per_position"] = [number(x) for x in diff.amax((0, 2))]
        av, ai = a.topk(2, dim=-1)
        bv, bi = b.topk(2, dim=-1)
        result["expected_winning_candidate_indices"] = a.argmax(-1).tolist()
        result["actual_winning_candidate_indices"] = b.argmax(-1).tolist()
        result["expected_top1_top2_margin"] = [[number(x) for x in row] for row in av[..., 0] - av[..., 1]]
        result["actual_top1_top2_margin"] = [[number(x) for x in row] for row in bv[..., 0] - bv[..., 1]]
    return result


def comparison(expected_tokens, expected_scores, tokens, scores, production=None):
    expected_tokens, tokens = cpu(expected_tokens), cpu(tokens)
    result = score_comparison(expected_scores, scores)
    result["scored_tokens_equal"] = torch.equal(expected_tokens, tokens)
    result["first_token_difference"] = None
    if expected_tokens.shape == tokens.shape:
        different = (expected_tokens != tokens).nonzero()
        if len(different):
            ix = tuple(different[0].tolist())
            result["first_token_difference"] = {"index": list(ix),
                "expected_token_id": int(expected_tokens[ix]), "actual_token_id": int(tokens[ix])}
            result["score_context_note"] = (
                "After the first different choice, free-running scores use different histories. "
                "Use the forced_reference_prefix comparisons to isolate arithmetic.")
    if production is not None:
        result["production_tokens_equal"] = torch.equal(expected_tokens, cpu(production))
    return result


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def head_config(head):
    return {"vocab_size": head.vocab_size, "hidden_size": head.hidden_size,
            "rank": head.rank, "top_k": head.top_k, "max_proposals": head.max_proposals,
            "heads": head.prefix_layers[0].heads, "layers": len(head.prefix_layers),
            "correction_scale": head.correction_scale, "history_scale": head.history_scale}


def snapshot_payload(runtime, prepared, expected, expected_scores, actual, actual_scores,
                     production, reason):
    """Only the small selector weights/tables. No target or backbone weights."""
    payload = {"format": "prefix_failure_snapshot_v1", "reason": reason,
               "head_config": head_config(runtime.head),
               "state_dict": {key: cpu(value) for key, value in runtime.head.state_dict().items()},
               "prepared": {key: cpu(value) for key, value in zip(prepared._fields, prepared)},
               "expected_tokens": cpu(expected), "expected_scores": cpu(expected_scores)}
    for key, value in (("active_scored_tokens", actual), ("active_scores", actual_scores),
                       ("active_production_tokens", production)):
        if value is not None:
            payload[key] = cpu(value)
    return payload


def projection_checks(head, prepared, forced):
    """Compare shared precomputation with the reference's projection shapes."""
    n, b, k = prepared.candidate_ids.shape
    result = {}
    first = head.prefix_layers[0]
    batched = first.query_projection(first.query_norm(prepared.queries))
    separate = torch.cat([first.query_projection(first.query_norm(prepared.queries[:, i:i+1]))
                          for i in range(b)], dim=1)
    result["first_query_projection_separate_vs_batched"] = score_comparison(separate, batched)
    if b <= 1:
        return result
    chosen = prepared.memories.gather(2, forced[..., None, None].expand(n, b, 1, head.rank)).squeeze(2)
    source = torch.cat((prepared.anchor_memory[:, None],
                        prepared.memories[:, :-1].reshape(n, (b-1)*k, head.rank)), dim=1)
    null = source.new_zeros(n, 1, head.rank)
    # Compare the entire sequence of prefix lengths used by the reference.
    for li, layer in enumerate(head.prefix_layers):
        for name in ("key_projection", "value_projection"):
            projection = getattr(layer, name)
            projected = projection(source)
            candidates = projected[:, 1:].reshape(n, b-1, k, head.rank)
            selected = candidates.gather(2, forced[:, :-1, None, None].expand(n, b-1, 1, head.rank)).squeeze(2)
            shared = torch.cat((null, projected[:, :1], selected), dim=1)
            rows = []
            for i in range(1, b):
                reference_memory = torch.cat((null, prepared.anchor_memory[:, None], chosen[:, :i]), dim=1)
                rows.append({"proposal_index_zero_based": i,
                             **score_comparison(projection(reference_memory), shared[:, :i+2])})
            result[f"layer_{li}_{name}_reference_vs_cached"] = rows
    return result


@torch.no_grad()
def collect_failure(runtime, raw, prepared, expected, expected_scores,
                    actual, actual_scores, production, reason, embedding_weight):
    directory = os.environ.get("DFLASH_PREFIX_FAILURE_DIR")
    if not directory:
        runtime.logger.warning("PREFIX_DIAGNOSTIC_DISABLED: set DFLASH_PREFIX_FAILURE_DIR to collect a failure")
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"prefix_failure_{time.time_ns()}_{os.getpid()}"
    destination = root / (stem + ".json")
    tensor_path = root / (stem + ".pt")
    suite = runtime.graph_selector
    winner = getattr(suite, "winner", "paired" if runtime.history_group_size == 2 else "cached")
    report = {"format": "prefix_failure_diagnostics_v1", "complete": False,
              "pid": os.getpid(), "reason": reason, "mode": runtime.mode,
              "checked_blocks_before_failure": runtime.checked, "active_variant": winner,
              "history_group_size": runtime.history_group_size,
              "torch_version": str(torch.__version__), "device": str(raw[0].device),
              "input_dtype": str(raw[0].dtype), "head_config": head_config(runtime.head),
              "snapshot": str(tensor_path.resolve()), "comparisons": {},
              "note": "Failure-only diagnostics, not a benchmark or a parity waiver. "
                      "The original strict exception is still raised. Snapshot replay covers "
                      "walk arithmetic from prepared tables, not full graph preparation or serving."}
    draft = os.environ.get("DFLASH_PREFIX_FUSION_DRAFT")
    if draft:
        cfg = Path(draft) / "config.json"
        report["draft_model"] = draft
        if cfg.is_file():
            report["draft_config_sha256"] = hashlib.sha256(cfg.read_bytes()).hexdigest()
    atomic_json(destination, report)
    runtime.logger.warning("PREFIX_DIAGNOSTIC_BEGIN report=%s snapshot=%s", destination, tensor_path)
    try:
        # Persist before any additional device computation, so later errors
        # cannot erase the original failing input and outputs.
        payload = snapshot_payload(runtime, prepared, expected, expected_scores,
                                   actual, actual_scores, production, reason)
        temporary = tensor_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(tensor_path)
        report["snapshot_bytes"] = tensor_path.stat().st_size

        def record(name, tokens, scores, prod=None):
            report["comparisons"][name] = comparison(expected, expected_scores, tokens, scores, prod)
            atomic_json(destination, report)
            return tokens, scores

        if actual_scores is not None and actual is not None:
            record("reference_vs_active_graph", actual, actual_scores, production)
        record("reference_vs_reference_repeat", *runtime.head.walk(prepared, return_scores=True))
        from .inference_v2 import CachedPrefixWalk
        from .paired_history_v2 import PairedHistoryWalk
        cached = CachedPrefixWalk(runtime.head)
        paired = PairedHistoryWalk(runtime.head)
        eager = {}
        for name, walker in (("cached", cached), ("paired", paired)):
            eager[name] = record(f"reference_vs_eager_{name}", *walker.walk(prepared, return_scores=True))
        report["comparisons"]["eager_cached_vs_eager_paired"] = comparison(*eager["cached"], *eager["paired"])
        atomic_json(destination, report)

        # Reuse the already-captured graphs; do not create another server,
        # retune the winner, or retry a failed device kernel.
        paired_graph = getattr(suite, "graphs", {}).get("paired")
        if paired_graph is None and winner == "paired":
            paired_graph = suite
        if paired_graph is not None:
            kw = {"embedding_weight": embedding_weight}
            gt, gs = paired_graph.select(*raw, return_scores=True, **kw)
            gp = paired_graph.select(*raw, **kw)
            record("reference_vs_paired_graph", gt, gs, gp)
            report["comparisons"]["eager_paired_vs_paired_graph"] = comparison(*eager["paired"], gt, gs, gp)
            if actual is not None and actual_scores is not None:
                report["comparisons"]["paired_graph_vs_active_graph"] = comparison(gt, gs, actual, actual_scores, production)
            atomic_json(destination, report)

        matches = prepared.candidate_ids.eq(expected[..., None])
        if not bool(matches.any(-1).all()):
            raise ValueError("Reference token absent from its candidate list")
        forced = matches.long().argmax(-1)
        for name, walker in (("cached", cached), ("paired", paired)):
            record(f"forced_reference_prefix_vs_{name}",
                   *walker.walk(prepared, forced_indices=forced, return_scores=True))
        if winner in {"select", "local", "select_softmax", "local_softmax", "select_unpaired", "local_unpaired"}:
            if winner.endswith("_unpaired"):
                from .fused_unpaired_v2 import FusedUnpairedHistoryWalk
                fused = FusedUnpairedHistoryWalk(runtime.head, variant=winner.removesuffix("_unpaired"))
            else:
                from .fused_history_v2 import FusedPairedHistoryWalk
                fused = FusedPairedHistoryWalk(runtime.head, variant=winner)
            ft, fs = record("reference_vs_eager_fused", *fused.walk(prepared, return_scores=True))
            report["comparisons"]["eager_paired_vs_eager_fused"] = comparison(*eager["paired"], ft, fs)
            if actual is not None and actual_scores is not None:
                report["comparisons"]["eager_fused_vs_active_graph"] = comparison(ft, fs, actual, actual_scores, production)
            record("forced_reference_prefix_vs_fused", *fused.walk(prepared, forced_indices=forced, return_scores=True))
        report["projection_checks"] = projection_checks(runtime.head, prepared, forced)
        report["complete"] = True
    except Exception as exc:
        # Preserve the original failure. Stop diagnosis on the first diagnostic
        # error instead of continuing work on a possibly unhealthy device.
        report["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(destination, report)
    summary = {name: {k: result.get(k) for k in ("close", "scored_tokens_equal", "max_abs_error", "failed_elements")}
               for name, result in report["comparisons"].items()}
    runtime.logger.warning("PREFIX_DIAGNOSTIC_RESULT %s", json.dumps({"report": str(destination),
                           "complete": report["complete"], "comparisons": summary}, sort_keys=True))
    return str(destination)
