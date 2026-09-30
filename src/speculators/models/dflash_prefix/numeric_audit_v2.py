"""Explicit diagnostic mode; never used by normal fast serving.

A: original preparation + original walk.
B: original preparation + cached walk (the first measured optimization).
C: token-cache preparation + cached walk (the current optimization).
Detailed score comparisons force A's prefix into B/C after a disagreement.
Forced token outputs are deliberately NOT used as evidence of token agreement.
"""
import json
import os
from pathlib import Path
import tempfile

import torch


RTOL = 0.005
ATOL = 0.02


def write_json(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def tensor_difference(reference, other):
    a, b = reference.detach().float().cpu(), other.detach().float().cpu()
    report = {"shape": list(a.shape), "other_shape": list(b.shape),
              "reference_dtype": str(reference.dtype), "other_dtype": str(other.dtype),
              "reference_nonfinite": int((~torch.isfinite(a)).sum()),
              "other_nonfinite": int((~torch.isfinite(b)).sum())}
    if a.shape != b.shape:
        report["shape_equal"] = False
        return report
    finite = torch.isfinite(a) & torch.isfinite(b)
    difference = (a - b).abs()
    values = difference[finite]
    report.update({"shape_equal": True, "exact_equal": bool(torch.equal(a, b)),
                   "max_abs_error_finite": float(values.max()) if values.numel() else None,
                   "mean_abs_error_finite": float(values.mean()) if values.numel() else None,
                   "outside_tolerance": int((~torch.isclose(a, b, rtol=RTOL, atol=ATOL)).sum()),
                   "elements": a.numel()})
    return report


def token_difference(reference, other):
    a, b = reference.detach().cpu(), other.detach().cpu()
    if a.shape != b.shape:
        return {"equal": False, "shape_equal": False}
    indices = a.ne(b).nonzero()
    first = indices[0].tolist() if len(indices) else None
    return {"equal": len(indices) == 0, "different_tokens": len(indices),
            "first_difference_batch_position": first,
            "reference_token_id": int(a[tuple(first)]) if first is not None else None,
            "other_token_id": int(b[tuple(first)]) if first is not None else None}


def score_difference(reference, other, candidate_ids):
    report = tensor_difference(reference, other)
    a, b = reference.detach().float().cpu(), other.detach().float().cpu()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return report
    # Common additive shifts cannot change argmax; report these separately.
    report["after_row_max_subtraction"] = tensor_difference(
        a - a.amax(-1, keepdim=True), b - b.amax(-1, keepdim=True))
    ids = candidate_ids.detach().cpu()
    report["same_prefix_argmax_tokens"] = token_difference(
        ids.gather(-1, a.argmax(-1, keepdim=True)).squeeze(-1),
        ids.gather(-1, b.argmax(-1, keepdim=True)).squeeze(-1))
    top_a = a.topk(min(2, a.shape[-1]), dim=-1).values
    top_b = b.topk(min(2, b.shape[-1]), dim=-1).values
    margins_a = top_a[..., 0] - top_a[..., -1]
    margins_b = top_b[..., 0] - top_b[..., -1]
    errors = (a - b).abs().amax(-1)
    report["reference_min_top1_top2_margin"] = float(margins_a.min())
    report["other_min_top1_top2_margin"] = float(margins_b.min())
    report["rows_with_margin_at_most_twice_max_error"] = int((margins_a <= 2 * errors).sum())
    report["per_row_first_32"] = [
        {"batch": n, "position_zero_based": i, "max_abs_error": float(errors[n, i]),
         "reference_margin": float(margins_a[n, i]), "other_margin": float(margins_b[n, i])}
        for n, i in [(n, i) for n in range(a.shape[0]) for i in range(a.shape[1])][:32]
    ]
    return report


def original_projected_kv(head, prepared):
    n, b, k = prepared.candidate_ids.shape
    heads = head.prefix_layers[0].heads
    source = torch.cat((prepared.anchor_memory[:, None],
                        prepared.memories[:, :-1].reshape(n, (b - 1) * k, head.rank)), dim=1)
    return torch.stack([
        torch.stack((layer.key_projection(source), layer.value_projection(source)), dim=2)
        for layer in head.prefix_layers
    ], dim=2).reshape(n, 1 + (b - 1) * k, len(head.prefix_layers),
                     2, heads, head.rank // heads).float()


class NumericAudit:
    def __init__(self, head, fast, logger, directory, max_events=4):
        if not directory:
            raise ValueError("Audit mode requires DFLASH_PREFIX_AUDIT_DIR")
        if not 1 <= max_events <= 32:
            raise ValueError("DFLASH_PREFIX_AUDIT_MAX_EVENTS must be in [1, 32]")
        self.head, self.fast, self.logger = head, fast, logger
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self.max_events = max_events
        self.events = 0
        self.path = self.directory / f"summary_{self.pid}.json"
        if self.path.exists():
            raise ValueError("Audit summary already exists; choose a fresh AUDIT_DIR")
        self.state = {"schema_version": 1, "pid": self.pid, "torch_version": str(torch.__version__),
                      "score_check": "audit", "rtol": RTOL, "atol": ATOL,
                      "observed_blocks": 0, "finite_token_equal_blocks": 0,
                      "score_drift_blocks": 0, "token_mismatch_blocks": 0,
                      "nonfinite_blocks": 0, "max_abs_score_error": 0.0,
                      "diagnostic_errors": 0, "events": [], "profile": None,
                      "note": "All reached blocks are checked. This file does not prove the evaluator completed. Audit timing is not benchmark timing."}
        write_json(self.path, self.state)

    @torch.no_grad()
    def details(self, reference, prepared, projected_kv, expected, actual,
                expected_scores, actual_scores):
        b_tokens, _ = self.fast.walk(reference, return_scores=True)
        report = {
            "paths": {"A": "original prepare + original walk",
                      "B": "original prepare + fast walk",
                      "C": "selected prepare/cache + fast walk"},
            "free_running_tokens": {"A_vs_B": token_difference(expected, b_tokens),
                                    "A_vs_C": token_difference(expected, actual),
                                    "B_vs_C": token_difference(b_tokens, actual)},
            "prepared_A_vs_C": {field: tensor_difference(getattr(reference, field), getattr(prepared, field))
                                for field in reference._fields},
        }
        # Never compare scores from different chosen prefixes after a mismatch.
        match = reference.candidate_ids.eq(expected[..., None])
        if not torch.equal(reference.candidate_ids, prepared.candidate_ids) or not match.any(-1).all():
            report["common_prefix_scores_error"] = "Shortlists differ or reference choice is missing"
            return report
        forced = match.to(torch.long).argmax(-1)
        _, b_scores = self.fast.walk(reference, forced_indices=forced, return_scores=True)
        _, c_scores = self.fast.walk(prepared, forced_indices=forced, return_scores=True,
                                     projected_kv=projected_kv)
        report["common_prefix_scores"] = {
            "context": "Each position is conditioned on A's actually selected preceding tokens. Forced outputs do not establish token equality.",
            "A_vs_B": score_difference(expected_scores, b_scores, reference.candidate_ids),
            "A_vs_C": score_difference(expected_scores, c_scores, reference.candidate_ids),
            "B_vs_C": score_difference(b_scores, c_scores, reference.candidate_ids),
        }
        if projected_kv is not None:
            report["projected_kv_A_vs_C"] = tensor_difference(
                original_projected_kv(self.head, reference), projected_kv)
        report["observed_score_error_before_any_prefix_divergence"] = tensor_difference(
            expected_scores[:, :1], actual_scores[:, :1])
        return report

    def record(self, *, reference, prepared, projected_kv, expected, actual,
               expected_scores, actual_scores, finite, tokens_equal, scores_close, error):
        s = self.state
        s["observed_blocks"] += 1
        if finite:
            s["max_abs_score_error"] = max(s["max_abs_score_error"], error)
        s["finite_token_equal_blocks"] += int(finite and tokens_equal)
        s["score_drift_blocks"] += int(finite and not scores_close)
        s["token_mismatch_blocks"] += int(not tokens_equal)
        s["nonfinite_blocks"] += int(not finite)
        fatal = not finite or not tokens_equal
        capture = fatal or (self.events < self.max_events and (s["observed_blocks"] == 1 or not scores_close))
        if capture:
            self.events += 1
            name = f"event_{self.pid}_{self.events:03d}.json"
            event = {"block": s["observed_blocks"], "finite": finite,
                     "tokens_equal": tokens_equal, "scores_within_tolerance": scores_close,
                     "max_abs_score_error": error if finite else None, "fatal": fatal}
            try:
                event.update(self.details(reference, prepared, projected_kv, expected, actual,
                                          expected_scores, actual_scores))
            except Exception as exc:
                # Preserve the observed failure even if extra diagnostics fail.
                event["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
                s["diagnostic_errors"] += 1
            write_json(self.directory / name, event)
            s["events"].append(name)
            self.logger.info("PREFIX_AUDIT_EVENT block=%d finite=%s tokens_equal=%s scores_close=%s file=%s",
                             s["observed_blocks"], finite, tokens_equal, scores_close, self.directory / name)
        write_json(self.path, s)

    def record_profile(self, report):
        self.state["profile"] = report
        write_json(self.path, self.state)

    def profile_failure(self, reference, prepared, projected_kv, outputs):
        expected, scores_a = self.head.walk(reference, return_scores=True)
        actual, scores_c = self.fast.walk(prepared, return_scores=True, projected_kv=projected_kv)
        event = {"fatal": True, "cause": "profiling alternatives selected different tokens",
                 "observed_profiling_tokens_B_vs_C": token_difference(outputs["uncached"], outputs["cached"])}
        try:
            event.update(self.details(reference, prepared, projected_kv, expected, actual, scores_a, scores_c))
        except Exception as exc:
            event["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
            self.state["diagnostic_errors"] += 1
        name = f"profile_failure_{self.pid}.json"
        write_json(self.directory / name, event)
        self.state["profiling_token_mismatch"] = True
        self.state["events"].append(name)
        write_json(self.path, self.state)
