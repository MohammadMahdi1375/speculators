"""Real-input parity and optional warmup-only selection of a fused backend.

The outer PrefixInferenceRuntime still validates against head.walk (the trained
reference), including the production token-only graph. This module additionally
compares each candidate against the working paired graph on every tuning block.
No per-token confidence rule, proposal truncation, or runtime weight changes.
"""
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from .npu_graph_selector_v2 import NPUGraphPrefixSelector
from .paired_history_v2 import PairedHistoryWalk
from .fused_history_v2 import FusedPairedHistoryWalk
from .token_cache_v2 import synchronize


VARIANTS = ("select", "local", "select_softmax", "local_softmax")


def check_result(expected, expected_scores, scored, scores, production):
    finite = bool(torch.isfinite(expected_scores).all() and torch.isfinite(scores).all())
    error = (expected_scores - scores).abs().max().item() if finite else None
    tokens_equal = torch.equal(expected, scored) and torch.equal(expected, production)
    close = finite and torch.allclose(expected_scores, scores, rtol=0.005, atol=0.02)
    reason = None if finite and tokens_equal and close else (
        "nonfinite_scores" if not finite else "token_mismatch" if not tokens_equal else "score_drift")
    return reason, error


class FusedSelectorSuite:
    """Same select interface as NPUGraphPrefixSelector, with a frozen winner."""

    def __init__(self, head, logger, mode, check_steps):
        if mode not in (*VARIANTS, "auto"):
            raise ValueError("FUSION must be off, auto, select, local, select_softmax, or local_softmax")
        self.head, self.logger, self.mode = head, logger, mode
        self.tune_steps = int(os.environ.get("DFLASH_PREFIX_FUSION_TUNE_STEPS", "32"))
        self.profile_steps = int(os.environ.get("DFLASH_PREFIX_FUSION_PROFILE_STEPS", "8"))
        self.minimum_gain = float(os.environ.get("DFLASH_PREFIX_FUSION_MIN_GAIN", "0.02"))
        if not 2 <= self.tune_steps <= check_steps or not 1 <= self.profile_steps <= self.tune_steps:
            raise ValueError("Need 2 <= FUSION_TUNE_STEPS <= VALIDATE_STEPS and 1 <= FUSION_PROFILE_STEPS <= FUSION_TUNE_STEPS")
        if not math.isfinite(self.minimum_gain) or not 0 <= self.minimum_gain < 1:
            raise ValueError("FUSION_MIN_GAIN must be finite in [0,1)")
        names = os.environ.get("DFLASH_PREFIX_FUSION_CANDIDATES", ",".join(VARIANTS)).split(",")
        if not names or any(x not in VARIANTS for x in names) or len(set(names)) != len(names):
            raise ValueError("Invalid/duplicate FUSION_CANDIDATES")
        self.names = names if mode == "auto" else [mode]
        self.graphs = {"paired": NPUGraphPrefixSelector(head, PairedHistoryWalk, logger)}
        for name in self.names:
            self.graphs[name] = NPUGraphPrefixSelector(
                head, partial(FusedPairedHistoryWalk, variant=name), logger)
        self.rejected = {}
        self.max_errors = {name: 0.0 for name in self.names}
        self.timings = {name: [] for name in self.graphs}
        self.checked = 0
        self.winner = "paired" if mode == "auto" else mode
        self.frozen = False
        self.report_path = os.environ.get("DFLASH_PREFIX_FUSION_REPORT")
        self.logger.info("PREFIX_FUSION mode=%s candidates=%s tuning_blocks=%d profile_blocks=%d "
                         "minimum_gain=%.3f; profiling is excluded from benchmark warmup",
                         mode, self.names, self.tune_steps, self.profile_steps, self.minimum_gain)

    def _call(self, name, args, kwargs, scores=False):
        return self.graphs[name].select(*args, **kwargs, return_scores=scores)

    def _time(self, name, args, kwargs):
        synchronize(args[0].device)
        start = time.perf_counter()
        output = self._call(name, args, kwargs)
        synchronize(args[0].device)
        self.timings[name].append((time.perf_counter() - start) * 1000)
        return output

    def _reject(self, name, reason, error):
        self.rejected[name] = {"reason": reason, "max_score_error_at_rejection": error,
                               "block": self.checked + 1}
        self.logger.warning("PREFIX_FUSION_REJECT variant=%s reason=%s block=%d max_score_error=%s",
                            name, reason, self.checked + 1, error)
        if self.mode != "auto":
            raise RuntimeError(f"PREFIX_PARITY_FAIL: fusion={name} {reason}. Restart with FUSION=off; do not benchmark this variant.")

    def _finish(self, args):
        import triton
        medians = {name: statistics.median(values) for name, values in self.timings.items()
                   if values and name not in self.rejected}
        if self.mode == "auto":
            self.winner = min(medians, key=medians.get)
            gain = medians["paired"] / medians[self.winner] - 1
            if gain < self.minimum_gain:
                self.winner = "paired"
        # Explicit selection never disguises slower results as an optimization.
        report = {
            "mode": self.mode, "selected": self.winner, "pid": os.getpid(),
            "tuning_blocks": self.checked, "profile_blocks": self.profile_steps,
            "batch": args[1].shape[0], "proposals": args[1].shape[1],
            "top_k": args[1].shape[2], "rank": self.head.rank,
            "dtype": str(args[0].dtype), "torch_version": str(torch.__version__),
            "triton_version": str(getattr(triton, "__version__", "unknown")),
            "triton_path": str(triton.__file__),
            "draft_model": os.environ.get("DFLASH_PREFIX_FUSION_DRAFT"),
            "rejected": self.rejected, "max_score_error": self.max_errors,
            "selector_ms_median": medians,
            "selector_ms_mean": {name: statistics.mean(values) for name, values in self.timings.items() if values},
            "isolated_speedup_vs_paired": medians["paired"] / medians[self.winner],
            "weights_unchanged": True, "full_proposal_count_retained": True,
            "note": "Warmup-only isolated selector timing. Capture/compilation excluded. "
                    "Not end-to-end throughput or proof on unseen inputs. Run the full evaluation in MODE=validate.",
        }
        draft_path = report["draft_model"]
        if draft_path:
            cfg = Path(draft_path) / "config.json"
            report["draft_config_sha256"] = hashlib.sha256(cfg.read_bytes()).hexdigest()
        self.frozen = True
        self.logger.info("PREFIX_FUSION_RESULT %s", json.dumps(report, sort_keys=True))
        if self.report_path:
            # PID suffix prevents separate workers from silently replacing a report.
            destination = Path(self.report_path + f".{os.getpid()}.json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
        # Keep graph objects alive for the process lifetime: no graph-pool or
        # pointer-lifetime changes while vLLM may still retain previous outputs.

    @torch.no_grad()
    def select(self, *args, embedding_weight, return_scores=False):
        kwargs = {"embedding_weight": embedding_weight}
        if self.frozen:
            return self._call(self.winner, args, kwargs, return_scores)
        if not return_scores:
            if self.checked == 0:
                raise RuntimeError("Fusion must receive a scored reference check before production replay")
            return self._call(self.winner, args, kwargs)

        expected, expected_scores = self._call("paired", args, kwargs, True)
        baseline_tokens = self._call("paired", args, kwargs)
        if not torch.equal(expected, baseline_tokens) or not bool(torch.isfinite(expected_scores).all()):
            raise RuntimeError("PREFIX_PARITY_FAIL: working paired graph score/token-only mismatch or nonfinite")
        valid = ["paired"]
        for name in self.names:
            if name in self.rejected:
                continue
            # Capture/JIT failures are fatal, not silently retried on a possibly
            # unhealthy device stream. Use FUSION=off after fixing/restarting.
            scored, scores = self._call(name, args, kwargs, True)
            production = self._call(name, args, kwargs)
            reason, error = check_result(expected, expected_scores, scored, scores, production)
            if error is not None:
                self.max_errors[name] = max(self.max_errors[name], error)
            if reason:
                self._reject(name, reason, error)
            else:
                valid.append(name)
        if self.checked < self.profile_steps:
            # Warmed graphs; rotate ordering across independent real blocks.
            shift = self.checked % len(valid)
            for name in valid[shift:] + valid[:shift]:
                token = self._time(name, args, kwargs)
                if not torch.equal(expected, token):
                    if name == "paired":
                        raise RuntimeError("PREFIX_PARITY_FAIL: paired timed replay mismatch")
                    self._reject(name, "timed_token_mismatch", None)
        self.checked += 1
        if self.checked == self.tune_steps:
            self._finish(args)
        return self._call(self.winner, args, kwargs, True)
