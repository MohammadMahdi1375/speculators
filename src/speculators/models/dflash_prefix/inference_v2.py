"""Optional eager inference cache for the existing local_prefix_v2 checkpoint.

No trainable tensors, changed candidate lists, approximation to the scoring
function, or change to target verification. Serving only; buffers are scratch
space and are NOT checkpoint state. Instantiate after loading/sharing weights.
The instance is for one serialized model worker, not concurrent Python threads.
"""

import math
import os
import json
import statistics
import time

import torch

from .selector_v2 import LocalPrefixSelector
from .token_cache_v2 import TokenProjectionCache, synchronize


class CachedPrefixWalk:
    def __init__(self, head):
        if not isinstance(head, LocalPrefixSelector):
            raise TypeError("CachedPrefixWalk requires local_prefix_v2")
        if head.training:
            raise RuntimeError("CachedPrefixWalk is inference-only; call eval() first")
        self.head = head  # plain object: never registers/duplicates model parameters
        self._signature = None
        self._history = None
        self._biases = None
        self._masks = None

    def _buffers(self, prepared):
        head = self.head
        n, b, k = prepared.candidate_ids.shape
        query = prepared.queries
        heads = head.prefix_layers[0].heads
        dim = head.rank // heads
        signature = (n, b, k, query.device, query.dtype)
        if self._signature != signature:
            # [batch, source, layer, key/value, head, head_dim]. Source zero is
            # null, source one is the anchor, then the actually selected tokens.
            self._history = torch.empty(
                n, b + 1, len(head.prefix_layers), 2, heads, dim,
                device=query.device, dtype=torch.float32,
            )
            self._biases, self._masks = [], []
            for i in range(b):
                # Match the reference including its masked immediate predecessor.
                distances = torch.tensor(
                    [0] + list(range(i + 1, 0, -1)),
                    device=query.device, dtype=torch.long,
                )
                allowed = torch.tensor(
                    [True] + [distance >= 2 for distance in range(i + 1, 0, -1)],
                    device=query.device, dtype=torch.bool,
                )
                self._masks.append(~allowed[None, None, None, :])
                self._biases.append(tuple(
                    layer.distance_bias[:, distances].float()[None, :, None, :]
                    for layer in head.prefix_layers
                ))
            self._signature = signature
        # Never retain a preceding request's selected tokens. Every consumed
        # slot is initialized now or filled by this walk before it is read.
        self._history[:, 0].zero_()
        return self._history

    @torch.no_grad()
    def walk(self, prepared, *, forced_indices=None, return_scores=False, projected_kv=None):
        head = self.head
        if head.training:
            raise RuntimeError("CachedPrefixWalk must not be used for training")
        ids, unary, context, successors, queries, predecessors, memories, prev, anchor = prepared
        n, b, k = ids.shape
        if b < 1 or b > head.max_proposals or k != head.top_k:
            raise ValueError("Proposal/shortlist shape disagrees with checkpoint")
        history_enabled = head.history_scale != 0 and b > 1
        # These conversions were previously repeated inside the sequential loop.
        context = context.float()
        successors = successors.float()
        predecessors = predecessors.float()
        prev = prev.float()
        if history_enabled:
            storage = self._buffers(prepared)
            heads = head.prefix_layers[0].heads
            dim = head.rank // heads
            # All candidate memories are known before choosing any token. The
            # last position need not become history inside this block.
            if projected_kv is None:
                source = torch.cat((anchor[:, None], memories[:, :-1].reshape(
                    n, (b - 1) * k, head.rank)), dim=1)
                projected = torch.stack([
                    torch.stack((layer.key_projection(source),
                                 layer.value_projection(source)), dim=2)
                    for layer in head.prefix_layers
                ], dim=2).reshape(n, 1 + (b - 1) * k,
                                 len(head.prefix_layers), 2, heads, dim).float()
            else:
                expected_shape = (n, 1 + (b-1)*k, len(head.prefix_layers), 2, heads, dim)
                if tuple(projected_kv.shape) != expected_shape or projected_kv.dtype != torch.float32:
                    raise ValueError("Invalid precomputed candidate K/V layout or precision")
                projected = projected_kv
            storage[:, 1].copy_(projected[:, 0])
            choices = projected[:, 1:].reshape(n, b - 1, k, -1)
            # Only layer zero's queries are independent across proposal positions.
            first = head.prefix_layers[0]
            first_q = first.query_projection(first.query_norm(queries)).reshape(
                n, b, k, heads, dim).float()

        selected, scores = [], []
        for i in range(b):
            with torch.autocast(device_type=unary.device.type, enabled=False):
                local = (context[:, i, None] * prev[:, None]
                         * successors[:, i]).sum(-1) / math.sqrt(head.rank)
            # At position one, the original history result is multiplied by
            # has_history=False. Skip this provably zero contribution entirely.
            history_score = 0.0
            if history_enabled and i > 0:
                original = queries[:, i:i + 1]
                query = original
                for li, layer in enumerate(head.prefix_layers):
                    if li == 0:
                        q = first_q[:, i].transpose(1, 2)
                    else:
                        q = layer.query_projection(layer.query_norm(query)).reshape(
                            n, k, heads, dim).transpose(1, 2).float()
                    key = storage[:, :i + 2, li, 0].transpose(1, 2)
                    value = storage[:, :i + 2, li, 1].transpose(1, 2)
                    with torch.autocast(device_type=query.device.type, enabled=False):
                        logits = q @ key.transpose(-1, -2)
                        logits = logits / math.sqrt(dim) + self._biases[i][li]
                        weights = logits.masked_fill(self._masks[i], -torch.inf).softmax(-1)
                        attended = weights @ value
                    attended = attended.transpose(1, 2).reshape(
                        n, 1, k, head.rank).to(query.dtype)
                    query = query + layer.output_projection(attended)
                    query = query + layer.ffn(layer.ffn_norm(query))
                history_score = head.history_scale * head.prefix_output(
                    query - original).squeeze(-1).float()[:, 0]
            logits = unary[:, i].float() + head.correction_scale * (local + history_score)
            index = logits.argmax(-1) if forced_indices is None else forced_indices[:, i]
            selected.append(ids[:, i].gather(-1, index[:, None]).squeeze(-1))
            if i + 1 < b:
                prev = predecessors[:, i].gather(
                    1, index[:, None, None].expand(n, 1, head.rank)).squeeze(1)
                if history_enabled:
                    # One gather for both K/V and all layers. Only the selected
                    # candidate is written into the shared history buffers.
                    chosen = choices[:, i].gather(
                        1, index[:, None, None].expand(n, 1, choices.shape[-1]))
                    storage[:, i + 2].copy_(chosen.reshape(
                        n, len(head.prefix_layers), 2, heads, dim))
            if return_scores:
                scores.append(logits)
        result = torch.stack(selected, dim=1)
        return (result, torch.stack(scores, dim=1)) if return_scores else result


class PrefixInferenceRuntime:
    """Reference, always-check, or fast with initial real-input parity checks."""

    def __init__(self, head, logger):
        self.head, self.logger = head, logger
        self.mode = os.environ.get("DFLASH_PREFIX_INFERENCE", "reference")
        if self.mode not in {"reference", "validate", "fast"}:
            raise ValueError("DFLASH_PREFIX_INFERENCE must be reference, validate, or fast")
        self.score_check = os.environ.get("DFLASH_PREFIX_SCORE_CHECK", "strict")
        if self.score_check not in {"strict", "audit"}:
            raise ValueError("DFLASH_PREFIX_SCORE_CHECK must be strict or audit")
        if self.score_check == "audit" and self.mode != "validate":
            raise ValueError("Score audit requires DFLASH_PREFIX_INFERENCE=validate (every block)")
        self.check_steps = int(os.environ.get("DFLASH_PREFIX_VALIDATE_STEPS", "32"))
        if self.check_steps < 1:
            raise ValueError("DFLASH_PREFIX_VALIDATE_STEPS must be positive")
        self.fast = CachedPrefixWalk(head)
        cache_enabled = os.environ.get("DFLASH_PREFIX_TOKEN_CACHE", "0")
        if cache_enabled not in {"0", "1"}:
            raise ValueError("DFLASH_PREFIX_TOKEN_CACHE must be 0 or 1")
        self.token_cache = None
        graph_enabled = os.environ.get("DFLASH_PREFIX_NPU_GRAPH", "0")
        if graph_enabled not in {"0", "1"}:
            raise ValueError("DFLASH_PREFIX_NPU_GRAPH must be 0 or 1")
        if graph_enabled == "1" and (cache_enabled != "0" or self.score_check != "strict" or self.mode == "reference"):
            raise ValueError("Prefix NPU graph requires TOKEN_CACHE=0, SCORE_CHECK=strict, and validate/fast mode")
        self.graph_walk = None
        self.graph_selector = None
        self.graph_scope = os.environ.get("DFLASH_PREFIX_GRAPH_SCOPE", "walk")
        if self.graph_scope not in {"walk", "selector"}:
            raise ValueError("DFLASH_PREFIX_GRAPH_SCOPE must be walk or selector")
        if self.graph_scope == "selector" and graph_enabled != "1":
            raise ValueError("GRAPH_SCOPE=selector requires DFLASH_PREFIX_NPU_GRAPH=1")
        self.history_group_size = int(os.environ.get("DFLASH_PREFIX_HISTORY_GROUP_SIZE", "1"))
        if self.history_group_size not in (1, 2):
            raise ValueError("DFLASH_PREFIX_HISTORY_GROUP_SIZE must be 1 or 2")
        if self.history_group_size == 2 and (self.graph_scope != "selector" or graph_enabled != "1"):
            raise ValueError("Paired history requires GRAPH_SCOPE=selector and NPU_GRAPH=1")
        fusion = os.environ.get("DFLASH_PREFIX_FUSION", "off")
        unpaired_fusion = fusion in {"select_unpaired", "local_unpaired"}
        if fusion != "off" and self.history_group_size != (1 if unpaired_fusion else 2):
            raise ValueError("Use HISTORY_GROUP_SIZE=1 for unpaired fusion, 2 for paired fusion")
        if unpaired_fusion and (graph_enabled != "1" or self.graph_scope != "selector"):
            raise ValueError("Unpaired fusion requires NPU_GRAPH=1 and GRAPH_SCOPE=selector")
        if graph_enabled == "1" and self.graph_scope == "selector":
            from .npu_graph_selector_v2 import NPUGraphPrefixSelector
            walker_factory = CachedPrefixWalk
            if self.history_group_size == 2:
                from .paired_history_v2 import PairedHistoryWalk
                walker_factory = PairedHistoryWalk
            self.graph_selector = NPUGraphPrefixSelector(head, walker_factory, logger)
            if unpaired_fusion:
                from functools import partial
                from .fused_unpaired_v2 import FusedUnpairedHistoryWalk
                self.graph_selector = NPUGraphPrefixSelector(
                    head, partial(FusedUnpairedHistoryWalk, variant=fusion.removesuffix("_unpaired")), logger)
                self.graph_selector.winner = fusion  # failure-report identity
                self.logger.info("PREFIX_FUSION_FIXED variant=%s history_group_size=1 native_softmax=True; "
                                 "full proposal count, strict reference validation retained", fusion)
            elif fusion != "off":
                from .fused_selector_v2 import FusedSelectorSuite
                self.graph_selector = FusedSelectorSuite(head, logger, fusion, self.check_steps)
        elif graph_enabled == "1":
            from .npu_graph_v2 import NPUGraphPrefixWalk
            self.graph_walk = NPUGraphPrefixWalk(head, CachedPrefixWalk, logger)
        if cache_enabled == "1" and self.mode != "reference":
            self.token_cache = TokenProjectionCache(
                head, chunk_tokens=int(os.environ.get("DFLASH_PREFIX_CACHE_CHUNK_TOKENS", "1024")),
                max_mib=float(os.environ.get("DFLASH_PREFIX_CACHE_MAX_MIB", "768")), logger=logger,
            )
        self.profile_steps = int(os.environ.get("DFLASH_PREFIX_PROFILE_STEPS", "0"))
        if self.profile_steps < 0 or self.profile_steps > self.check_steps:
            raise ValueError("PROFILE_STEPS must be between zero and VALIDATE_STEPS")
        self._profile_rows = []
        self.selector_profile_steps = int(os.environ.get("DFLASH_PREFIX_SELECTOR_PROFILE_STEPS", "0"))
        if not 0 <= self.selector_profile_steps <= self.check_steps:
            raise ValueError("SELECTOR_PROFILE_STEPS must be between zero and VALIDATE_STEPS")
        if self.selector_profile_steps and self.graph_selector is None:
            raise ValueError("SELECTOR_PROFILE_STEPS requires GRAPH_SCOPE=selector")
        if self.graph_selector is not None and self.profile_steps:
            raise ValueError("Combined graph profiling uses SELECTOR_PROFILE_STEPS; set PROFILE_STEPS=0")
        if fusion != "off" and self.selector_profile_steps:
            raise ValueError("Fused selector uses FUSION_PROFILE_STEPS; set SELECTOR_PROFILE_STEPS=0")
        self._selector_profile_rows = []
        self._profile_walk_graph = None
        self._profile_unpaired_graph = None
        self.checked = 0
        self.max_score_error = 0.0
        self.audit = None
        if self.score_check == "audit":
            from .numeric_audit_v2 import NumericAudit
            self.audit = NumericAudit(
                head, self.fast, logger, os.environ.get("DFLASH_PREFIX_AUDIT_DIR"),
                max_events=int(os.environ.get("DFLASH_PREFIX_AUDIT_MAX_EVENTS", "4")),
            )
            self.logger.info("PREFIX_SCORE_AUDIT enabled: score drift is recorded; token mismatch/nonfinite remains fatal. directory=%s",
                             self.audit.directory)
        self.logger.info("PREFIX_INFERENCE mode=%s, initial_check_steps=%d token_cache=%s profile_steps=%d",
                         self.mode, self.check_steps, self.token_cache is not None, self.profile_steps)
        if self.graph_selector is not None:
            self.logger.info("PREFIX_COMBINED_GRAPH enabled: scope=selector token_only_production=True "
                             "selector_profile_steps=%d; all trained proposal positions retained",
                             self.selector_profile_steps)
            self.logger.info("PREFIX_HISTORY_GROUP size=%d; local token choices remain sequential; "
                             "all history positions/layers retained", self.history_group_size)

    def _timed(self, device, operation):
        synchronize(device)
        start = time.perf_counter()
        result = operation()
        synchronize(device)
        return result, (time.perf_counter() - start) * 1000

    def _profile(self, hidden, ids, unary, anchors, embedding_weight):
        # Warm up both alternatives first. Cache construction is outside timing.
        kwargs = {"embedding_weight": embedding_weight}
        uncached = self.head.prepare_tables(hidden, ids, unary, anchors, **kwargs)
        self.fast.walk(uncached)
        cached, kv = self.token_cache.prepare(hidden, ids, unary, anchors, **kwargs)
        self.fast.walk(cached, projected_kv=kv)
        measurements = {}
        outputs = {}
        def run_uncached():
            table, measurements["uncached_prepare_ms"] = self._timed(hidden.device, lambda:
                self.head.prepare_tables(hidden, ids, unary, anchors, **kwargs))
            outputs["uncached"], measurements["uncached_walk_ms"] = self._timed(hidden.device, lambda:
                self.fast.walk(table))
        def run_cached():
            (table, kv), measurements["cached_prepare_ms"] = self._timed(hidden.device, lambda:
                self.token_cache.prepare(hidden, ids, unary, anchors, **kwargs))
            outputs["cached"], measurements["cached_walk_ms"] = self._timed(hidden.device, lambda:
                self.fast.walk(table, projected_kv=kv))
        # Alternate order to reduce a consistent hot/cold ordering bias.
        operations = (run_uncached, run_cached) if len(self._profile_rows) % 2 == 0 else (run_cached, run_uncached)
        for operation in operations:
            operation()
        if not torch.equal(outputs["uncached"], outputs["cached"]):
            if self.audit is not None:
                self.audit.profile_failure(uncached, cached, kv, outputs)
            raise RuntimeError("PREFIX_PARITY_FAIL: profiling alternatives selected different tokens")
        self._profile_rows.append(measurements)
        if len(self._profile_rows) == self.profile_steps:
            report = {"blocks": self.profile_steps, "batch": hidden.shape[0],
                      "proposals": hidden.shape[1], "token_cache_mib":
                      self.token_cache.table.numel() * self.token_cache.table.element_size() / 1024**2,
                      "note": "Synchronized isolated selector timings during warmup; excludes target, draft backbone and LM-head/top-k. Not end-to-end throughput."}
            for name in measurements:
                values = [row[name] for row in self._profile_rows]
                report[name + "_mean"] = statistics.mean(values)
                report[name + "_median"] = statistics.median(values)
            old = report["uncached_prepare_ms_mean"] + report["uncached_walk_ms_mean"]
            new = report["cached_prepare_ms_mean"] + report["cached_walk_ms_mean"]
            report["isolated_head_speedup"] = old / new if new else None
            self.logger.info("PREFIX_CACHE_PROFILE %s", json.dumps(report, sort_keys=True))
            if self.audit is not None:
                self.audit.record_profile(report)

    @torch.no_grad()
    def select(self, hidden, ids, unary, anchors, *, embedding_weight):
        if self.graph_selector is not None:
            return self._select_combined(hidden, ids, unary, anchors, embedding_weight=embedding_weight)
        kwargs = {"embedding_weight": embedding_weight}
        checking = self.mode == "validate" or self.checked < self.check_steps
        walk = self.fast if self.graph_walk is None else self.graph_walk
        projected_kv = None
        if self.token_cache is None:
            prepared = self.head.prepare_tables(hidden, ids, unary, anchors, **kwargs)
        else:
            self.token_cache.ensure_built(embedding_weight)
            if self.audit is None and len(self._profile_rows) < self.profile_steps:
                self._profile(hidden, ids, unary, anchors, embedding_weight)
            prepared, projected_kv = self.token_cache.prepare(hidden, ids, unary, anchors, **kwargs)
        if self.mode == "reference":
            return self.head.greedy_walk(prepared)
        if checking:
            reference = prepared if self.token_cache is None else self.head.prepare_tables(
                hidden, ids, unary, anchors, **kwargs)
            expected, expected_scores = self.head.walk(reference, return_scores=True)
            actual, actual_scores = walk.walk(prepared, return_scores=True, projected_kv=projected_kv)
            finite = bool(torch.isfinite(expected_scores).all() and torch.isfinite(actual_scores).all())
            tokens_equal = torch.equal(expected, actual)
            scores_close = finite and torch.allclose(expected_scores, actual_scores, rtol=0.005, atol=0.02)
            error = (expected_scores - actual_scores).abs().max().item() if finite else None
            if self.audit is not None:
                self.audit.record(reference=reference, prepared=prepared, projected_kv=projected_kv,
                                  expected=expected, actual=actual, expected_scores=expected_scores,
                                  actual_scores=actual_scores, finite=finite, tokens_equal=tokens_equal,
                                  scores_close=scores_close, error=error)
            if not finite:
                raise RuntimeError("PREFIX_PARITY_FAIL: non-finite selector scores")
            self.max_score_error = max(self.max_score_error, error)
            if not tokens_equal:
                count = int(expected.ne(actual).sum().item())
                raise RuntimeError(
                    f"PREFIX_PARITY_FAIL: {count} selected tokens differ; max_score_error={error:.6g}. "
                    "Rerun with DFLASH_PREFIX_INFERENCE=reference. Do not benchmark this fast path."
                )
            # A token match alone should not hide a large score discrepancy.
            if not scores_close and self.score_check == "strict":
                raise RuntimeError(
                    f"PREFIX_PARITY_FAIL: score discrepancy={error:.6g}; "
                    "use DFLASH_PREFIX_INFERENCE=reference and report this error."
                )
            self.checked += 1
            if self.audit is not None:
                if self.token_cache is not None and len(self._profile_rows) < self.profile_steps:
                    self._profile(hidden, ids, unary, anchors, embedding_weight)
                if self.checked == 1 or self.checked == self.check_steps or self.checked % 256 == 0:
                    self.logger.info(
                        "PREFIX_TOKEN_PARITY checked_blocks=%d score_drift_blocks=%d max_score_error=%.6g; audit continues, no numerical parity claim",
                        self.checked, self.audit.state["score_drift_blocks"], self.max_score_error,
                    )
            elif self.checked == self.check_steps or self.checked % 256 == 0:
                self.logger.info(
                    "PREFIX_PARITY_PASS checked_blocks=%d max_score_error=%.6g mode=%s npu_graph=%s%s",
                    self.checked, self.max_score_error, self.mode, self.graph_walk is not None,
                    "; steady-state fast path enabled" if self.mode == "fast" else "; validation continues",
                )
            return actual
        return walk.walk(prepared, projected_kv=projected_kv)

    def _profile_combined(self, hidden, ids, unary, anchors, embedding_weight):
        # Optional synchronized diagnostics on the first real warmup inputs.
        # Neither graph construction nor reference validation is timed here.
        from .npu_graph_v2 import NPUGraphPrefixWalk
        from .npu_graph_selector_v2 import NPUGraphPrefixSelector
        kwargs = {"embedding_weight": embedding_weight}
        if self._profile_walk_graph is None:
            self._profile_walk_graph = NPUGraphPrefixWalk(self.head, CachedPrefixWalk, self.logger)
            table = self.head.prepare_tables(hidden, ids, unary, anchors, **kwargs)
            self._profile_walk_graph.walk(table)
        if self.history_group_size == 2 and self._profile_unpaired_graph is None:
            self._profile_unpaired_graph = NPUGraphPrefixSelector(self.head, CachedPrefixWalk, self.logger)
            self._profile_unpaired_graph.select(hidden, ids, unary, anchors, **kwargs)
        measurements, results = {}, {}

        def previous_path():
            table, measurements["eager_prepare_ms"] = self._timed(hidden.device, lambda:
                self.head.prepare_tables(hidden, ids, unary, anchors, **kwargs))
            results["previous"], measurements["previous_walk_graph_ms"] = self._timed(hidden.device, lambda:
                self._profile_walk_graph.walk(table))

        def combined_path():
            results["combined"], measurements["combined_token_graph_ms"] = self._timed(hidden.device, lambda:
                self.graph_selector.select(hidden, ids, unary, anchors, **kwargs))

        def unpaired_path():
            results["unpaired"], measurements["unpaired_combined_token_graph_ms"] = self._timed(hidden.device, lambda:
                self._profile_unpaired_graph.select(hidden, ids, unary, anchors, **kwargs))

        order = (previous_path, combined_path) if self.history_group_size == 1 else (previous_path, unpaired_path, combined_path)
        rotation = len(self._selector_profile_rows) % len(order)
        order = order[rotation:] + order[:rotation]
        for operation in order:
            operation()
        if not torch.equal(results["previous"], results["combined"]):
            raise RuntimeError("PREFIX_PARITY_FAIL: previous walk graph and combined graph selected different tokens")
        if "unpaired" in results and not torch.equal(results["unpaired"], results["combined"]):
            raise RuntimeError("PREFIX_PARITY_FAIL: paired and unpaired combined graphs selected different tokens")
        measurements["previous_selector_total_ms"] = (
            measurements["eager_prepare_ms"] + measurements["previous_walk_graph_ms"])
        self._selector_profile_rows.append(measurements)
        if len(self._selector_profile_rows) == self.selector_profile_steps:
            report = {"blocks": self.selector_profile_steps, "batch": hidden.shape[0],
                      "proposals": hidden.shape[1], "capture_excluded": True,
                      "history_group_size": self.history_group_size,
                      "note": "Synchronized isolated selector timings during warmup. Excludes draft backbone, "
                              "LM-head/top-k, verifier, and end-to-end serving. Not a throughput benchmark."}
            for name in measurements:
                values = [row[name] for row in self._selector_profile_rows]
                report[name + "_mean"] = statistics.mean(values)
                report[name + "_median"] = statistics.median(values)
            report["isolated_selector_speedup"] = (
                report["previous_selector_total_ms_mean"] / report["combined_token_graph_ms_mean"])
            if self.history_group_size == 2:
                report["paired_vs_unpaired_speedup"] = (
                    report["unpaired_combined_token_graph_ms_mean"] / report["combined_token_graph_ms_mean"])
            self.logger.info("PREFIX_SELECTOR_PROFILE %s", json.dumps(report, sort_keys=True))

    def _diagnose_combined_failure(self, raw, prepared, expected, expected_scores,
                                   scored_tokens, actual_scores, actual, reason, embedding_weight):
        # Failure-only instrumentation: never waive the caller's strict check.
        try:
            from .failure_diagnostics_v2 import collect_failure
            collect_failure(self, raw, prepared, expected, expected_scores,
                            scored_tokens, actual_scores, actual, reason, embedding_weight)
        except Exception as exc:
            self.logger.warning("PREFIX_DIAGNOSTIC_ERROR %s: %s; original parity failure retained",
                                type(exc).__name__, exc)

    @torch.no_grad()
    def _select_combined(self, hidden, ids, unary, anchors, *, embedding_weight):
        kwargs = {"embedding_weight": embedding_weight}
        checking = self.mode == "validate" or self.checked < self.check_steps
        if not checking:
            # All embedding/projection preparation happens inside replay. No
            # reference execution, score-output stack, host reduction or sync.
            return self.graph_selector.select(hidden, ids, unary, anchors, **kwargs)

        prepared = self.head.prepare_tables(hidden, ids, unary, anchors, **kwargs)
        expected, expected_scores = self.head.walk(prepared, return_scores=True)
        try:
            scored_tokens, actual_scores = self.graph_selector.select(
                hidden, ids, unary, anchors, return_scores=True, **kwargs)
            actual = self.graph_selector.select(hidden, ids, unary, anchors, **kwargs)
        except RuntimeError as exc:
            if "PREFIX_PARITY_FAIL" in str(exc):
                self._diagnose_combined_failure(
                    (hidden, ids, unary, anchors), prepared, expected, expected_scores,
                    None, None, None, "inner_graph_parity: " + str(exc), embedding_weight)
            raise
        finite = bool(torch.isfinite(expected_scores).all() and torch.isfinite(actual_scores).all())
        if not finite:
            self._diagnose_combined_failure(
                (hidden, ids, unary, anchors), prepared, expected, expected_scores,
                scored_tokens, actual_scores, actual, "nonfinite_scores", embedding_weight)
            raise RuntimeError("PREFIX_PARITY_FAIL: non-finite combined selector scores")
        error = (expected_scores - actual_scores).abs().max().item()
        self.max_score_error = max(self.max_score_error, error)
        if not torch.equal(expected, scored_tokens) or not torch.equal(expected, actual):
            self._diagnose_combined_failure(
                (hidden, ids, unary, anchors), prepared, expected, expected_scores,
                scored_tokens, actual_scores, actual, "token_mismatch", embedding_weight)
            raise RuntimeError("PREFIX_PARITY_FAIL: combined score/token-only graph selected tokens differ "
                               f"from reference; max_score_error={error:.6g}. See PREFIX_DIAGNOSTIC_RESULT; validation stopped.")
        if not torch.allclose(expected_scores, actual_scores, rtol=0.005, atol=0.02):
            self._diagnose_combined_failure(
                (hidden, ids, unary, anchors), prepared, expected, expected_scores,
                scored_tokens, actual_scores, actual, "score_drift", embedding_weight)
            raise RuntimeError(f"PREFIX_PARITY_FAIL: combined score discrepancy={error:.6g}; see PREFIX_DIAGNOSTIC_RESULT; validation stopped.")
        if len(self._selector_profile_rows) < self.selector_profile_steps:
            self._profile_combined(hidden, ids, unary, anchors, embedding_weight)
        self.checked += 1
        if self.checked == self.check_steps or self.checked % 256 == 0:
            self.logger.info(
                "PREFIX_PARITY_PASS checked_blocks=%d max_score_error=%.6g mode=%s npu_graph=True "
                "scope=selector production_tokens_checked=True%s",
                self.checked, self.max_score_error, self.mode,
                "; steady-state fast path enabled" if self.mode == "fast" else "; validation continues",
            )
        return actual
