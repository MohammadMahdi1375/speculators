"""Opt-in serving of compiled_prefix_v3 with the existing NPU graph API.

This compares graph replay against eager execution of the *new* head. It does
not claim equivalence to local_prefix_v2, improved tau, or measured NPU speed.
No trained teacher or target-verification shortcut is used during serving.
"""

import os

import torch

from .npu_graph_selector_v2 import NPUGraphPrefixSelector, _weight_identity


class _CompiledWalk:
    def __init__(self, head):
        self.head = head

    def walk(self, prepared, *, return_scores=False):
        return self.head.walk(prepared, return_scores=return_scores)


class CompiledPrefixGraph(NPUGraphPrefixSelector):
    """Capture the candidate-table preparation and full greedy walk together."""

    def __init__(self, head, logger):
        super().__init__(head, _CompiledWalk, logger)
        self.parameter_identity = None
        self.logit_cap = head.compiled_logit_cap

    def _capture(self, inputs, embedding_weight):
        self.parameter_identity = tuple(
            _weight_identity(parameter) for parameter in self.head.parameters()
        )
        super()._capture(inputs, embedding_weight)
        self.logger.info(
            "COMPILED_PREFIX_GRAPH_CAPTURED kind=compiled_prefix_v3 "
            "scope=selector raw_input_copies=4 score_and_token_graphs=True; "
            "includes candidate-table preparation and every proposal"
        )

    def select(self, *args, **kwargs):
        if self.head.compiled_logit_cap != self.logit_cap:
            raise RuntimeError("Compiled prefix logit cap changed; restart serving")
        if self.parameter_identity is not None and self.parameter_identity != tuple(
            _weight_identity(parameter) for parameter in self.head.parameters()
        ):
            raise RuntimeError("Compiled prefix weights changed after capture; restart serving")
        return super().select(*args, **kwargs)


class CompiledPrefixInferenceRuntime:
    """Eager reference, continuous validation, or initially validated replay.

Graph mode uses one fixed batch of one and the existing combined-graph input
and stream guards. Fast replay has no correctness reductions or host/device
synchronization after initial validation; CPU metadata guards remain enabled.
"""

    def __init__(self, head, logger):
        if head.training:
            raise ValueError("Compiled prefix serving requires head.eval()")
        self.head, self.logger = head, logger
        self.mode = os.environ.get("DFLASH_PREFIX_INFERENCE", "reference")
        if self.mode not in {"reference", "validate", "fast"}:
            raise ValueError("DFLASH_PREFIX_INFERENCE must be reference, validate, or fast")
        graph_flag = os.environ.get("DFLASH_PREFIX_NPU_GRAPH", "0")
        if graph_flag not in {"0", "1"}:
            raise ValueError("DFLASH_PREFIX_NPU_GRAPH must be 0 or 1")
        if os.environ.get("DFLASH_PREFIX_SCORE_CHECK", "strict") != "strict":
            raise ValueError("compiled_prefix_v3 requires DFLASH_PREFIX_SCORE_CHECK=strict")
        if os.environ.get("DFLASH_PREFIX_TOKEN_CACHE", "0") != "0":
            raise ValueError("compiled_prefix_v3 does not use the v2 token projection cache")
        if os.environ.get("DFLASH_PREFIX_FUSION", "off") != "off":
            raise ValueError("Set DFLASH_PREFIX_FUSION=off; v2 fused kernels cannot run compiled_prefix_v3")
        if os.environ.get("DFLASH_PREFIX_HISTORY_GROUP_SIZE", "1") != "1":
            raise ValueError("Set DFLASH_PREFIX_HISTORY_GROUP_SIZE=1 for compiled_prefix_v3")
        if int(os.environ.get("DFLASH_PREFIX_PROFILE_STEPS", "0")) != 0:
            raise ValueError("Use DFLASH_PREFIX_SELECTOR_PROFILE_STEPS for compiled diagnostics")
        profile_steps = int(os.environ.get("DFLASH_PREFIX_SELECTOR_PROFILE_STEPS", "0"))
        if not 0 <= profile_steps <= 32:
            raise ValueError("DFLASH_PREFIX_SELECTOR_PROFILE_STEPS must be in [0, 32]")
        self.check_steps = int(os.environ.get("DFLASH_PREFIX_VALIDATE_STEPS", "32"))
        if self.check_steps < 1:
            raise ValueError("DFLASH_PREFIX_VALIDATE_STEPS must be positive")
        if graph_flag == "1" and self.mode != "reference":
            if os.environ.get("DFLASH_PREFIX_GRAPH_SCOPE", "selector") != "selector":
                raise ValueError("compiled_prefix_v3 graph requires GRAPH_SCOPE=selector")
            self.graph = CompiledPrefixGraph(head, logger)
        else:
            self.graph = None
        self.checked = 0
        self.max_score_error = 0.0
        self.failed = False
        self.profiler = None
        if profile_steps:
            if self.graph is None or self.mode != "fast":
                raise ValueError("Compiled profiling requires fast mode with the NPU graph enabled")
            from .compiled_profile import CompiledSelectorProfiler
            self.profiler = CompiledSelectorProfiler(
                head, self.graph, logger, steps=profile_steps,
                output_dir=os.environ.get("DFLASH_PREFIX_PROFILE_DIR"),
            )
        self.logger.info(
            "COMPILED_PREFIX_INFERENCE kind=compiled_prefix_v3 mode=%s "
            "npu_graph=%s scope=selector check_steps=%d token_cache=False "
            "legacy_fusion=False; experimental architecture, full target verification",
            self.mode, self.graph is not None, self.check_steps,
        )

    def _eager(self, hidden, ids, unary, anchors, embedding_weight, return_scores):
        prepared = self.head.prepare_tables(
            hidden, ids, unary, anchors, embedding_weight=embedding_weight
        )
        return self.head.walk(prepared, return_scores=return_scores)

    @torch.no_grad()
    def select(self, hidden, ids, unary, anchors, *, embedding_weight):
        if self.failed:
            raise RuntimeError("Compiled prefix validation previously failed; restart serving")
        if self.head.training:
            raise RuntimeError("Compiled prefix serving cannot run in training mode")
        inputs = (hidden, ids, unary, anchors)
        if self.graph is None:
            # Reference mode never captures/replays a graph. Without a graph,
            # validate/fast also use this same eager implementation and make no
            # optimization-parity or steady-state-fast claim.
            return self._eager(*inputs, embedding_weight, False)
        checking = self.mode == "validate" or self.checked < self.check_steps
        if not checking:
            actual = self.graph.select(*inputs, embedding_weight=embedding_weight)
            if self.profiler is not None and not self.profiler.done:
                self.profiler.observe(inputs, embedding_weight, actual)
            return actual

        try:
            expected, expected_scores = self._eager(*inputs, embedding_weight, True)
            scored, actual_scores = self.graph.select(
                *inputs, embedding_weight=embedding_weight, return_scores=True
            )
            actual = self.graph.select(*inputs, embedding_weight=embedding_weight)
            finite = bool(
                torch.isfinite(expected_scores).all()
                and torch.isfinite(actual_scores).all()
            )
            if not finite:
                raise RuntimeError("COMPILED_PREFIX_PARITY_FAIL: nonfinite scores")
            error = (expected_scores - actual_scores).abs().max().item()
            self.max_score_error = max(self.max_score_error, error)
            if not torch.equal(expected, scored) or not torch.equal(expected, actual):
                raise RuntimeError(
                    "COMPILED_PREFIX_PARITY_FAIL: eager, scored replay, and "
                    f"production replay tokens differ; max_score_error={error:.6g}"
                )
            if not torch.allclose(expected_scores, actual_scores, rtol=1e-5, atol=1e-5):
                raise RuntimeError(
                    f"COMPILED_PREFIX_PARITY_FAIL: score discrepancy={error:.6g}"
                )
        except Exception:
            self.failed = True
            raise
        self.checked += 1
        if self.checked == self.check_steps or self.checked % 256 == 0:
            self.logger.info(
                "COMPILED_PREFIX_PARITY_PASS kind=compiled_prefix_v3 "
                "checked_blocks=%d max_score_error=%.6g mode=%s npu_graph=True "
                "scope=selector production_tokens_checked=True%s",
                self.checked, self.max_score_error, self.mode,
                "; steady-state fast path enabled" if self.mode == "fast"
                else "; validation continues",
            )
        return actual
