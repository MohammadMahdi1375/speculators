"""Optional NPUGraph replay of the existing fast prefix walk.

The vocabulary projection cache is deliberately excluded. Original per-block
preparation remains eager. Capture preserves CachedPrefixWalk's operations and
shapes; no tensor projections are moved across blocks. One fixed input shape,
one device, and one serialized caller stream per instance. No weight hot reload.
"""
import time

import torch

from .selector_v2 import PreparedPrefix


def _npu_backend(device):
    if device.type != "npu":
        raise ValueError("DFLASH_PREFIX_NPU_GRAPH=1 requires an NPU device")
    import torch_npu
    backend = torch_npu.npu
    for name in ("NPUGraph", "graph", "Stream", "stream", "current_stream", "synchronize"):
        if not hasattr(backend, name):
            raise RuntimeError(f"Installed torch_npu lacks npu.{name}; restart with DFLASH_PREFIX_NPU_GRAPH=0")
    return backend


def _signature(prepared):
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in prepared)


class NPUGraphPrefixWalk:
    def __init__(self, head, walker_factory, logger, warmup_steps=3):
        if head.training:
            raise ValueError("NPUGraph prefix walk requires eval mode")
        if not 1 <= warmup_steps <= 10:
            raise ValueError("NPU graph warmup steps must be in [1, 10]")
        self.head, self.logger = head, logger
        # Dedicated scratch buffers are retained by this graph. Eager validation
        # uses a different walker and cannot overwrite graph-private history.
        self.walker = walker_factory(head)
        self.warmup_steps = warmup_steps
        self.graph = None
        self.static = None
        self.outputs = None
        self.backend = None
        self.owner_stream = None
        self.capture_stream = None
        self.signature = None
        self.scales = None
        self.replays = 0

    @torch.no_grad()
    def _capture(self, prepared):
        if prepared.candidate_ids.shape[0] != 1:
            raise ValueError("Initial prefix NPUGraph implementation requires batch=1 / --max-num-seqs 1")
        if any(t.device != prepared.queries.device for t in prepared):
            raise ValueError("All prefix graph inputs must be on the same device")
        self.backend = _npu_backend(prepared.queries.device)
        backend = self.backend
        self.signature = _signature(prepared)
        self.scales = (self.head.correction_scale, self.head.history_scale)
        self.static = PreparedPrefix(*(t.detach().clone(memory_format=torch.preserve_format) for t in prepared))
        if _signature(self.static) != self.signature:
            raise ValueError("Prefix graph input cloning changed strides; restart with DFLASH_PREFIX_NPU_GRAPH=0")
        self.owner_stream = backend.current_stream()
        self.capture_stream = backend.Stream()
        self.capture_stream.wait_stream(self.owner_stream)
        start = time.perf_counter()
        try:
            # Initialize shape-dependent masks/history and warm up kernels before
            # capture. Tensor construction from Python lists stays outside it.
            with backend.stream(self.capture_stream):
                for _ in range(self.warmup_steps):
                    self.walker.walk(self.static, return_scores=True)
            backend.synchronize()
            graph = backend.NPUGraph()
            with backend.graph(graph, stream=self.capture_stream):
                self.outputs = self.walker.walk(self.static, return_scores=True)
            self.owner_stream.wait_stream(self.capture_stream)
            backend.synchronize()
            self.graph = graph
        except Exception as exc:
            raise RuntimeError(
                "PREFIX_NPU_GRAPH_CAPTURE_FAILED: restart with DFLASH_PREFIX_NPU_GRAPH=0 "
                "and TOKEN_CACHE=0. No silent eager fallback. Original error: " + str(exc)
            ) from exc
        self.logger.info(
            "PREFIX_NPU_GRAPH_CAPTURED batch=%d proposals=%d top_k=%d dtype=%s capture_ms=%.1f; "
            "token_cache=0, original preparation retained; parity checks follow",
            *prepared.candidate_ids.shape, prepared.queries.dtype,
            (time.perf_counter() - start) * 1000,
        )

    @torch.no_grad()
    def walk(self, prepared, *, forced_indices=None, return_scores=False, projected_kv=None):
        if self.head.training:
            raise RuntimeError("NPUGraph prefix walk cannot be used for training")
        if forced_indices is not None or projected_kv is not None:
            raise ValueError("Prefix NPUGraph supports greedy original-preparation inputs only")
        if self.graph is None:
            self._capture(prepared)
        if _signature(prepared) != self.signature:
            raise ValueError("Prefix NPUGraph input shape/layout/precision changed; restart with DFLASH_PREFIX_NPU_GRAPH=0")
        if (self.head.correction_scale, self.head.history_scale) != self.scales:
            raise ValueError("Prefix inference scales changed after capture; restart serving")
        if self.backend.current_stream() != self.owner_stream:
            raise RuntimeError("Prefix NPUGraph caller stream changed; one serialized stream is required")
        # Stable memory addresses are mandatory. Updating Python references alone
        # would replay stale candidates, unary scores, and selected-token history.
        for destination, source in zip(self.static, prepared):
            destination.copy_(source)
        self.graph.replay()
        self.replays += 1
        # A later replay must never mutate output tokens retained by vLLM.
        tokens = self.outputs[0].clone()
        if return_scores:
            return tokens, self.outputs[1].clone()
        return tokens
