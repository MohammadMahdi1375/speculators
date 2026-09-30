"""Capture original per-block preparation and the unchanged cached prefix walk.

Serving only. Fixed batch=1, shape, stride, dtype, device, caller stream and
weights. No vocabulary cache, approximation, confidence cutoff or skipped
verification. Separate graph-private walkers prevent scratch/output aliasing.
Both graphs are built on the first call, outside steady-state measurement.
"""
import time

import torch

from .npu_graph_v2 import _npu_backend, _signature


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # An inference tensor may not expose a version counter.
        return None


def _weight_identity(weight):
    return (id(weight), weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
            weight.dtype, weight.device, _tensor_version(weight))


class NPUGraphPrefixSelector:
    def __init__(self, head, walker_factory, logger, warmup_steps=3):
        if head.training:
            raise ValueError("Combined prefix graph requires eval mode")
        if not 1 <= warmup_steps <= 10:
            raise ValueError("Graph warmup steps must be in [1, 10]")
        self.head, self.logger = head, logger
        self.warmup_steps = warmup_steps
        self.walkers = {False: walker_factory(head), True: walker_factory(head)}
        self.graphs, self.outputs = {}, {}
        self.static = self.backend = self.signature = None
        self.owner_stream = self.capture_stream = None
        self.embedding_weight = self.embedding_identity = None
        self.scales = self.execution = None
        self.replays = {False: 0, True: 0}
        self.failed = False

    def _execution(self, device):
        # This is a metadata guard, not a device synchronization. An outer
        # autocast policy must not change underneath an already captured graph.
        enabled = torch.is_autocast_enabled(device.type)
        return (torch.is_inference_mode_enabled(), enabled,
                torch.get_autocast_dtype(device.type) if enabled else None)

    def _operation(self, return_scores):
        table = self.head.prepare_tables(*self.static,
                                         embedding_weight=self.embedding_weight)
        return self.walkers[return_scores].walk(table, return_scores=return_scores)

    @torch.no_grad()
    def _capture(self, inputs, embedding_weight):
        hidden, ids, unary, anchors = inputs
        if ids.ndim != 3 or ids.shape[0] != 1:
            raise ValueError("Combined prefix graph requires batch=1 / --max-num-seqs 1")
        if (hidden.ndim != 3 or hidden.shape[:2] != ids.shape[:2]
                or hidden.shape[-1] != self.head.hidden_size
                or unary.shape != ids.shape or anchors.shape != (1,)
                or ids.shape[-1] != self.head.top_k
                or not 1 <= ids.shape[1] <= self.head.max_proposals):
            raise ValueError("Invalid combined prefix graph input shapes")
        if ids.dtype not in (torch.int32, torch.int64) or anchors.dtype not in (torch.int32, torch.int64):
            raise ValueError("Candidate/anchor IDs must be integer tensors")
        if not hidden.is_floating_point() or not unary.is_floating_point():
            raise ValueError("Hidden states and unary logits must be floating point")
        if tuple(embedding_weight.shape) != (self.head.vocab_size, self.head.hidden_size):
            raise ValueError("Matching full target embedding weight is required")
        if any(t.device != hidden.device for t in (*inputs, embedding_weight)):
            raise ValueError("All combined graph inputs/embedding must share one device")
        self.backend = _npu_backend(hidden.device)
        backend = self.backend
        self.signature = _signature(inputs)
        self.static = tuple(t.detach().clone(memory_format=torch.preserve_format) for t in inputs)
        if _signature(self.static) != self.signature:
            raise ValueError("Combined graph input cloning changed strides; use GRAPH_SCOPE=walk")
        self.embedding_weight = embedding_weight  # Persistent model weight, never copied per block.
        self.embedding_identity = _weight_identity(embedding_weight)
        self.scales = (self.head.correction_scale, self.head.history_scale)
        self.execution = self._execution(hidden.device)
        self.owner_stream = backend.current_stream()
        self.capture_stream = backend.Stream()
        self.capture_stream.wait_stream(self.owner_stream)
        start = time.perf_counter()
        try:
            # Warm shape-dependent masks, kernel caches and private history
            # buffers before capture. Original GEMM batching is preserved.
            for return_scores in (True, False):
                with backend.stream(self.capture_stream):
                    for _ in range(self.warmup_steps):
                        self._operation(return_scores)
                backend.synchronize()
                graph = backend.NPUGraph()
                with backend.graph(graph, stream=self.capture_stream):
                    output = self._operation(return_scores)
                self.owner_stream.wait_stream(self.capture_stream)
                backend.synchronize()
                self.graphs[return_scores] = graph
                self.outputs[return_scores] = output
        except Exception as exc:
            self.failed = True
            self.graphs.clear()
            raise RuntimeError(
                "PREFIX_COMBINED_GRAPH_CAPTURE_FAILED: restart with "
                "DFLASH_PREFIX_GRAPH_SCOPE=walk to use the previous graph. "
                "No silent fallback. Original error: " + str(exc)
            ) from exc
        self.logger.info(
            "PREFIX_COMBINED_GRAPH_CAPTURED batch=%d proposals=%d top_k=%d "
            "dtype=%s capture_ms=%.1f raw_input_copies=4 score_graph=True "
            "token_only_graph=True; original preparation and scoring retained",
            *ids.shape, hidden.dtype, (time.perf_counter() - start) * 1000,
        )

    @torch.no_grad()
    def select(self, hidden, ids, unary, anchors, *, embedding_weight, return_scores=False):
        if self.failed:
            raise RuntimeError("Previous combined graph capture failed; restart serving")
        if self.head.training:
            raise RuntimeError("Combined prefix graph cannot be used for training")
        inputs = (hidden, ids, unary, anchors)
        if not self.graphs:
            self._capture(inputs, embedding_weight)
        if _signature(inputs) != self.signature:
            raise ValueError("Combined graph input shape/layout/precision changed; restart serving")
        if _weight_identity(embedding_weight) != self.embedding_identity:
            raise ValueError("Combined graph embedding weight changed; restart serving")
        if (self.head.correction_scale, self.head.history_scale) != self.scales:
            raise ValueError("Prefix inference scales changed after capture; restart serving")
        if self._execution(hidden.device) != self.execution:
            raise ValueError("Combined graph inference/autocast mode changed; restart serving")
        if self.backend.current_stream() != self.owner_stream:
            raise RuntimeError("Combined prefix graph caller stream changed; one serialized stream is required")
        # Refresh data at the captured addresses. No .item(), CPU copy or device
        # synchronize occurs in this steady-state path.
        for destination, source in zip(self.static, inputs):
            destination.copy_(source)
        self.graphs[return_scores].replay()
        self.replays[return_scores] += 1
        output = self.outputs[return_scores]
        # vLLM may retain these IDs after another replay; never return mutable
        # graph-owned output storage directly.
        if return_scores:
            return output[0].clone(), output[1].clone()
        return output.clone()
