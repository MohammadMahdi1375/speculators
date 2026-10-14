"""Bounded, opt-in timings on real compiled-selector serving inputs.

This is a diagnostic, not an optimization. It synchronizes deliberately, keeps
full target verification, and never returns a replacement token to the caller.
An ARM file written after HTTP warmup excludes vLLM's startup/dummy inputs.
Separate stage graphs only attribute costs; their sum is not the latency of
the combined production graph. No tensor values or model weights are exported.
"""

import json
import os
from pathlib import Path
import statistics
import time

import torch


class CapturedStage:
    """Diagnostic-only graph; inputs must remain at their captured addresses."""

    def __init__(self, operation, backend):
        owner = backend.current_stream()
        self.stream = backend.Stream()
        self.stream.wait_stream(owner)
        with backend.stream(self.stream):
            for _ in range(3):
                operation()
        backend.synchronize()
        self.graph = backend.NPUGraph()
        with backend.graph(self.graph, stream=self.stream):
            self.output = operation()
        owner.wait_stream(self.stream)
        backend.synchronize()

    def replay(self):
        self.graph.replay()
        return self.output


def measure(operation, synchronize):
    """Wall latency including submission and completion; retain output to sync."""
    synchronize()
    start = time.perf_counter()
    result = operation()
    synchronize()
    return result, (time.perf_counter() - start) * 1000


def tensor_bytes(tables):
    # Logical tensor payload, not allocator/graph peak memory.
    return {name: value.numel() * value.element_size()
            for name, value in zip(tables._fields, tables)
            if isinstance(value, torch.Tensor)}


def assert_tables_equal(expected, actual):
    error = 0.0
    for name, left, right in zip(expected._fields, expected, actual):
        if isinstance(left, torch.Tensor):
            if left.is_floating_point():
                if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                    raise RuntimeError(f"Nonfinite diagnostic table: {name}")
                difference = (left - right).abs().max().item() if left.numel() else 0.0
                error = max(error, difference)
                if not torch.allclose(left, right, atol=1e-5, rtol=1e-5):
                    raise RuntimeError(f"Diagnostic preparation graph differs: {name}, error={difference}")
            elif not torch.equal(left, right):
                raise RuntimeError(f"Diagnostic preparation graph changes IDs: {name}")
        elif left != right:
            raise RuntimeError(f"Diagnostic preparation changes scale: {name}")
    return error


class CompiledSelectorProfiler:
    """Profile a small number of actual calls after the production parity gate."""

    def __init__(self, head, graph, logger, *, steps, output_dir, repeats=3):
        if not 1 <= steps <= 32 or not 1 <= repeats <= 5:
            raise ValueError("Profile steps must be 1..32 and repeats 1..5")
        if not output_dir:
            raise ValueError("DFLASH_PREFIX_PROFILE_DIR is required for profiling")
        self.directory = Path(output_dir).expanduser().resolve(strict=True)
        if not self.directory.is_dir():
            raise ValueError("Profile directory must exist")
        self.head, self.graph, self.logger = head, graph, logger
        self.steps, self.repeats = steps, repeats
        self.rows = []
        self.done = False
        self.preparation = self.walk = None
        self.max_table_error = 0.0
        self.report_path = self.directory / f"selector_profile_{os.getpid()}.json"
        self.metadata = {}
        self.logger.info(
            "COMPILED_PREFIX_PROFILE_READY blocks=%d repeats=%d arm=%s report=%s; "
            "profiling timings are not serving throughput",
            steps, repeats, self.directory / "ARM", self.report_path,
        )

    def _prepare(self, inputs, embedding_weight):
        return self.head.prepare_tables(*inputs, embedding_weight=embedding_weight)

    def _initialize(self, inputs, embedding_weight):
        if self.graph.backend is None or not self.graph.graphs:
            raise RuntimeError("Production graph must be captured before profiling")
        if inputs[1].shape[0] != 1 or inputs[1].shape[1] != self.head.max_proposals:
            raise ValueError("Profile requires one request and every trained proposal")
        backend = self.graph.backend
        # Production select has just refreshed all four raw input buffers.
        self.preparation = CapturedStage(
            lambda: self._prepare(self.graph.static, embedding_weight), backend
        )
        self.walk = CapturedStage(
            lambda: self.head.walk(self.preparation.output), backend
        )
        payload = tensor_bytes(self.preparation.output)
        self.metadata = {
            "schema_version": 1, "kind": "compiled_prefix_v3",
            "measurement": "synchronized wall latency on real serving inputs",
            "input_source": "HTTP request after explicit warmup and ARM marker",
            "device": str(inputs[0].device), "dtype": str(inputs[0].dtype),
            "unary_dtype": str(inputs[2].dtype), "torch_version": str(torch.__version__),
            "batch": inputs[1].shape[0], "proposals": inputs[1].shape[1],
            "top_k": inputs[1].shape[2], "rank": self.head.rank, "heads": self.head.heads,
            "steps_requested": self.steps, "repeats_per_block": self.repeats,
            "capture_excluded": True, "parity_excluded_from_timings": True,
            "table_logical_bytes": payload, "table_logical_bytes_total": sum(payload.values()),
            "excludes": ["draft backbone", "LM head and top-k", "target verification",
                         "HTTP/client", "graph capture", "correctness comparisons"],
            "note": "Stage timings are isolated and include a completion sync. Do not sum "
                    "them into end-to-end latency or report this request as a speed benchmark.",
        }
        self.logger.info("COMPILED_PREFIX_PROFILE_STARTED blocks=%d capture_excluded=True", self.steps)

    def _write(self, status, error=None):
        names = list(self.rows[0]["timings_ms"]) if self.rows else []
        summary = {}
        for name in names:
            values = [v for row in self.rows for v in row["timings_ms"][name]]
            summary[name] = {"mean": statistics.mean(values), "median": statistics.median(values),
                             "min": min(values), "max": max(values), "samples": len(values)}
        report = {**self.metadata, "status": status, "blocks": len(self.rows),
                  "max_table_score_error": self.max_table_error,
                  "observed_tokens_equal": status == "complete",
                  "timings_ms": summary, "blocks_detail": self.rows, "error": error}
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.report_path)
        self.logger.info("COMPILED_PREFIX_SELECTOR_PROFILE %s", json.dumps({
            "status": status, "blocks": len(self.rows), "report": str(self.report_path),
            "mean_ms": {name: values["mean"] for name, values in summary.items()},
            "observed_tokens_equal": report["observed_tokens_equal"], "error": error,
        }))

    @torch.no_grad()
    def observe(self, inputs, embedding_weight, production_tokens):
        if self.done or not (self.directory / "ARM").is_file():
            return
        try:
            if self.preparation is None:
                self._initialize(inputs, embedding_weight)
            backend = self.graph.backend
            # Existing production select refreshed graph.static before this call.
            # Replay separate preparation once so the stage walk sees THIS block.
            stage_tables = self.preparation.replay()
            expected_tables = self._prepare(inputs, embedding_weight)
            self.max_table_error = max(
                self.max_table_error, assert_tables_equal(expected_tables, stage_tables)
            )
            expected = self.head.walk(expected_tables)
            stage_tokens = self.walk.replay()
            if not torch.equal(expected, production_tokens) or not torch.equal(expected, stage_tokens):
                raise RuntimeError("Diagnostic walk or production graph changes selected tokens")

            def production_select():
                return self.graph.select(*inputs, embedding_weight=embedding_weight)

            def copy_inputs():
                for destination, source in zip(self.graph.static, inputs):
                    destination.copy_(source)

            operations = {
                "prepare_eager": lambda: self._prepare(inputs, embedding_weight),
                "walk_eager": lambda: self.head.walk(expected_tables),
                # Stage replay excludes raw-input/table copies and output clones.
                "prepare_graph_replay": self.preparation.replay,
                "walk_graph_replay": self.walk.replay,
                # Production graph API: metadata guards + 4 copies + replay + ID clone.
                "combined_graph_select": production_select,
                "combined_graph_replay_only": self.graph.graphs[False].replay,
                "raw_input_copies": copy_inputs,
                "output_clone": lambda: self.graph.outputs[False].clone(),
            }
            # Warm each operation outside timing, rotate measurement order per
            # block/repeat. Inputs remain fixed throughout this diagnostic block.
            for operation in operations.values():
                operation()
            backend.synchronize()
            timings = {name: [] for name in operations}
            names = list(operations)
            for repeat in range(self.repeats):
                offset = (len(self.rows) + repeat) % len(names)
                for name in names[offset:] + names[:offset]:
                    result, milliseconds = measure(operations[name], backend.synchronize)
                    timings[name].append(milliseconds)
                    del result
            # All repeated operations must leave the production token policy intact.
            final = production_select()
            if not torch.equal(production_tokens, final):
                raise RuntimeError("Profiling changed the production graph's output")
            self.rows.append({"block": len(self.rows) + 1, "tokens_equal": True, "timings_ms": timings})
            if len(self.rows) == self.steps:
                self._write("complete")
                self.done = True
                # Release diagnostic graph buffers after synchronizing. The
                # production graph and its private buffers remain untouched.
                backend.synchronize()
                self.walk = self.preparation = None
        except Exception as error:
            self.done = True
            self._write("failed", f"{type(error).__name__}: {error}")
            raise RuntimeError(f"Compiled selector diagnostic failed; see {self.report_path}") from error
