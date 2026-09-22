"""Concurrent vLLM prefills of actual proposals, using DSpark's file connector.

Only frozen clean-sequence features may be cached by the data loader. Branch
features are collected from the current draft every update and deleted after
reading. The connector asks for one output token to export prefill features;
that output is not used as a new training response.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hs_connectors.transfer import FileTransfer
from openai import OpenAI

from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import generate_hidden_states


class BranchTarget:
    def __init__(
        self,
        endpoint,
        model,
        layers,
        hidden_size,
        *,
        concurrency=8,
        timeout=300,
        transfer=None,
        client=None,
    ):
        if concurrency < 1 or timeout <= 0:
            raise ValueError("Positive concurrency and timeout required")
        self.client = client or OpenAI(
            base_url=endpoint, api_key="EMPTY", max_retries=0, timeout=timeout
        )
        # This client only reads/deletes server-returned paths, never its cache.
        self.transfer = transfer or FileTransfer(Path.cwd() / "unused-branch-cache")
        self.model, self.layers, self.hidden_size = model, layers, hidden_size
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=concurrency)

    def submit(self, tokens, anchor, proposals):
        if anchor < 1 or proposals < 1 or len(tokens) != anchor + proposals + 1:
            raise ValueError("Expected full prefix, anchor, and actual proposal IDs")
        return self.executor.submit(self._read, list(tokens), anchor, proposals)

    def _read(self, tokens, anchor, proposals):
        handle = generate_hidden_states(
            self.client,
            self.model,
            {"input_ids": tokens},
            timeout=self.timeout,
            max_retries=2,
        )
        try:
            payload = self.transfer.get_generated(handle)
            if payload is None:
                raise RuntimeError("vLLM returned no hidden states")
            check_hidden_states(payload, tokens)
            states = payload["hidden_states"]
            expected = (len(tokens), self.layers + 1, self.hidden_size)
            if tuple(states.shape) != expected:
                raise ValueError(f"Server layer mismatch: {states.shape} != {expected}")
            # One-token causal shift: state at anchor scores proposal 0.
            # Last connector slot is BEFORE target final RMSNorm.
            return states[anchor : anchor + proposals, -1].clone()
        finally:
            self.transfer.delete(handle)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.client.close()
