"""Inference-only lookup of fixed token projections for local_prefix_v2.

Build after the checkpoint and shared target embeddings are loaded. No learned
state is added, no quantization is introduced, and cache entries are not saved.
Restart serving after any weight, dtype, or device change; hot reload is not
supported. One instance belongs to one serialized inference worker.
"""
import math
import time

import torch
import torch.nn.functional as F

from .selector_v2 import LocalPrefixSelector, PreparedPrefix, rms


def synchronize(device):
    if device.type == "cpu":
        return
    backend = getattr(torch, device.type, None)
    if backend is None or not hasattr(backend, "synchronize"):
        raise RuntimeError(f"No synchronize function for {device.type}")
    backend.synchronize(device)


class TokenProjectionCache:
    def __init__(self, head, *, chunk_tokens=1024, max_mib=768, logger=None):
        if not isinstance(head, LocalPrefixSelector) or head.training:
            raise ValueError("Token cache requires an eval-mode local_prefix_v2 head")
        if chunk_tokens < 1 or not math.isfinite(max_mib) or max_mib <= 0:
            raise ValueError("Cache chunk size and memory budget must be positive")
        self.head, self.logger = head, logger
        self.chunk_tokens, self.max_mib = chunk_tokens, max_mib
        self.table = None
        self._embedding_signature = None
        self.width = (4 + 2 * len(head.prefix_layers)) * head.rank
        self.build_ms = None

    def _signature(self, embedding):
        return (embedding.data_ptr(), tuple(embedding.shape), embedding.dtype, embedding.device)

    def _rows(self, embedding):
        head = self.head
        tokens = rms(embedding.detach())
        predecessor = rms(head.predecessor_projection(tokens))
        successor = head.successor_projection(tokens)
        candidate = head.candidate_projection(tokens)
        memory = rms(head.memory_projection(tokens))
        columns = [predecessor, successor, candidate, memory]
        for layer in head.prefix_layers:
            columns.extend((layer.key_projection(memory), layer.value_projection(memory)))
        return torch.cat(columns, dim=-1)

    @torch.no_grad()
    def ensure_built(self, embedding):
        if self.head.training:
            raise RuntimeError("Token cache must not be used for training")
        if tuple(embedding.shape) != (self.head.vocab_size, self.head.hidden_size):
            raise ValueError("The matching full target token embedding is required")
        signature = self._signature(embedding)
        if self.table is not None:
            if signature != self._embedding_signature:
                raise RuntimeError("Target embeddings changed: restart serving to rebuild the cache")
            return
        synchronize(embedding.device)
        start = time.perf_counter()
        first_end = min(self.chunk_tokens, self.head.vocab_size)
        first = self._rows(embedding[:first_end])
        needed = self.head.vocab_size * self.width * first.element_size()
        if needed > self.max_mib * 1024 ** 2:
            raise ValueError(
                f"Token cache needs {needed / 1024**2:.2f} MiB, budget={self.max_mib:g} MiB. "
                "TOKEN_CACHE=0 restores the previous fast path."
            )
        if self.logger:
            self.logger.info("PREFIX_TOKEN_CACHE building rows=%d width=%d dtype=%s table_mib=%.2f chunk=%d",
                             self.head.vocab_size, self.width, first.dtype,
                             needed / 1024**2, self.chunk_tokens)
        try:
            table = torch.empty(self.head.vocab_size, self.width,
                                device=first.device, dtype=first.dtype)
            table[:first_end].copy_(first)
            finite = torch.isfinite(first).all()
            del first
            for start_row in range(first_end, self.head.vocab_size, self.chunk_tokens):
                end_row = min(start_row + self.chunk_tokens, self.head.vocab_size)
                rows = self._rows(embedding[start_row:end_row])
                table[start_row:end_row].copy_(rows)
                finite.logical_and_(torch.isfinite(rows).all())
            if not finite.item():
                raise RuntimeError("Non-finite token projection cache entries")
        except RuntimeError as error:
            raise RuntimeError(
                "Token cache construction failed. TOKEN_CACHE=0 restores the previous fast path. "
                f"Original error: {error}"
            ) from error
        synchronize(embedding.device)
        self.build_ms = (time.perf_counter() - start) * 1000
        self.table, self._embedding_signature = table, signature
        if self.logger:
            self.logger.info("PREFIX_TOKEN_CACHE_READY table_mib=%.2f build_ms=%.1f; excluded from steady-state timing",
                             needed / 1024**2, self.build_ms)

    @torch.no_grad()
    def prepare(self, hidden, ids, unary, anchors, *, embedding_weight):
        self.ensure_built(embedding_weight)
        head = self.head
        if hidden.ndim != 3 or ids.ndim != 3 or hidden.shape[:2] != ids.shape[:2]:
            raise ValueError("Expected hidden [N,B,H] and candidates [N,B,K]")
        n, b, k = ids.shape
        if not (1 <= b <= head.max_proposals) or k != head.top_k or anchors.shape != (n,):
            raise ValueError("Token-cache input shape disagrees with the trained selector")
        r = head.rank
        # One wide lookup handles both the anchor and all candidate identities.
        lookup_ids = torch.cat((anchors[:, None], ids.reshape(n, b * k)), dim=1).long()
        features = F.embedding(lookup_ids, self.table)
        anchor = features[:, 0]
        candidates = features[:, 1:].reshape(n, b, k, self.width)
        context = rms(head.hidden_projection(rms(hidden)))
        if context.dtype != self.table.dtype:
            raise RuntimeError("Projection precision context changed: restart serving to rebuild the cache")
        query = rms(context.unsqueeze(-2) + candidates[..., 2*r:3*r])
        prepared = PreparedPrefix(
            ids, unary.float(), context, candidates[..., r:2*r], query,
            candidates[..., :r], candidates[..., 3*r:4*r],
            anchor[:, :r], anchor[:, 3*r:4*r],
        )
        layers = len(head.prefix_layers)
        heads = head.prefix_layers[0].heads
        dim = r // heads
        # Packed order matches CachedPrefixWalk: [N, source, layer, K/V, H, D].
        projected = torch.cat((anchor[:, None, 4*r:],
                               candidates[:, :-1, :, 4*r:].reshape(n, (b-1)*k, 2*layers*r)), dim=1)
        projected = projected.reshape(n, 1+(b-1)*k, layers, 2, heads, dim).float()
        return prepared, projected
