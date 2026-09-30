"""Fused choices with CachedPrefixWalk's individual history-operation shapes.

Paired BF16 history changed a score on the supplied NPU failure snapshot.
Keep history queries, softmax lengths, output projections, FFNs and final score
projection at one proposal per call. Candidate K/V and first-query caching,
actual chosen-token dependencies, graph capture and vector selection remain.
"""
import math

import torch

from .fused_history_v2 import FusedPairedHistoryWalk


class FusedUnpairedHistoryWalk(FusedPairedHistoryWalk):
    def __init__(self, head, variant="local"):
        if variant not in {"select", "local"}:
            raise ValueError("Unpaired fusion supports select/local with native softmax only")
        super().__init__(head, variant)

    def _history_scores(self, queries, first_q, storage, start, end):
        if end != start + 1:
            raise ValueError("Unpaired history requires exactly one proposal per call")
        head = self.head
        n, _, k, _ = queries.shape
        i = start
        heads = head.prefix_layers[0].heads
        dim = head.rank // heads
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
        return head.history_scale * head.prefix_output(query - original).squeeze(-1).float()

    @torch.no_grad()
    def walk(self, prepared, *, forced_indices=None, return_scores=False, projected_kv=None):
        head = self.head
        if head.training:
            raise RuntimeError("FusedUnpairedHistoryWalk must not be used for training")
        ids, unary, context, successors, queries, predecessors, memories, prev, anchor = prepared
        n, b, k = ids.shape
        if n != 1:
            raise ValueError("Fused unpaired selector requires batch=1")
        if b < 1 or b > head.max_proposals or k != head.top_k:
            raise ValueError("Proposal/shortlist shape disagrees with checkpoint")
        if head.history_scale == 0 or b == 1:
            return super().walk(prepared, forced_indices=forced_indices,
                                return_scores=return_scores, projected_kv=projected_kv)
        context, successors, predecessors, prev = (
            tensor.float() for tensor in (context, successors, predecessors, prev))
        storage = self._buffers(prepared)
        heads = head.prefix_layers[0].heads
        dim = head.rank // heads
        # Keep the original projection batch shapes/operations unchanged.
        if projected_kv is None:
            source = torch.cat((anchor[:, None], memories[:, :-1].reshape(
                n, (b - 1) * k, head.rank)), dim=1)
            projected = torch.stack([
                torch.stack((layer.key_projection(source), layer.value_projection(source)), dim=2)
                for layer in head.prefix_layers
            ], dim=2).reshape(n, 1 + (b - 1) * k,
                             len(head.prefix_layers), 2, heads, dim).float()
        else:
            shape = (n, 1 + (b - 1) * k, len(head.prefix_layers), 2, heads, dim)
            if tuple(projected_kv.shape) != shape or projected_kv.dtype != torch.float32:
                raise ValueError("Invalid precomputed candidate K/V layout or precision")
            projected = projected_kv
        storage[:, 1].copy_(projected[:, 0])
        choices = projected[:, 1:].reshape(n, b - 1, k, -1)
        first = head.prefix_layers[0]
        first_q = first.query_projection(first.query_norm(queries)).reshape(
            n, b, k, heads, dim).float()
        selected = torch.empty((n, b), dtype=ids.dtype, device=ids.device)
        scores = torch.empty((n, b, k), dtype=torch.float32, device=ids.device) if return_scores else None
        previous_rows = torch.empty((b, head.rank), dtype=torch.float32, device=ids.device)

        def choose(i, history_score):
            nonlocal prev
            local = None
            if not self.fuse_local:
                with torch.autocast(device_type=unary.device.type, enabled=False):
                    local = (context[:, i, None] * prev[:, None] * successors[:, i]).sum(-1) / math.sqrt(head.rank)
            self.choose_commit(
                unary=unary[0, i], local=local[0] if local is not None else None,
                history=history_score[0] if isinstance(history_score, torch.Tensor) else None,
                context=context[0, i], previous=prev[0], successors=successors[0, i],
                ids=ids[0, i], predecessors=predecessors[0, i],
                choices=choices[0, i] if i + 1 < b else None,
                storage=storage[0, i + 2].reshape(-1) if i + 1 < b else None,
                next_previous=previous_rows[i] if i + 1 < b else previous_rows[i, :0],
                selected=selected[0, i:i+1], scores=scores[0, i] if return_scores else None,
                correction_scale=head.correction_scale, fuse_local=self.fuse_local,
                forced=forced_indices[0, i:i+1] if forced_indices is not None else None,
            )
            if i + 1 < b:
                prev = previous_rows[i:i+1]

        # The first proposal has exactly zero history correction.
        choose(0, 0.0)
        for start in range(1, b):
            end = start + 1
            history = self._history_scores(queries, first_q, storage, start, end)
            for i in range(start, end):
                choose(i, history[:, i - start])
        return (selected, scores) if return_scores else selected
