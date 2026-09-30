"""Paired selector with opt-in vector fusion. All attention GEMMs retain their
original batch shapes, precision, weights, and chosen-prefix dependency.
"""
import math

import torch

from .paired_history_v2 import PairedHistoryWalk


class FusedPairedHistoryWalk(PairedHistoryWalk):
    def __init__(self, head, variant):
        super().__init__(head)
        if variant not in {"select", "local", "select_softmax", "local_softmax"}:
            raise ValueError("Unknown fused selector variant")
        if (head.top_k != 16 or head.rank not in (64, 256)
                or len(head.prefix_layers) != 2 or head.prefix_layers[0].heads != 4):
            raise ValueError("Fused v1 supports K16, rank64/256, 4 heads, 2 layers")
        self.variant = variant
        self.fuse_local = variant.startswith("local")
        self.fuse_softmax = variant.endswith("softmax")
        # Lazy import keeps every old launcher usable without importing Triton.
        try:
            from .fused_kernels_v2 import choose_commit, attention_softmax
        except ImportError as exc:
            raise RuntimeError("Prefix fusion requires your installed Triton-Ascend. "
                               "Restart with FUSION=off if unavailable; do not replace it "
                               "with the generic CUDA Triton package.") from exc
        self.choose_commit = choose_commit
        self.attention_softmax = attention_softmax

    def _history_scores(self, queries, first_q, storage, start, end):
        head = self.head
        n, _, k, _ = queries.shape
        count = end - start
        heads = head.prefix_layers[0].heads
        dim = head.rank // heads
        original = queries[:, start:end]
        query = original
        for li, layer in enumerate(head.prefix_layers):
            if li == 0:
                q = first_q[:, start:end].reshape(n, count * k, heads, dim).transpose(1, 2)
            else:
                q = layer.query_projection(layer.query_norm(query)).reshape(
                    n, count * k, heads, dim).transpose(1, 2).float()
            key = storage[:, :end + 1, li, 0].transpose(1, 2)
            value = storage[:, :end + 1, li, 1].transpose(1, 2)
            with torch.autocast(device_type=query.device.type, enabled=False):
                logits = q @ key.transpose(-1, -2)
                if self.fuse_softmax:
                    weights = self.attention_softmax(
                        logits, self._pair_biases[start][li], self._pair_masks[start], dim)
                else:
                    logits = logits / math.sqrt(dim) + self._pair_biases[start][li]
                    weights = logits.masked_fill(self._pair_masks[start], -torch.inf).softmax(-1)
                attended = weights @ value
            attended = attended.transpose(1, 2).reshape(n, count, k, head.rank).to(query.dtype)
            query = query + layer.output_projection(attended)
            query = query + layer.ffn(layer.ffn_norm(query))
        return head.history_scale * head.prefix_output(query - original).squeeze(-1).float()

    @torch.no_grad()
    def walk(self, prepared, *, forced_indices=None, return_scores=False, projected_kv=None):
        head = self.head
        if head.training:
            raise RuntimeError("FusedPairedHistoryWalk must not be used for training")
        ids, unary, context, successors, queries, predecessors, memories, prev, anchor = prepared
        n, b, k = ids.shape
        if n != 1:
            raise ValueError("Fused paired selector requires batch=1")
        if b < 1 or b > head.max_proposals or k != head.top_k:
            raise ValueError("Proposal/shortlist shape disagrees with checkpoint")
        if head.history_scale == 0 or b == 1:
            return super().walk(prepared, forced_indices=forced_indices,
                                return_scores=return_scores, projected_kv=projected_kv)
        context, successors, predecessors, prev = (
            tensor.float() for tensor in (context, successors, predecessors, prev))
        storage = self._pair_buffers(prepared)
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
        for start in range(1, b, 2):
            end = min(start + 2, b)
            history = self._history_scores(queries, first_q, storage, start, end)
            for i in range(start, end):
                choose(i, history[:, i - start])
        return (selected, scores) if return_scores else selected
