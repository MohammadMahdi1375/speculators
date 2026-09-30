"""Pair adjacent history queries without approximating the trained selector.

The history mask allows source positions <= query_position - 2. Thus the
history scores for draft positions p and p+1 are both known once token p-1
has been chosen. The immediate-predecessor local scores remain sequential.
This is valid only for the audited local_prefix_v2 history mask.

BF16 batched GEMMs/softmax can round differently. Keep the original reference
validation; a selected-token mismatch is an error, never an accepted shortcut.
"""
import math

import torch

from .inference_v2 import CachedPrefixWalk


class PairedHistoryWalk(CachedPrefixWalk):
    def __init__(self, head):
        super().__init__(head)
        self._pair_signature = None
        self._pair_masks = {}
        self._pair_biases = {}

    def _pair_buffers(self, prepared):
        storage = self._buffers(prepared)
        # Initialize future candidate slots once per block, rather than launch
        # a separate zero-fill for each pair. Chosen tokens overwrite their
        # slots in order; the as-yet-unchosen masked slot stays exactly zero.
        storage[:, 2:].zero_()
        if self._pair_signature != self._signature:
            _, b, k = prepared.candidate_ids.shape
            device = prepared.queries.device
            self._pair_masks, self._pair_biases = {}, {}
            for start in range(1, b, 2):
                end = min(start + 2, b)
                # Same memory length as the LAST query's original walk.
                # For a pair, its last slot is the as-yet-unchosen token at
                # 'start'. Both queries mask that slot, so initialize it to 0.
                length = end + 1
                distances = torch.tensor([
                    [0] + [i + 1 - j for j in range(length - 1)]
                    for i in range(start, end)
                ], device=device, dtype=torch.long)
                allowed = distances >= 2
                allowed[:, 0] = True  # constant null memory
                self._pair_masks[start] = ~allowed.repeat_interleave(k, 0)[None, None]
                self._pair_biases[start] = tuple(
                    layer.distance_bias[:, distances].float().repeat_interleave(k, 1)[None]
                    for layer in self.head.prefix_layers
                )
            self._pair_signature = self._signature
        return storage

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
            raise RuntimeError("PairedHistoryWalk must not be used for training")
        ids, unary, context, successors, queries, predecessors, memories, prev, anchor = prepared
        n, b, k = ids.shape
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
        selected, scores = [], []

        def choose(i, history_score):
            nonlocal prev
            with torch.autocast(device_type=unary.device.type, enabled=False):
                local = (context[:, i, None] * prev[:, None] * successors[:, i]).sum(-1) / math.sqrt(head.rank)
            logits = unary[:, i].float() + head.correction_scale * (local + history_score)
            index = logits.argmax(-1) if forced_indices is None else forced_indices[:, i]
            selected.append(ids[:, i].gather(-1, index[:, None]).squeeze(-1))
            if i + 1 < b:
                prev = predecessors[:, i].gather(
                    1, index[:, None, None].expand(n, 1, head.rank)).squeeze(1)
                chosen = choices[:, i].gather(
                    1, index[:, None, None].expand(n, 1, choices.shape[-1]))
                storage[:, i + 2].copy_(chosen.reshape(n, len(head.prefix_layers), 2, heads, dim))
            if return_scores:
                scores.append(logits)

        # The first proposal has exactly zero history correction.
        choose(0, 0.0)
        for start in range(1, b, 2):
            end = min(start + 2, b)
            history = self._history_scores(queries, first_q, storage, start, end)
            for i in range(start, end):
                choose(i, history[:, i - start])
        result = torch.stack(selected, dim=1)
        return (result, torch.stack(scores, dim=1)) if return_scores else result
