"""Local transitions plus causal attention to older, actually chosen tokens.

This head uses projections of the frozen target token embeddings. It never
inserts labels into the serving shortlist. Teacher scoring and the sequential
walk share the same scoring function. The portable walk prioritizes correctness
and acceptance; it is not a fused inference implementation.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


def rms(x):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                  + 1e-6)).to(x.dtype)


def unary_topk(logits, k):
    """Natural top-k with full-vocabulary argmax first, including BF16 ties."""
    ids = logits.topk(k, dim=-1).indices
    best = logits.argmax(-1, keepdim=True)
    match = ids.eq(best)
    position = torch.where(match.any(-1), match.long().argmax(-1), k - 1)
    order = torch.arange(k, device=ids.device).expand_as(ids).clone()
    order.scatter_(-1, position.unsqueeze(-1), 0)
    order[..., 0] = position
    ids = ids.gather(-1, order)
    ids[..., 0] = best.squeeze(-1)
    return logits.gather(-1, ids), ids


class PreparedPrefix(NamedTuple):
    candidate_ids: torch.Tensor
    unary_logits: torch.Tensor
    context: torch.Tensor
    successors: torch.Tensor
    queries: torch.Tensor
    predecessors: torch.Tensor
    memories: torch.Tensor
    anchor_predecessor: torch.Tensor
    anchor_memory: torch.Tensor


class PrefixCrossAttention(nn.Module):
    def __init__(self, rank, heads, max_proposals):
        super().__init__()
        self.heads = heads
        self.head_dim = rank // heads
        self.query_norm = nn.LayerNorm(rank)
        self.query_projection = nn.Linear(rank, rank, bias=False)
        self.key_projection = nn.Linear(rank, rank, bias=False)
        self.value_projection = nn.Linear(rank, rank, bias=False)
        self.output_projection = nn.Linear(rank, rank, bias=False)
        self.ffn_norm = nn.LayerNorm(rank)
        self.ffn = nn.Sequential(nn.Linear(rank, 2 * rank), nn.SiLU(),
                                 nn.Linear(2 * rank, rank))
        # Column zero is the constant null memory; real distances start at two.
        self.distance_bias = nn.Parameter(torch.zeros(heads, max_proposals + 1))

    def forward(self, query, memory, distances, allowed):
        n, b, k, rank = query.shape
        length = memory.shape[1]
        q = self.query_projection(self.query_norm(query)).reshape(n, b * k, self.heads,
                                                       self.head_dim).transpose(1, 2)
        key = self.key_projection(memory).reshape(n, length, self.heads,
                                         self.head_dim).transpose(1, 2)
        value = self.value_projection(memory).reshape(n, length, self.heads,
                                           self.head_dim).transpose(1, 2)
        bias = self.distance_bias[:, distances].repeat_interleave(k, dim=1)
        mask = allowed.repeat_interleave(k, dim=0)
        # FP32 softmax/reductions, including under NPU BF16 autocast.
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = q.float() @ key.float().transpose(-1, -2)
            logits = logits / math.sqrt(self.head_dim) + bias.float()[None]
            weights = logits.masked_fill(~mask[None, None], -torch.inf).softmax(-1)
            attended = weights @ value.float()
        attended = attended.transpose(1, 2).reshape(n, b, k, rank).to(query.dtype)
        query = query + self.output_projection(attended)
        return query + self.ffn(self.ffn_norm(query))


class LocalPrefixSelector(nn.Module):
    """Explicit local scorer with an independently initialized prefix residual."""

    requires_token_embeddings = True
    supports_triton_walk = False

    def __init__(self, vocab_size, hidden_size, rank=256, top_k=16,
                 max_proposals=15, heads=4, layers=2, correction_scale=1.0,
                 history_scale=1.0):
        super().__init__()
        if not (2 <= top_k <= vocab_size and max_proposals >= 1):
            raise ValueError("Invalid shortlist or proposal count")
        if heads < 1 or rank % heads or layers < 1:
            raise ValueError("rank must be divisible by heads; layers must be positive")
        if not all(math.isfinite(x) and 0 <= x <= 1
                   for x in (correction_scale, history_scale)):
            raise ValueError("Inference scales must be finite and in [0, 1]")
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.rank, self.top_k, self.max_proposals = rank, top_k, max_proposals
        self.correction_scale, self.history_scale = correction_scale, history_scale
        self.predecessor_projection = nn.Linear(hidden_size, rank, bias=False)
        self.successor_projection = nn.Linear(hidden_size, rank, bias=False)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.candidate_projection = nn.Linear(hidden_size, rank, bias=False)
        self.memory_projection = nn.Linear(hidden_size, rank, bias=False)
        self.prefix_layers = nn.ModuleList([
            PrefixCrossAttention(rank, heads, max_proposals) for _ in range(layers)
        ])
        self.prefix_output = nn.Linear(rank, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Both corrections initially vanish. Their output matrices still receive
        # gradients at initialization; upstream branches learn after that update.
        nn.init.zeros_(self.successor_projection.weight)
        nn.init.zeros_(self.prefix_output.weight)

    @property
    def gate(self):
        # Compatibility with legacy metrics. There is no learned global gate.
        return self.prefix_output.weight.new_tensor(self.correction_scale).float()

    def set_history_trainable(self, enabled):
        for name, parameter in self.named_parameters():
            if name.startswith(("candidate_projection.", "memory_projection.",
                                "prefix_layers.", "prefix_output.")):
                parameter.requires_grad_(enabled)

    def _tokens(self, ids, embedding_weight):
        if embedding_weight is None or embedding_weight.shape != (
                self.vocab_size, self.hidden_size):
            raise ValueError("The matching, full target token embedding is required")
        return rms(F.embedding(ids.long(), embedding_weight.detach()))

    def _prepare(self, hidden, ids, embedding_weight):
        if hidden.ndim != 3 or ids.ndim != 3 or hidden.shape[:2] != ids.shape[:2]:
            raise ValueError("Expected hidden [N,B,H] and candidates [N,B,K]")
        if ids.shape[-1] != self.top_k or ids.shape[1] > self.max_proposals:
            raise ValueError("Shortlist/proposal shape does not match the checkpoint")
        tokens = self._tokens(ids, embedding_weight)
        context = rms(self.hidden_projection(rms(hidden)))
        successors = self.successor_projection(tokens)
        query = rms(context.unsqueeze(-2) + self.candidate_projection(tokens))
        return context, successors, query, tokens

    def _prefix_score(self, query, memory, query_positions, source_positions):
        n, b, k, _ = query.shape
        if self.history_scale == 0:
            return query.new_zeros(n, b, k).float()
        # Only sources at least two tokens earlier: the immediate predecessor
        # has its own unnormalized local transition and is never diluted here.
        distances = query_positions[:, None] - source_positions[None, :]
        allowed = distances >= 2
        has_history = allowed.any(-1)
        null = memory.new_zeros(n, 1, self.rank)
        memory = torch.cat((null, memory), dim=1)
        allowed = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=query.device),
                             allowed), dim=1)
        distances = torch.cat((torch.zeros(b, 1, dtype=torch.long,
                                          device=query.device),
                               distances.clamp(0, self.max_proposals)), dim=1)
        original = query
        for layer in self.prefix_layers:
            query = layer(query, memory, distances, allowed)
        score = self.prefix_output(query - original).squeeze(-1).float()
        return self.history_scale * score * has_history[None, :, None]

    def _scores(self, unary, context, successors, query, previous, memory,
                query_positions, source_positions):
        with torch.autocast(device_type=unary.device.type, enabled=False):
            local = (context.float().unsqueeze(-2) * previous.float().unsqueeze(-2)
                     * successors.float()).sum(-1) / math.sqrt(self.rank)
        history = self._prefix_score(query, memory, query_positions, source_positions)
        return unary.float() + self.correction_scale * (local + history)

    def score_teacher_prefix(self, hidden, candidate_ids, unary_logits, anchor_ids,
                             prefix_token_ids, *, embedding_weight=None):
        context, successors, query, _ = self._prepare(hidden, candidate_ids,
                                                      embedding_weight)
        n, b, _ = candidate_ids.shape
        if prefix_token_ids.shape != (n, b):
            raise ValueError("Reference tokens must have shape [N,B]")
        source_ids = torch.cat((anchor_ids[:, None], prefix_token_ids), dim=1)
        source = self._tokens(source_ids, embedding_weight)
        previous = rms(self.predecessor_projection(source[:, :b]))
        memory = rms(self.memory_projection(source))
        return self._scores(unary_logits, context, successors, query, previous,
                            memory, torch.arange(1, b + 1, device=hidden.device),
                            torch.arange(b + 1, device=hidden.device))

    def prepare_tables(self, hidden, candidate_ids, unary_logits, anchor_ids,
                       *, embedding_weight=None):
        context, successors, query, tokens = self._prepare(hidden, candidate_ids,
                                                           embedding_weight)
        anchor = self._tokens(anchor_ids, embedding_weight)
        return PreparedPrefix(
            candidate_ids, unary_logits.float(), context, successors, query,
            rms(self.predecessor_projection(tokens)),
            rms(self.memory_projection(tokens)),
            rms(self.predecessor_projection(anchor)),
            rms(self.memory_projection(anchor)),
        )

    def walk(self, prepared, *, forced_indices=None, return_scores=False):
        ids, unary, context, successors, queries, predecessors, memories, prev, anchor = prepared
        n, b, _ = ids.shape
        history = [anchor]
        selected, score_rows = [], []
        for i in range(b):
            memory = torch.stack(history, dim=1)
            logits = self._scores(
                unary[:, i:i+1], context[:, i:i+1], successors[:, i:i+1],
                queries[:, i:i+1], prev[:, None], memory,
                torch.tensor([i + 1], device=ids.device),
                torch.arange(i + 1, device=ids.device),
            )[:, 0]
            index = logits.argmax(-1) if forced_indices is None else forced_indices[:, i]
            selected.append(ids[:, i].gather(-1, index[:, None]).squeeze(-1))
            gather = index[:, None, None].expand(n, 1, self.rank)
            prev = predecessors[:, i].gather(1, gather).squeeze(1)
            history.append(memories[:, i].gather(1, gather).squeeze(1))
            if return_scores:
                score_rows.append(logits)
        result = torch.stack(selected, dim=1)
        return (result, torch.stack(score_rows, dim=1)) if return_scores else result

    def greedy_walk(self, prepared):
        return self.walk(prepared)

    def greedy_walk_scaled(self, prepared, scale):
        previous = self.correction_scale
        try:
            self.correction_scale = previous * scale
            return self.greedy_walk(prepared)
        finally:
            self.correction_scale = previous

    @torch.no_grad()
    def greedy(self, hidden, candidate_ids, unary_logits, anchor_ids,
               *, embedding_weight=None):
        return self.greedy_walk(self.prepare_tables(
            hidden, candidate_ids, unary_logits, anchor_ids,
            embedding_weight=embedding_weight))


def make_prefix_head(vocab_size, hidden_size, rank, top_k, max_proposals,
                     gate_init=0.1, correction_scale=1.0, kind="legacy",
                     heads=4, layers=2, history_scale=1.0):
    if kind == "legacy":
        from .model_definitions import PrefixAttentionSelector
        return PrefixAttentionSelector(vocab_size, hidden_size, rank, top_k,
                                       max_proposals, gate_init, correction_scale)
    if kind != "local_prefix_v2":
        raise ValueError(f"Unknown prefix selector kind: {kind}")
    return LocalPrefixSelector(vocab_size, hidden_size, rank, top_k, max_proposals,
                               heads, layers, correction_scale, history_scale)
