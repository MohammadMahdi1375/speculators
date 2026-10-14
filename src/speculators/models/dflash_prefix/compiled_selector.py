"""Opt-in multi-head compiled prefix selector for a newly trained checkpoint.

The local transition retains local_prefix_v2's exact feature definition. Older
tokens contribute candidate-specific scalar attention readouts whose numerator
and denominator are additive after candidate-pair preparation. The legacy
single-head PrefixAttentionSelector already uses this basic compilation idea;
this variant combines multi-head older-history attention with the established
local transition. It is not equivalent to local_prefix_v2's two nonlinear layers.

Bounded energies cap*tanh(energy/cap) are part of training and inference, not an
approximate serving clip. Projections may use BF16 autocast; attention energies,
exponentials, local scores and accumulated statistics always use FP32. The eager
walk is a correctness implementation, not an NPU throughput claim.
"""

import math
from typing import NamedTuple

import torch
from torch import nn

from .selector_v2 import LocalPrefixSelector, rms


class CompiledPrefixTables(NamedTuple):
    candidate_ids: torch.Tensor
    unary_logits: torch.Tensor
    local_pairs: torch.Tensor  # [N,P,K,Kprevious]; row zero repeats anchor
    numerator: torch.Tensor  # [N,H,P,K,P,Ksource]
    denominator: torch.Tensor
    initial_numerator: torch.Tensor  # [N,H,P,K], anchor if distance >= 2
    initial_denominator: torch.Tensor  # includes constant null weight 1
    head_weights: torch.Tensor
    correction_scale: float
    history_scale: float


class CompiledPrefixSelector(LocalPrefixSelector):
    """One compiled attention layer over older chosen tokens plus local scores."""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        rank=256,
        top_k=16,
        max_proposals=15,
        heads=4,
        correction_scale=1.0,
        history_scale=1.0,
        compiled_logit_cap=8.0,
    ):
        # Do not instantiate the parent's unused nonlinear attention layers.
        nn.Module.__init__(self)
        if not (2 <= top_k <= vocab_size and max_proposals >= 1):
            raise ValueError("Invalid shortlist or proposal count")
        if heads < 1 or rank < 1 or rank % heads:
            raise ValueError("rank must be positive and divisible by heads")
        if not all(
            math.isfinite(x) and 0 <= x <= 1
            for x in (correction_scale, history_scale)
        ):
            raise ValueError("Inference scales must be finite and in [0, 1]")
        if not math.isfinite(compiled_logit_cap) or not 0 < compiled_logit_cap <= 8:
            raise ValueError("compiled_logit_cap must be finite and in (0, 8]")
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.rank, self.top_k, self.max_proposals = rank, top_k, max_proposals
        self.heads, self.head_dim = heads, rank // heads
        self.correction_scale, self.history_scale = correction_scale, history_scale
        self.compiled_logit_cap = float(compiled_logit_cap)
        for name in (
            "predecessor_projection",
            "successor_projection",
            "hidden_projection",
            "candidate_projection",
            "memory_projection",
        ):
            setattr(self, name, nn.Linear(hidden_size, rank, bias=False))
        for name in ("history_query", "history_key", "history_readout", "history_value"):
            setattr(self, name, nn.Linear(rank, rank, bias=False))
        self.distance_bias = nn.Parameter(torch.zeros(heads, max_proposals + 1))
        self.head_weights = nn.Parameter(torch.zeros(heads))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.successor_projection.weight)

    @property
    def gate(self):
        return self.head_weights.new_tensor(self.correction_scale).float()

    def set_history_trainable(self, enabled):
        for name, parameter in self.named_parameters():
            if name.startswith(
                ("candidate_projection.", "memory_projection.", "history_")
            ) or name in {"distance_bias", "head_weights"}:
                parameter.requires_grad_(enabled)

    def initialize_from_teacher(self, teacher):
        """Copy compatible features and freeze the copied local transition only."""
        if type(teacher) is not LocalPrefixSelector:
            raise ValueError("Expected a local_prefix_v2 teacher")
        for name in ("vocab_size", "hidden_size", "rank", "top_k", "max_proposals"):
            if getattr(self, name) != getattr(teacher, name):
                raise ValueError(f"Teacher/student {name} differs")
        if any(layer.heads != self.heads for layer in teacher.prefix_layers):
            raise ValueError("Teacher/student head counts differ")
        with torch.no_grad():
            for name in (
                "predecessor_projection",
                "successor_projection",
                "hidden_projection",
                "candidate_projection",
                "memory_projection",
            ):
                getattr(self, name).weight.copy_(getattr(teacher, name).weight)
        for name in ("predecessor_projection", "successor_projection", "hidden_projection"):
            getattr(self, name).requires_grad_(False)
        return self

    def _heads(self, tensor):
        return tensor.reshape(
            tensor.shape[0], -1, self.heads, self.head_dim
        ).transpose(1, 2)

    def _history_tables(self, query, memory, query_positions, source_positions):
        n, p, k, _ = query.shape
        q = self._heads(self.history_query(query))
        r = self._heads(self.history_readout(query))
        key = self._heads(self.history_key(memory))
        value = self._heads(self.history_value(memory))
        distances = query_positions[:, None] - source_positions[None, :]
        allowed = distances >= 2
        bias = self.distance_bias[:, distances.clamp(0, self.max_proposals)]
        bias = bias.repeat_interleave(k, dim=1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            energy = q.float() @ key.float().transpose(-1, -2)
            energy = energy / math.sqrt(self.head_dim) + bias.float()[None]
            cap = self.compiled_logit_cap
            weight = (cap * torch.tanh(energy / cap)).exp()
            weight = weight.masked_fill(
                ~allowed.repeat_interleave(k, 0)[None, None], 0
            )
            readout = (r.float() @ value.float().transpose(-1, -2)) / math.sqrt(
                self.head_dim
            )
            numerator = weight * readout
        shape = (n, self.heads, p, k, memory.shape[1])
        return numerator.reshape(shape), weight.reshape(shape)

    def score_teacher_prefix(
        self,
        hidden,
        candidate_ids,
        unary_logits,
        anchor_ids,
        prefix_token_ids,
        *,
        embedding_weight=None,
    ):
        context, successors, query, _ = self._prepare(
            hidden, candidate_ids, embedding_weight
        )
        n, p, _ = candidate_ids.shape
        if prefix_token_ids.shape != (n, p):
            raise ValueError("Reference tokens must have shape [N,P]")
        source = self._tokens(
            torch.cat((anchor_ids[:, None], prefix_token_ids), 1), embedding_weight
        )
        previous = rms(self.predecessor_projection(source[:, :p]))
        memory = rms(self.memory_projection(source))
        positions = torch.arange(1, p + 1, device=hidden.device)
        a, d = self._history_tables(
            query, memory, positions, torch.arange(p + 1, device=hidden.device)
        )
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            local = (
                context.float()[:, :, None]
                * previous.float()[:, :, None]
                * successors.float()
            ).sum(-1) / math.sqrt(self.rank)
            history = (
                a.sum(-1)
                / (1 + d.sum(-1))
                * self.head_weights.float()[None, :, None, None]
            ).sum(1)
            return unary_logits.float() + self.correction_scale * (
                local + self.history_scale * history
            )

    def prepare_tables(
        self, hidden, candidate_ids, unary_logits, anchor_ids, *, embedding_weight=None
    ):
        context, successors, query, tokens = self._prepare(
            hidden, candidate_ids, embedding_weight
        )
        n, p, k = candidate_ids.shape
        anchor = self._tokens(anchor_ids, embedding_weight)
        previous = rms(self.predecessor_projection(tokens))
        anchor_previous = rms(self.predecessor_projection(anchor))
        previous = torch.cat(
            (
                anchor_previous[:, None, None].expand(n, 1, k, self.rank),
                previous[:, :-1],
            ),
            1,
        )
        memory = rms(self.memory_projection(tokens)).flatten(1, 2)
        anchor_memory = rms(self.memory_projection(anchor))[:, None]
        positions = torch.arange(1, p + 1, device=hidden.device)
        a, d = self._history_tables(
            query, memory, positions, positions.repeat_interleave(k)
        )
        anchor_a, anchor_d = self._history_tables(
            query, anchor_memory, positions, positions.new_zeros(1)
        )
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            local = torch.einsum(
                "npr,npkr,npdr->npkd",
                context.float(),
                successors.float(),
                previous.float(),
            ) / math.sqrt(self.rank)
        return CompiledPrefixTables(
            candidate_ids,
            unary_logits.float(),
            local,
            a.reshape(n, self.heads, p, k, p, k),
            d.reshape(n, self.heads, p, k, p, k),
            anchor_a.squeeze(-1),
            1 + anchor_d.squeeze(-1),
            self.head_weights.float(),
            self.correction_scale,
            self.history_scale,
        )

    @staticmethod
    def walk(tables, *, forced_indices=None, return_scores=False):
        """Follow actual choices using only prepared scalar tables and local pairs."""
        (
            ids, unary, pairs, a, d, numerator, denominator, heads, scale, history_scale
        ) = tables
        n, p, k = ids.shape
        previous = torch.zeros(n, dtype=torch.long, device=ids.device)
        tokens, score_rows = [], []
        with torch.autocast(device_type=ids.device.type, enabled=False):
            for i in range(p):
                local = pairs[:, i].gather(
                    -1, previous[:, None, None].expand(n, k, 1)
                ).squeeze(-1)
                history = (
                    numerator[:, :, i] / denominator[:, :, i] * heads[None, :, None]
                ).sum(1)
                score = unary[:, i] + scale * (local + history_scale * history)
                index = (
                    score.argmax(-1)
                    if forced_indices is None
                    else forced_indices[:, i]
                )
                tokens.append(ids[:, i].gather(-1, index[:, None]).squeeze(-1))
                if return_scores:
                    score_rows.append(score)
                if i + 1 < p:
                    gather = index[:, None, None, None, None].expand(
                        n, heads.numel(), p, k, 1
                    )
                    numerator = numerator + a[..., i, :].gather(-1, gather).squeeze(-1)
                    denominator = denominator + d[..., i, :].gather(-1, gather).squeeze(-1)
                previous = index
        tokens = torch.stack(tokens, 1)
        return (tokens, torch.stack(score_rows, 1)) if return_scores else tokens

    @staticmethod
    def greedy_walk(tables):
        return CompiledPrefixSelector.walk(tables)

    @staticmethod
    def greedy_walk_scaled(tables, scale):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("scale must be finite and in [0,1]")
        return CompiledPrefixSelector.greedy_walk(
            tables._replace(correction_scale=tables.correction_scale * scale)
        )
