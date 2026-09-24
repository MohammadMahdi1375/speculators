"""One-head prefix selector; native PyTorch operations work on CPU/CUDA/NPU.

Candidate features are fixed by a single DFlash backbone pass. Training reads
teacher prefix tokens directly. Inference prepares candidate-pair tables and
walks them using ONLY actual earlier selections. No gold-token insertion.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class PrefixTables(NamedTuple):
    candidate_ids: torch.Tensor  # [N, B, K]
    unary_logits: torch.Tensor  # [N, B, K]
    numerator: torch.Tensor  # [N, B, K, B, K], query then source
    denominator: torch.Tensor
    anchor_numerator: torch.Tensor  # [N, B, K]
    anchor_denominator: torch.Tensor
    gate: torch.Tensor


class PrefixAttentionSelector(nn.Module):
    """Candidate-specific attention with bounded scores and FP32 reductions.

    The vocabulary codebook is shared across query/key/readout/value features.
    Large hidden projections are computed once per position, not once per
    candidate. One implementation is used by training and the vLLM adapter.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        rank: int = 64,
        top_k: int = 16,
        max_proposals: int = 15,
        gate_init: float = 0.1,
        correction_scale: float = 1.0,
    ):
        super().__init__()
        if not 1 < top_k <= vocab_size or max_proposals < 1:
            raise ValueError("Invalid candidate count or proposal length")
        if not math.isfinite(correction_scale) or not 0 <= correction_scale <= 1:
            raise ValueError("correction_scale must be finite and in [0, 1]")
        self.correction_scale = float(correction_scale)
        self.top_k = top_k
        self.rank = rank
        self.max_proposals = max_proposals
        self.token_codes = nn.Embedding(vocab_size, rank)
        self.hidden_proj = nn.Linear(hidden_size, rank, bias=False)
        self.query_proj = nn.Linear(rank, rank, bias=False)
        self.key_proj = nn.Linear(rank, rank, bias=False)
        self.readout_proj = nn.Linear(rank, rank, bias=False)
        self.value_proj = nn.Linear(rank, rank, bias=False)
        self.distance_bias = nn.Parameter(torch.zeros(max_proposals + 1))
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(gate_init / (4.0 - gate_init)))
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_codes.weight, std=0.1)
        for layer in (
            self.hidden_proj,
            self.query_proj,
            self.key_proj,
            self.readout_proj,
            self.value_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)

    @property
    def gate(self):
        return (4.0 * self.correction_scale) * self.gate_logit.float().sigmoid()

    @staticmethod
    def _rms(x):
        return (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).to(x.dtype)

    def _features(self, hidden, candidate_ids):
        # [N,B,1,r] + [N,B,K,r]; the hidden projection is NOT repeated K times.
        context = self.hidden_proj(hidden).unsqueeze(-2)
        return self._rms(F.silu(context + self.token_codes(candidate_ids)))

    def _anchor_features(self, anchor_ids):
        # No anchor hidden state is required by serving. This exact definition
        # also holds in training, where a backbone anchor state is available.
        return self._rms(F.silu(self.token_codes(anchor_ids)))

    def _attention(self, q, readout, key, value, bias):
        # Bounded exponentials avoid FP16 overflow and selected-subset underflow.
        # float() alone does not prevent autocast from casting matmul back to
        # BF16. Serving normally has no autocast context; align the two paths.
        with torch.autocast(device_type=q.device.type, enabled=False):
            weight = torch.exp(
                (
                    torch.matmul(q.float(), key.float().transpose(-1, -2))
                    / math.sqrt(self.rank)
                    + bias.float()
                ).clamp(-8.0, 8.0)
            )
            compatibility = torch.matmul(
                readout.float(), value.float().transpose(-1, -2)
            ) / math.sqrt(self.rank)
        return weight * compatibility, weight

    def _check_inputs(self, hidden, candidate_ids):
        if hidden.ndim != 3 or candidate_ids.ndim != 3:
            raise ValueError("Expected hidden [N,B,H] and candidates [N,B,K]")
        n, b, k = candidate_ids.shape
        if hidden.shape[:2] != (n, b) or k != self.top_k:
            raise ValueError("Hidden/candidate shapes or configured top_k do not match")
        if b > self.max_proposals:
            raise ValueError("Proposal count exceeds the trained block length")
        return n, b, k

    def _queries(self, hidden, candidate_ids):
        z = self._features(hidden, candidate_ids)
        return z, self.query_proj(z), self.readout_proj(z)

    def score_teacher_prefix(
        self,
        hidden,
        candidate_ids,
        unary_logits,
        anchor_ids,
        prefix_token_ids,
    ):
        """Score each slot under reference x_<i; x_i/future tokens are masked.

        Reference tokens need not be in the shortlisted candidates. This is
        ordinary teacher forcing, not inserting target labels into candidates.
        The serving distribution itself is restricted to its natural shortlist.
        """
        n, b, k = self._check_inputs(hidden, candidate_ids)
        if prefix_token_ids.shape != (n, b):
            raise ValueError("prefix_token_ids must be [N,B]")
        _, q, readout = self._queries(hidden, candidate_ids)
        source = self._features(hidden, prefix_token_ids.unsqueeze(-1)).squeeze(-2)
        anchor = self._anchor_features(anchor_ids).unsqueeze(1)
        source = torch.cat((anchor, source), dim=1)
        key, value = self.key_proj(source), self.value_proj(source)
        query_pos = torch.arange(1, b + 1, device=hidden.device)
        source_pos = torch.arange(b + 1, device=hidden.device)
        distances = query_pos[:, None] - source_pos[None, :]
        bias = self.distance_bias[distances.clamp(0, self.max_proposals)].float()
        a, d = self._attention(
            q.flatten(1, 2),
            readout.flatten(1, 2),
            key,
            value,
            bias.repeat_interleave(k, dim=0).unsqueeze(0),
        )
        causal = (source_pos[None, :] < query_pos[:, None]).repeat_interleave(k, 0)
        a = a.masked_fill(~causal, 0)
        d = d.masked_fill(~causal, 0)
        # A fixed null entry (weight=1, compatibility=0) makes the denominator
        # strictly positive. Anchor is an additional learned token entry.
        correction = (a.sum(-1) / (1.0 + d.sum(-1))).view(n, b, k)
        return unary_logits.float() + self.gate * correction

    def prepare_tables(self, hidden, candidate_ids, unary_logits, anchor_ids):
        """Prepare neural features/interactions before any sampled token is known."""
        n, b, k = self._check_inputs(hidden, candidate_ids)
        z, q, readout = self._queries(hidden, candidate_ids)
        key, value = self.key_proj(z), self.value_proj(z)
        pos = torch.arange(b, device=hidden.device)
        distances = pos[:, None] - pos[None, :]
        bias = self.distance_bias[distances.clamp(0, self.max_proposals)].float()
        bias = bias.repeat_interleave(k, 0).repeat_interleave(k, 1)
        a, d = self._attention(
            q.flatten(1, 2),
            readout.flatten(1, 2),
            key.flatten(1, 2),
            value.flatten(1, 2),
            bias.unsqueeze(0),
        )
        causal = (
            (pos[None, :] < pos[:, None])
            .repeat_interleave(k, 0)
            .repeat_interleave(k, 1)
        )
        a, d = a.masked_fill(~causal, 0), d.masked_fill(~causal, 0)
        anchor = self._anchor_features(anchor_ids).unsqueeze(1)
        anchor_bias = (
            self.distance_bias[torch.arange(1, b + 1, device=hidden.device)]
            .repeat_interleave(k)
            .float()[None, :, None]
        )
        anchor_a, anchor_d = self._attention(
            q.flatten(1, 2),
            readout.flatten(1, 2),
            self.key_proj(anchor),
            self.value_proj(anchor),
            anchor_bias,
        )
        return PrefixTables(
            candidate_ids,
            unary_logits.float(),
            a.view(n, b, k, b, k),
            d.view(n, b, k, b, k),
            anchor_a.view(n, b, k),
            1.0 + anchor_d.view(n, b, k),
            self.gate,
        )

    @staticmethod
    def walk(tables, *, temperature=0.0, generator=None, forced_indices=None):
        """Native-device reference walk, with normalized realized proposal q.

        Greedy proposals return point masses, NOT pre-argmax softmax vectors.
        The initial Ascend serving adapter uses greedy proposals. This eager
        loop is a correctness baseline; it is not a fused performance kernel.
        """
        ids, unary, a, d, numerator, denominator, gate = tables
        n, b, k = ids.shape
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        chosen, probabilities, row_logits = [], [], []
        for i in range(b):
            logits = unary[:, i] + gate * numerator[:, i] / denominator[:, i]
            if temperature == 0:
                index = logits.argmax(-1)
                probs = F.one_hot(index, k).float()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                index = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            if forced_indices is not None:
                index = forced_indices[:, i]
            chosen.append(ids[:, i].gather(1, index[:, None]).squeeze(1))
            probabilities.append(probs)
            row_logits.append(logits)
            # Gather one source candidate for every future query candidate.
            gather_index = index[:, None, None, None].expand(n, b, k, 1)
            numerator = numerator + a[:, :, :, i, :].gather(-1, gather_index).squeeze(
                -1
            )
            denominator = denominator + d[:, :, :, i, :].gather(
                -1, gather_index
            ).squeeze(-1)
        return (
            torch.stack(chosen, 1),
            torch.stack(probabilities, 1),
            torch.stack(row_logits, 1),
        )

    @torch.no_grad()
    def greedy(self, hidden, candidate_ids, unary_logits, anchor_ids):
        return self.greedy_walk(
            self.prepare_tables(hidden, candidate_ids, unary_logits, anchor_ids)
        )

    @staticmethod
    def greedy_walk(tables):
        """Greedy-only walk: no unused one-hot probabilities or logits history."""
        ids, unary, a, d, numerator, denominator, gate = tables
        n, b, k = ids.shape
        chosen = []
        for i in range(b):
            logits = unary[:, i] + gate * numerator[:, i] / denominator[:, i]
            index = logits.argmax(-1)
            chosen.append(ids[:, i].gather(1, index[:, None]).squeeze(1))
            if i + 1 < b:
                gather_index = index[:, None, None, None].expand(n, b, k, 1)
                numerator = numerator + a[:, :, :, i, :].gather(
                    -1, gather_index
                ).squeeze(-1)
                denominator = denominator + d[:, :, :, i, :].gather(
                    -1, gather_index
                ).squeeze(-1)
        return torch.stack(chosen, 1)
