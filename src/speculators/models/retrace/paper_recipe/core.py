"""Parallel DFlash proposals with ReTrace conditioning; no DSpARK dependency."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from transformers import AutoConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.retrace.config import ReTraceSpeculatorConfig
from .memory import (
    ReTraceConditioner,
    ReTraceMemory,
    condition_query_block,
)
from speculators.utils.loading import load_model_layers


@dataclass
class DraftBlock:
    hidden: torch.Tensor
    logits: torch.Tensor
    token_ids: torch.Tensor


class ReTraceDraftModel(DFlashDraftModel):
    config_class: ClassVar[type[ReTraceSpeculatorConfig]] = ReTraceSpeculatorConfig

    def __init__(self, config):
        super().__init__(config)
        # Add AFTER the base model's post_init: preserve the zero residual.
        self.retrace = ReTraceConditioner(self.hidden_size)

    def load_verifier_weights(self):
        """Load only frozen shared tensors, including Qwen's tied output head."""
        path = self.config.speculators_config.verifier.name_or_path
        if path is None:
            return
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"], path
        )
        embedding = weights["embed_tokens.weight"]
        head = weights.get("lm_head.weight")
        if head is None:
            if not config.tie_word_embeddings:
                raise ValueError("Untied target is missing lm_head.weight")
            head = embedding
        dtype, device = self.dtype, self.device
        self.embed_tokens = nn.Embedding.from_pretrained(
            embedding.to(device=device, dtype=dtype), freeze=True
        )
        self.lm_head = nn.Linear(
            self.hidden_size,
            self.verifier_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lm_head.weight = nn.Parameter(
            head.to(device=device, dtype=dtype), requires_grad=False
        )
        self.verifier_lm_head = self.lm_head
        self.verifier_norm.weight = nn.Parameter(
            weights["model.norm.weight"].to(device=device, dtype=dtype),
            requires_grad=False,
        )

    @classmethod
    def from_training_args(cls, verifier_config, t2d=None, d2t=None, **kwargs):
        config = ReTraceSpeculatorConfig(
            **cls._build_base_config_kwargs("retrace", verifier_config, **kwargs)
        )
        model = cls(config)
        model.load_verifier_weights()
        return model

    def forward(
        self, context_hidden=None, anchor=None, memory=None, beta=1.0, **kwargs
    ):
        if context_hidden is None or anchor is None:
            raise ValueError(
                "ReTrace requires live proposal verification. Use python -m speculators.models.retrace.paper_recipe.train_pretrained; stock teacher-forced batches contain no rejected trajectory."
            )
        return self.propose(context_hidden, anchor, memory, beta)

    def propose(
        self, context_hidden, anchor, memory: ReTraceMemory | None = None, beta=1.0
    ):
        if (
            context_hidden.ndim != 3
            or context_hidden.shape[0] != 1
            or anchor.shape != (1,)
        ):
            raise ValueError("One trajectory per worker is supported")
        length = context_hidden.shape[1]
        ids = torch.full(
            (1, self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=anchor.device,
        )
        ids[:, 0] = anchor
        hidden = self.embed_tokens(ids)
        if self.config.retrace_enabled and memory is not None:
            hidden = condition_query_block(self.retrace, hidden, memory, beta)
        positions = torch.arange(length + self.block_size, device=anchor.device)[None]
        rotary = self.rotary_emb(context_hidden, positions)
        context = self.hidden_norm(self.fc(context_hidden.detach()))
        mask = hidden.new_zeros(1, 1, self.block_size, length + self.block_size)
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                target_hidden=context,
                attention_mask=mask,
                position_embeddings=rotary,
                use_cache=False,
            )
        hidden = self.norm(hidden)[:, 1:]
        logits = self.lm_head(hidden)
        return DraftBlock(hidden, logits, logits.detach().argmax(-1))

    def training_blocks(self, context_hidden, anchors, positions, memory, beta=1.0):
        """Train independent blocks with clean causal context and detached memory.

        Context keys at/after each anchor are invisible to that block. Query
        tokens see only their own block. This is the DFlash block mask; memory
        comes from real preceding verification rounds, never clean future states.
        """
        if context_hidden.ndim != 3 or context_hidden.shape[0] != 1:
            raise ValueError("Expected one clean context sequence")
        count, size = anchors.numel(), self.block_size
        if count == 0 or positions.shape != (count,):
            raise ValueError("Need matching nonempty anchors and positions")
        device = anchors.device
        length = context_hidden.shape[1]
        ids = torch.full(
            (count, size), self.mask_token_id, device=device, dtype=torch.long
        )
        ids[:, 0] = anchors
        hidden = self.embed_tokens(ids)
        if self.config.retrace_enabled and memory is not None:
            hidden = condition_query_block(self.retrace, hidden, memory, beta)
        hidden = hidden.reshape(1, count * size, -1)
        query_positions = positions[:, None] + torch.arange(size, device=device)
        all_positions = torch.cat(
            (torch.arange(length, device=device), query_positions.flatten())
        )[None]
        rotary = self.rotary_emb(hidden, all_positions)
        context = self.hidden_norm(self.fc(context_hidden.detach()))
        block_ids = torch.arange(count, device=device).repeat_interleave(size)
        visible_context = (
            torch.arange(length, device=device)[None] < positions[block_ids, None]
        )
        visible_queries = block_ids[:, None] == block_ids[None]
        visible = torch.cat((visible_context, visible_queries), dim=-1)
        mask = hidden.new_zeros(1, 1, count * size, length + count * size)
        mask.masked_fill_(~visible[None, None], float("-inf"))
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                target_hidden=context,
                attention_mask=mask,
                position_embeddings=rotary,
                use_cache=False,
            )
        hidden = self.norm(hidden).reshape(count, size, -1)[:, 1:]
        logits = self.lm_head(hidden)
        return DraftBlock(hidden, logits, logits.detach().argmax(-1))
