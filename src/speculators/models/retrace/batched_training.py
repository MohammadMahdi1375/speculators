"""Concurrent prompt rollouts with per-row speculative cache rollback.

The global prompt batch, greedy verification, one-round rejected memory and
clean replay loss are unchanged. Only no-gradient execution is batched. Cache
slots are private to a trajectory; rejected suffixes are overwritten at their
absolute positions and can never become visible through the causal mask.
"""

import weakref
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from speculators.models.dflash.model_definitions import apply_rotary_pos_emb

from .clean_training import TrainingTrajectory
from .core import DraftBlock
from .execution import draft_attention
from .memory import ReTraceMemory, carry_suffix, condition_query_block
from .target import LocalTarget, TargetStates


class _RowKVLayer(CacheLayerMixin):
    """Fixed-capacity storage, with distinct write positions for each row."""

    is_sliding = False

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def lazy_initialization(self, key_states, value_states):
        shape = list(key_states.shape)
        shape[-2] = self.owner.capacity
        self.keys = key_states.new_zeros(shape)
        self.values = value_states.new_zeros(shape)
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        index = self.owner.positions[:, None, :, None].expand_as(key_states)
        self.keys.scatter_(2, index, key_states)
        self.values.scatter_(2, index, value_states)
        end = self.owner.end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def get_seq_length(self):
        return self.owner.end if self.is_initialized else 0

    def get_max_length(self):
        return self.owner.capacity

    def get_mask_sizes(self, query_length):
        return self.owner.end, 0


class _RowKVCache(Cache):
    def __init__(self, layers, capacity):
        self.capacity, self.end, self.positions = capacity, 0, None
        # Do not create a Python reference cycle retaining large device buffers
        # after a microbatch finishes.
        super().__init__(
            layers=[_RowKVLayer(weakref.proxy(self)) for _ in range(layers)]
        )

    def prepare(self, positions, end):
        if end > self.capacity or end <= 0:
            raise ValueError("Target cache capacity exceeded")
        self.positions, self.end = positions, end

    def select(self, indices):
        for layer in self.layers:
            if layer.is_initialized:
                layer.keys = layer.keys.index_select(0, indices)
                layer.values = layer.values.index_select(0, indices)


class BatchedTarget:
    """Dense Qwen3 target with independent committed lengths in each row."""

    def __init__(self, target, capacity):
        if not isinstance(target, LocalTarget):
            raise ValueError("Batched rollouts require the local frozen Qwen3 target")
        self.model, self.layer_ids = target.model, target.layer_ids
        config = self.model.config
        if config.model_type != "qwen3" or any(
            kind != "full_attention" for kind in config.layer_types
        ):
            raise ValueError("Batched target supports dense full-attention Qwen3 only")
        if config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Batched target requires eager or sdpa attention")
        self.cache = _RowKVCache(config.num_hidden_layers, capacity)
        self.calls = 0

    @torch.no_grad()
    def states(self, ids, lengths, *, prefill=False):
        batch, width = ids.shape
        if batch != len(lengths) or not lengths or min(lengths) < 0:
            raise ValueError("Invalid per-row target lengths")
        device = ids.device
        steps = torch.arange(width, device=device)
        sizes = torch.tensor(lengths, device=device)
        if prefill:
            if min(lengths) < 1 or max(lengths) > width:
                raise ValueError("Invalid padded prefill lengths")
            positions = steps[None].expand(batch, -1)
            end = width
            valid_keys = torch.arange(end, device=device)[None] < sizes[:, None]
        else:
            positions = sizes[:, None] + steps[None]
            end = max(lengths) + width
            valid_keys = torch.arange(end, device=device)[None] < (
                sizes[:, None] + width
            )
        visible = (
            torch.arange(end, device=device)[None, None] <= positions[:, :, None]
        ) & valid_keys[:, None]
        mask = torch.zeros(
            batch, 1, width, end, device=device, dtype=self.model.dtype
        ).masked_fill_(~visible[:, None], float("-inf"))
        self.cache.prepare(positions, end)
        out = self.model.model(
            ids,
            position_ids=positions,
            attention_mask={"full_attention": mask},
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self.calls += 1
        return TargetStates(
            torch.cat([out.hidden_states[i] for i in self.layer_ids], -1).detach(),
            out.last_hidden_state.detach(),
        )

    def select(self, indices):
        self.cache.select(indices)


class BatchedDraftCache:
    """No-grad context projections, separate absolute-position slots per row."""

    def __init__(self, draft, capacity):
        self.draft, self.capacity = draft, capacity
        self.keys = [None] * len(draft.layers)
        self.values = [None] * len(draft.layers)
        self.versions = tuple(p._version for p in draft.parameters())

    def _guard(self):
        if torch.is_grad_enabled() or self.draft.training:
            raise RuntimeError("Batched cache is restricted to eval/no-grad rollouts")
        if self.versions != tuple(p._version for p in self.draft.parameters()):
            raise RuntimeError("Discard the batched cache after a parameter update")

    def update(self, auxiliary, positions):
        self._guard()
        batch, width = auxiliary.shape[:2]
        if positions.shape != (batch, width):
            raise ValueError("Context write positions do not match hidden states")
        context = self.draft.hidden_norm(self.draft.fc(auxiliary.detach()))
        for i, layer in enumerate(self.draft.layers):
            attn = layer.self_attn
            key = attn.k_norm(
                attn.k_proj(context).view(batch, width, -1, attn.head_dim)
            )
            value = attn.v_proj(context).view(batch, width, -1, attn.head_dim)
            if self.keys[i] is None:
                shape = (batch, self.capacity, *key.shape[2:])
                self.keys[i], self.values[i] = (
                    key.new_zeros(shape),
                    value.new_zeros(shape),
                )
            index = positions[:, :, None, None].expand_as(key)
            self.keys[i].scatter_(1, index, key)
            self.values[i].scatter_(1, index, value)

    def select(self, indices):
        for i in range(len(self.keys)):
            if self.keys[i] is not None:
                self.keys[i] = self.keys[i].index_select(0, indices)
                self.values[i] = self.values[i].index_select(0, indices)

    def propose(self, lengths, anchor, memory, beta):
        self._guard()
        draft, batch = self.draft, len(lengths)
        if not lengths or min(lengths) < 1 or anchor.shape != (batch,):
            raise ValueError("Need matching nonempty draft prefixes and anchors")
        end, size, device = max(lengths), draft.block_size, anchor.device
        ids = torch.full(
            (batch, size), draft.mask_token_id, device=device, dtype=torch.long
        )
        ids[:, 0] = anchor
        hidden = draft.embed_tokens(ids)
        if draft.config.retrace_enabled:
            hidden = condition_query_block(draft.retrace, hidden, memory, beta)
        prefix = torch.arange(end, device=device)
        counts = torch.tensor(lengths, device=device)
        positions = torch.cat(
            (
                prefix[None].expand(batch, -1),
                counts[:, None] + torch.arange(size, device=device)[None],
            ),
            dim=1,
        )
        rotary = draft.rotary_emb(hidden, positions)
        visible = torch.cat(
            (
                prefix[None] < counts[:, None],
                torch.ones(batch, size, device=device, dtype=torch.bool),
            ),
            dim=1,
        )
        mask = hidden.new_zeros(batch, 1, size, end + size)
        mask.masked_fill_(~visible[:, None, None], float("-inf"))
        for i, layer in enumerate(draft.layers):
            attn, residual = layer.self_attn, hidden
            query_hidden = layer.input_layernorm(hidden)
            query = attn.q_norm(
                attn.q_proj(query_hidden).view(batch, size, -1, attn.head_dim)
            ).transpose(1, 2)
            key = attn.k_norm(
                attn.k_proj(query_hidden).view(batch, size, -1, attn.head_dim)
            )
            value = attn.v_proj(query_hidden).view(batch, size, -1, attn.head_dim)
            key = torch.cat((self.keys[i][:, :end], key), 1).transpose(1, 2)
            value = torch.cat((self.values[i][:, :end], value), 1).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *rotary)
            attended, _ = draft_attention(
                attn,
                query,
                key,
                value,
                mask,
                dropout=0.0,
                scaling=attn.scaling,
                sliding_window=attn.sliding_window,
            )
            hidden = residual + attn.o_proj(attended.reshape(batch, size, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = draft.norm(hidden)[:, 1:]
        logits = draft.lm_head(hidden)
        return DraftBlock(hidden, logits, logits.detach().argmax(-1))


@dataclass
class RolloutBatch:
    traces: list[TrainingTrajectory]
    target_forwards: int
    draft_forwards: int


@dataclass
class _Request:
    prompt: list[int]
    generated: list[int]
    positions: list[int] = field(default_factory=list)
    memories: list[ReTraceMemory] = field(default_factory=list)
    accepted: int = 0
    committed: int = 0


@torch.no_grad()
def collect_batched_trajectories(
    target, draft, prompts, response_length, eos, beta, *, trace_storage="device"
):
    """Complete every prompt in a microbatch before gradient replay.

    Finished rows are removed from both caches; original row identity is retained
    explicitly. One forward verifies all currently active trajectories. No clean
    future token, rejected branch label, or other request's memory is substituted.
    """
    if not prompts or any(not p for p in prompts) or response_length < 1:
        raise ValueError("Need nonempty prompts and a positive response budget")
    if trace_storage not in ("device", "cpu"):
        raise ValueError("Invalid trace storage")
    if draft.training:
        raise RuntimeError("Collect trajectories with the drafter in eval mode")
    target.reset()
    device = draft.device
    storage = device if trace_storage == "device" else "cpu"
    batch, size = len(prompts), draft.block_size
    k = size - 1
    lengths = [len(p) for p in prompts]
    capacity = max(lengths) + response_length + size
    teacher, cache = BatchedTarget(target, capacity), BatchedDraftCache(draft, capacity)
    ids = torch.zeros(batch, max(lengths), device=device, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        ids[i, : len(prompt)] = torch.tensor(prompt, device=device)
    pref = teacher.states(ids, lengths, prefill=True)
    rows = torch.arange(batch, device=device)
    last = torch.tensor(lengths, device=device) - 1
    anchors = draft.lm_head(pref.scoring[rows, last]).argmax(-1)
    anchor_list = anchors.tolist()
    requests = [
        _Request(list(p), [token])
        for p, token in zip(prompts, anchor_list, strict=True)
    ]
    context = pref.auxiliary.new_zeros(batch, capacity, pref.auxiliary.shape[-1])
    context[:, : ids.shape[1]] = pref.auxiliary
    cache.update(
        pref.auxiliary,
        torch.arange(ids.shape[1], device=device)[None].expand(batch, -1),
    )
    empty = torch.zeros(batch, k, draft.hidden_size, device=device, dtype=torch.float16)
    memory = ReTraceMemory(
        empty,
        torch.zeros_like(empty),
        torch.zeros(batch, k, device=device, dtype=torch.bool),
    )
    active = list(range(batch))
    traces = [None] * batch
    forwards = 0

    def retire_finished():
        nonlocal active, lengths, anchors, anchor_list, context, memory
        keep = []
        for row, index in enumerate(active):
            request = requests[index]
            if (
                len(request.generated) < response_length
                and request.generated[-1] not in eos
            ):
                keep.append(row)
                continue
            end = request.positions[-1] if request.positions else len(request.prompt)
            traces[index] = TrainingTrajectory(
                request.prompt + request.generated,
                len(request.prompt),
                context[row : row + 1, :end].detach().to(storage).clone(),
                request.positions,
                request.memories,
                request.accepted,
                k * len(request.positions),
                request.committed,
                len(request.positions) + 1,
            )
        if len(keep) != len(active):
            indices = torch.tensor(keep, device=device, dtype=torch.long)
            teacher.select(indices)
            cache.select(indices)
            context = context.index_select(0, indices)
            anchors = anchors.index_select(0, indices)
            memory = ReTraceMemory(
                *(
                    x.index_select(0, indices)
                    for x in (memory.draft, memory.target, memory.valid)
                )
            )
            lengths, active = [lengths[i] for i in keep], [active[i] for i in keep]
            anchor_list = [anchor_list[i] for i in keep]

    retire_finished()
    while active:
        for row, index in enumerate(active):
            request = requests[index]
            if lengths[row] != len(request.prompt) + len(request.generated) - 1:
                raise RuntimeError("Committed prefix and anchor position disagree")
            request.positions.append(lengths[row])
            request.memories.append(
                ReTraceMemory(
                    memory.draft[row : row + 1],
                    memory.target[row : row + 1],
                    memory.valid[row : row + 1],
                ).detached(device=storage)
            )
        block = cache.propose(lengths, anchors, memory, beta)
        forwards += 1
        query = torch.cat((anchors[:, None], block.token_ids), 1)
        verified = teacher.states(query, lengths)
        # Score the entire verification block in one batched head call, including
        # the possible bonus token. These states are proposal-aligned for memory.
        desired = draft.lm_head(verified.scoring).argmax(-1)
        accepted = (desired[:, :k] == block.token_ids).long().cumprod(-1).sum(-1)
        correction = desired.gather(1, accepted[:, None]).squeeze(1)
        memory = carry_suffix(
            block.hidden, verified.scoring[:, :k], accepted
        ).detached()
        proposals, accepted_list, correction_list = (
            block.token_ids.tolist(),
            accepted.tolist(),
            correction.tolist(),
        )
        old_lengths = lengths
        lengths = []
        for row, index in enumerate(active):
            request, count = requests[index], accepted_list[row]
            remaining = response_length - len(request.generated)
            emitted = []
            for token in proposals[row][:count] + [correction_list[row]]:
                if len(emitted) == remaining:
                    break
                emitted.append(token)
                if token in eos:
                    break
            request.generated.extend(emitted)
            request.accepted += min(count, len(emitted))
            request.committed += len(emitted)
            lengths.append(old_lengths[row] + count + 1)
        positions = (
            torch.tensor(old_lengths, device=device)[:, None]
            + torch.arange(size, device=device)[None]
        )
        context.scatter_(
            1, positions[:, :, None].expand_as(verified.auxiliary), verified.auxiliary
        )
        # Only committed context can be read. Extra row padding written here is
        # masked out and overwritten before it can become a committed position.
        width = max(accepted_list) + 1
        cache.update(verified.auxiliary[:, :width], positions[:, :width])
        anchors, anchor_list = correction, correction_list
        retire_finished()
    return RolloutBatch(traces, teacher.calls, forwards)
