"""Continuously refill local rollout slots within ONE fixed-weight update.

Only prompts belonging to the current global batch can enter a slot. Finished
traces have private storage before slots are reused. Live target and draft K/V
rows are compacted only on retirement; ordinary decode forwards use views, not
copies of complete caches. Empty rows are never sent through the transformer.
"""

import weakref
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache

from .batched_training import (
    BatchedDraftCache, BatchedTarget, RolloutBatch, _Request, _RowKVLayer,
    collect_batched_trajectories,
)
from .clean_training import TrainingTrajectory, collect_trajectory
from .memory import ReTraceMemory, carry_suffix


class _ReusableKVLayer(_RowKVLayer):
    def lazy_initialization(self, key_states, value_states):
        shape = list(key_states.shape)
        shape[0], shape[-2] = self.owner.slots, self.owner.capacity
        self.keys, self.values = key_states.new_zeros(shape), value_states.new_zeros(shape)
        self.dtype, self.device, self.is_initialized = key_states.dtype, key_states.device, True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        start, count = self.owner.row_start, key_states.shape[0]
        index = self.owner.positions[:, None, :, None].expand_as(key_states)
        keys, values = self.keys[start:start + count], self.values[start:start + count]
        keys.scatter_(2, index, key_states)
        values.scatter_(2, index, value_states)
        return keys[:, :, :self.owner.end], values[:, :, :self.owner.end]


class _ReusableKVCache(Cache):
    def __init__(self, layers, capacity, slots):
        self.capacity, self.slots = capacity, slots
        self.end, self.positions, self.row_start = 0, None, 0
        super().__init__(layers=[_ReusableKVLayer(weakref.proxy(self)) for _ in range(layers)])

    def prepare(self, positions, end):
        if not 0 < end <= self.capacity or self.row_start + positions.shape[0] > self.slots:
            raise ValueError("Continuous target cache capacity exceeded")
        self.positions, self.end = positions, end

    def compact(self, indices):
        if not indices.numel():
            return
        for layer in self.layers:
            if layer.is_initialized:
                layer.keys[:len(indices)].copy_(layer.keys.index_select(0, indices))
                layer.values[:len(indices)].copy_(layer.values.index_select(0, indices))


class ReusableTarget(BatchedTarget):
    def __init__(self, target, capacity, slots):
        super().__init__(target, capacity)
        self.cache = _ReusableKVCache(self.model.config.num_hidden_layers, capacity, slots)

    def states(self, ids, lengths, *, prefill=False, row_start=0):
        self.cache.row_start = row_start
        return super().states(ids, lengths, prefill=prefill)


class ReusableDraft(BatchedDraftCache):
    def __init__(self, draft, capacity, slots):
        super().__init__(draft, capacity)
        self.slots = slots

    def update(self, auxiliary, positions, row_start=0):
        self._guard()
        batch, width = auxiliary.shape[:2]
        if positions.shape != (batch, width) or row_start + batch > self.slots:
            raise ValueError("Invalid continuous draft cache write")
        context = self.draft.hidden_norm(self.draft.fc(auxiliary.detach()))
        for index, layer in enumerate(self.draft.layers):
            attn = layer.self_attn
            key = attn.k_norm(attn.k_proj(context).view(batch, width, -1, attn.head_dim))
            value = attn.v_proj(context).view(batch, width, -1, attn.head_dim)
            if self.keys[index] is None:
                shape = (self.slots, self.capacity, *key.shape[2:])
                self.keys[index], self.values[index] = key.new_zeros(shape), value.new_zeros(shape)
            positions_expanded = positions[:, :, None, None].expand_as(key)
            self.keys[index][row_start:row_start + batch].scatter_(1, positions_expanded, key)
            self.values[index][row_start:row_start + batch].scatter_(1, positions_expanded, value)

    def compact(self, indices):
        if not indices.numel():
            return
        for key, value in zip(self.keys, self.values, strict=True):
            if key is not None:
                key[:len(indices)].copy_(key.index_select(0, indices))
                value[:len(indices)].copy_(value.index_select(0, indices))

    def propose(self, lengths, anchors, memory, beta):
        # A slice is a view: inactive slot capacity incurs no transformer work.
        all_keys, all_values = self.keys, self.values
        self.keys = [key[:len(lengths)] for key in all_keys]
        self.values = [value[:len(lengths)] for value in all_values]
        try:
            return super().propose(lengths, anchors, memory, beta)
        finally:
            self.keys, self.values = all_keys, all_values


@dataclass
class CompletedRollouts:
    positions: list[int]
    indices: list[int]
    batch: RolloutBatch


class ContinuousRollouts:
    """next_ready does only no-grad work and returns traces ready for replay.

    It deliberately is not a generator decorated with no_grad: the caller may
    run gradient replay between calls without inheriting a suspended context.
    """

    def __init__(self, target, draft, groups, prompts, response_length, eos, beta,
                 *, max_active=2, trace_storage="device"):
        if max_active < 1 or response_length < 1 or trace_storage not in {"cpu", "device"}:
            raise ValueError("Invalid continuous rollout settings")
        self.draft, self.groups, self.prompts = draft, iter(groups), prompts
        self.response_length, self.eos, self.beta = response_length, eos, beta
        self.device = draft.embed_tokens.weight.device
        self.storage = self.device if trace_storage == "device" else torch.device("cpu")
        self.max_active, self.size, self.k = max_active, draft.block_size, draft.block_size - 1
        # The pool's maximum prompt length is a conservative per-update bound.
        capacity = max(map(len, prompts)) + response_length + self.size
        target.reset()
        self.teacher = ReusableTarget(target, capacity, max_active)
        self.cache = ReusableDraft(draft, capacity, max_active)
        self.capacity, self.context = capacity, None
        self.requests, self.identities, self.lengths = [], [], []
        self.anchors = torch.empty(0, dtype=torch.long, device=self.device)
        empty = torch.zeros(0, self.k, draft.hidden_size, device=self.device, dtype=torch.float16)
        self.memory = ReTraceMemory(empty, empty.clone(), torch.zeros(0, self.k, device=self.device, dtype=torch.bool))
        self.exhausted, self.draft_forwards = False, 0
        self.reported_target, self.reported_draft = 0, 0

    def _fill(self):
        new = []
        while len(self.requests) + len(new) < self.max_active and not self.exhausted:
            group = next(self.groups, None)
            if group is None:
                self.exhausted = True
                break
            positions, indices = group
            if len(positions) != 1 or len(indices) != 1:
                raise ValueError("Continuous rollouts require one-prompt queue claims")
            new.append((positions[0], indices[0]))
        if not new:
            return
        start = len(self.requests)
        prompts = [self.prompts[index] for _, index in new]
        lengths = [len(prompt) for prompt in prompts]
        ids = torch.zeros(len(new), max(lengths), device=self.device, dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)] = torch.tensor(prompt, device=self.device)
        pref = self.teacher.states(ids, lengths, prefill=True, row_start=start)
        if self.context is None:
            self.context = pref.auxiliary.new_zeros(self.max_active, self.capacity, pref.auxiliary.shape[-1])
        self.context[start:start + len(new), :ids.shape[1]].copy_(pref.auxiliary)
        positions = torch.arange(ids.shape[1], device=self.device)[None].expand(len(new), -1)
        self.cache.update(pref.auxiliary, positions, row_start=start)
        last = torch.tensor(lengths, device=self.device) - 1
        anchor = self.draft.lm_head(pref.scoring[torch.arange(len(new), device=self.device), last]).argmax(-1)
        for prompt, token in zip(prompts, anchor.tolist(), strict=True):
            self.requests.append(_Request(list(prompt), [token]))
        self.identities.extend(new)
        self.lengths.extend(lengths)
        self.anchors = torch.cat((self.anchors, anchor))
        extra = torch.zeros(len(new), self.k, self.draft.hidden_size, device=self.device, dtype=torch.float16)
        # FP16 is the paper's detached storage format, not the BF16 forward dtype.
        # Concatenating these storage buffers must not enter BF16 autocast.
        with torch.autocast(device_type=self.device.type, enabled=False):
            self.memory = ReTraceMemory(
                torch.cat((self.memory.draft, extra)), torch.cat((self.memory.target, extra)),
                torch.cat((self.memory.valid, torch.zeros(len(new), self.k, device=self.device, dtype=torch.bool))),
            )

    def _retire(self):
        ready, identities, keep = [], [], []
        for row, request in enumerate(self.requests):
            if len(request.generated) < self.response_length and request.generated[-1] not in self.eos:
                keep.append(row)
                continue
            end = request.positions[-1] if request.positions else len(request.prompt)
            ready.append(TrainingTrajectory(
                request.prompt + request.generated, len(request.prompt),
                self.context[row:row + 1, :end].detach().to(self.storage).clone(),
                request.positions, request.memories, request.accepted,
                self.k * len(request.positions), request.committed, len(request.positions) + 1,
            ))
            identities.append(self.identities[row])
        if not ready:
            return None
        rows = torch.tensor(keep, device=self.device, dtype=torch.long)
        if keep:
            self.teacher.cache.compact(rows)
            self.cache.compact(rows)
            self.context[:len(keep)].copy_(self.context.index_select(0, rows))
        self.requests = [self.requests[i] for i in keep]
        self.identities = [self.identities[i] for i in keep]
        self.lengths = [self.lengths[i] for i in keep]
        self.anchors = self.anchors.index_select(0, rows)
        self.memory = ReTraceMemory(*(x.index_select(0, rows) for x in (
            self.memory.draft, self.memory.target, self.memory.valid)))
        result = CompletedRollouts(
            [x[0] for x in identities], [x[1] for x in identities],
            RolloutBatch(ready, self.teacher.calls - self.reported_target,
                         self.draft_forwards - self.reported_draft),
        )
        self.reported_target, self.reported_draft = self.teacher.calls, self.draft_forwards
        return result

    @torch.no_grad()
    def next_ready(self):
        if self.draft.training:
            raise RuntimeError("Set the draft to eval before collecting a rollout")
        self.cache._guard()
        while True:
            self._fill()
            if not self.requests:
                return None
            ready = self._retire()
            if ready is not None:
                return ready
            for row, request in enumerate(self.requests):
                if self.lengths[row] != len(request.prompt) + len(request.generated) - 1:
                    raise RuntimeError("Continuous prefix/anchor alignment differs")
                request.positions.append(self.lengths[row])
                request.memories.append(ReTraceMemory(
                    self.memory.draft[row:row + 1], self.memory.target[row:row + 1],
                    self.memory.valid[row:row + 1],
                ).detached(device=self.storage))
            block = self.cache.propose(self.lengths, self.anchors, self.memory, self.beta)
            self.draft_forwards += 1
            verified = self.teacher.states(torch.cat((self.anchors[:, None], block.token_ids), 1), self.lengths)
            desired = self.draft.lm_head(verified.scoring).argmax(-1)
            accepted = (desired[:, :self.k] == block.token_ids).long().cumprod(-1).sum(-1)
            correction = desired.gather(1, accepted[:, None]).squeeze(1)
            self.memory = carry_suffix(block.hidden, verified.scoring[:, :self.k], accepted).detached()
            # One host transfer replaces separate proposal/acceptance/correction synchronizations.
            host = torch.cat((block.token_ids, accepted[:, None], correction[:, None]), 1).tolist()
            old_lengths, lengths, accepted_list = self.lengths, [], []
            for row, request in enumerate(self.requests):
                proposal, count, bonus = host[row][:self.k], host[row][-2], host[row][-1]
                accepted_list.append(count)
                remaining, emitted = self.response_length - len(request.generated), []
                for token in proposal[:count] + [bonus]:
                    if len(emitted) == remaining:
                        break
                    emitted.append(token)
                    if token in self.eos:
                        break
                request.generated.extend(emitted)
                request.accepted += min(count, len(emitted))
                request.committed += len(emitted)
                lengths.append(old_lengths[row] + count + 1)
            positions = torch.tensor(old_lengths, device=self.device)[:, None] + torch.arange(self.size, device=self.device)[None]
            self.context[:len(self.requests)].scatter_(1, positions[:, :, None].expand_as(verified.auxiliary), verified.auxiliary)
            width = max(accepted_list) + 1
            self.cache.update(verified.auxiliary[:, :width], positions[:, :width])
            self.lengths, self.anchors = lengths, correction
            ready = self._retire()
            if ready is not None:
                return ready


class CohortRollouts:
    """Adapter retaining the previous serial/cohort path for paired comparison."""

    def __init__(self, target, draft, groups, prompts, args, eos, beta):
        self.target, self.draft, self.groups, self.prompts = target, draft, iter(groups), prompts
        self.args, self.eos, self.beta = args, eos, beta

    def next_ready(self):
        group = next(self.groups, None)
        if group is None:
            return None
        positions, indices = group
        if self.args.rollout_batch_size > 1:
            result = collect_batched_trajectories(
                self.target, self.draft, [self.prompts[i] for i in indices],
                self.args.response_length, self.eos, self.beta, trace_storage=self.args.trace_storage)
        else:
            trace = collect_trajectory(
                self.target, self.draft, self.prompts[indices[0]], self.args.response_length,
                self.eos, self.beta, performance_mode=self.args.performance_mode,
                trace_storage=self.args.trace_storage)
            result = RolloutBatch([trace], trace.target_calls, len(trace.positions))
        return CompletedRollouts(positions, indices, result)
