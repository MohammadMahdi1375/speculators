"""Clean-token CE on stored answers and short, recurrent ReTrace chains.

The frozen target verifies actual proposed branches. A chain advances only if
ALL newly committed tokens match the stored answer. After a mismatch, it stops:
clean-context features are never reused for a different committed prefix.

Four rounds per prompt bound the work independently of full response length.
This sampler is an explicit implementation choice; it is not published as the
authors' exact Table 2 training sampler.
"""

from dataclasses import dataclass

import torch
from torch import nn

from speculators.model import SpeculatorModel
from ..paper_recipe.clean_training import block_ce_sum
from ..paper_recipe.core import ReTraceDraftModel as RuntimeModel
from ..paper_recipe.memory import ReTraceMemory, carry_suffix


@dataclass
class Successor:
    position: int
    accepted: int
    aligned: bool
    memory: ReTraceMemory


def successor(tokens, anchor, proposals, draft_hidden, score_hidden, desired):
    """One-token-shifted target scores, including the full-accept bonus row."""
    k = len(proposals)
    if score_hidden.shape[1] != k + 1 or desired.shape != (1, k + 1):
        raise ValueError("Need K verification scoring states plus the bonus state")
    target_ids = desired[0].tolist()
    accepted = next((j for j in range(k) if proposals[j] != target_ids[j]), k)
    next_anchor = anchor + accepted + 1
    committed = proposals[:accepted] + [target_ids[accepted]]
    aligned = (
        next_anchor < len(tokens)
        and tokens[anchor + 1 : next_anchor + 1] == committed
    )
    memory = carry_suffix(
        draft_hidden,
        score_hidden[:, :k],
        torch.tensor([accepted], device=draft_hidden.device),
    ).detached()  # FP16, detached, and only the immediately preceding round.
    return Successor(next_anchor, accepted, aligned, memory)


def empty_memory(k, hidden, device):
    value = torch.zeros(1, k, hidden, device=device, dtype=torch.float16)
    return ReTraceMemory(value, torch.zeros_like(value), torch.zeros(1, k, device=device, dtype=torch.bool))


def combine_memory(items, dtype, device):
    # Do not feed FP16 concatenation into BF16 autocast's cached math path.
    with torch.autocast(device_type=device.type, enabled=False):
        return ReTraceMemory(
            torch.cat([item.draft.to(device=device, dtype=dtype) for item in items]),
            torch.cat([item.target.to(device=device, dtype=dtype) for item in items]),
            torch.cat([item.valid.to(device=device) for item in items]),
        )


def response_labels(tokens, mask, positions, k, device):
    rows = []
    for position in positions:
        row, valid_prefix = [], True
        for j in range(1, k + 1):
            index = position + j
            valid_prefix = valid_prefix and index < len(tokens) and bool(mask[index])
            row.append(tokens[index] if valid_prefix else -100)
        rows.append(row)
    return torch.tensor(rows, device=device, dtype=torch.long)


@SpeculatorModel.register("retrace_stored_ce_training")
class ReTraceDraftModel(RuntimeModel):
    def configure_stored(self, target, *, chain_rounds=4, chains_per_prompt=1,
                         blocks_per_forward=4, global_prompt_batch=32,
                         world_size=1, rank=0, seed=42):
        if min(chain_rounds, chains_per_prompt, blocks_per_forward, global_prompt_batch, world_size) < 1:
            raise ValueError("Positive training budgets required")
        if chain_rounds < 2:
            raise ValueError("ReTrace training needs at least two rounds per chain")
        self.branch_target = target
        self.chain_rounds, self.chains_per_prompt = chain_rounds, chains_per_prompt
        self.blocks_per_forward = blocks_per_forward
        self.global_prompt_batch, self.world_size = global_prompt_batch, world_size
        self.rank, self.seed = rank, seed
        self.step_provider = lambda: 0
        self.config.training_objective = "clean_block_ce"
        self.config.stored_training_recipe = "stored_chain_clean_ce_v2"
        self.config.stored_chain_rounds = chain_rounds
        self.config.stored_chains_per_prompt = chains_per_prompt
        self.config.stored_sampler_is_author_validated = False
        # Stock Trainer.step controls dropout. ReTrace's detached proposal and
        # gradient replay must evaluate the same deterministic block function.
        for module in self.modules():
            if isinstance(module, nn.Dropout) and module.p != 0:
                raise ValueError("Nonzero dropout would mismatch proposal and replay")

    def _documents(self, input_ids, loss_mask, document_ids, hidden_states):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Expected packed [1, sequence] features")
        ids, mask, docids = input_ids[0].tolist(), loss_mask[0].bool().tolist(), document_ids[0].tolist()
        documents, start = [], 0
        while start < len(ids):
            end = start + 1
            while end < len(ids) and docids[end] == docids[start]:
                end += 1
            if docids[start] >= 0:
                tokens, losses = ids[start:end], mask[start:end]
                # A source anchor is a known response token. Require K clean
                # successor labels initially; later chain rounds may use tails.
                candidates = [position for position in range(1, len(tokens) - self.block_size + 1)
                              if all(losses[position : position + self.block_size])]
                if not candidates:
                    raise ValueError("A stored prompt has no eligible source anchor")
                documents.append(dict(tokens=tokens, mask=losses, candidates=candidates,
                                      aux=hidden_states[:, start:end], blocks=[]))
            start = end
        if not documents:
            raise ValueError("Empty prompt batch")
        return documents

    def forward(self, input_ids, hidden_states, verifier_last_hidden_states,
                loss_mask, document_ids, prompt_count=None, **_):
        if not hasattr(self, "branch_target"):
            raise RuntimeError("Stored-chain adapter was not configured")
        docs = self._documents(input_ids, loss_mask, document_ids, hidden_states)
        if prompt_count is not None and len(docs) != prompt_count:
            raise ValueError("Packed prompt count changed")
        device, dtype = input_ids.device, self.embed_tokens.weight.dtype
        k, h = self.block_size - 1, self.hidden_size
        step = int(self.step_provider())
        warmup = self.config.retrace_warmup_steps
        beta = self.config.retrace_beta_max * (min(1.0, step / warmup) if warmup else 1.0)
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * step + 97 * self.rank)
        active = []
        for index, doc in enumerate(docs):
            order = torch.randperm(len(doc["candidates"]), generator=generator).tolist()
            for source in order[:self.chains_per_prompt]:
                active.append((index, doc["candidates"][source], empty_memory(k, h, device)))
        source_chains = len(active)
        requests = accepted_sum = aligned_pairs = unaligned_pairs = conditioned_positions = 0
        reused_source_rounds = trained_blocks = conditioned_blocks = 0
        # Nested autocast with CACHE DISABLED is necessary: the outer stock
        # Trainer autocast scope also covers the subsequent gradient replay.
        with torch.no_grad(), torch.autocast(
            device.type, dtype=torch.get_autocast_dtype(device.type),
            enabled=torch.is_autocast_enabled(device.type), cache_enabled=False,
        ):
            for depth in range(self.chain_rounds):
                pending = []
                for doc_id, doc in enumerate(docs):
                    items = [(anchor, mem) for index, anchor, mem in active if index == doc_id]
                    for offset in range(0, len(items), self.blocks_per_forward):
                        group = items[offset : offset + self.blocks_per_forward]
                        positions = [anchor for anchor, _ in group]
                        memories = [mem for _, mem in group]
                        for anchor, mem in group:
                            doc["blocks"].append((anchor, mem))
                            trained_blocks += 1
                            conditioned_blocks += int(bool(mem.valid.any()))
                            conditioned_positions += int(mem.valid.sum())
                        if depth == self.chain_rounds - 1:
                            continue  # No successor consumes the last verification.
                        pos = torch.tensor(positions, device=device)
                        tokens = torch.tensor(doc["tokens"], device=device)
                        result = self.training_blocks(
                            doc["aux"][:, :max(positions)], tokens[pos], pos,
                            combine_memory(memories, dtype, device), beta,
                        )
                        proposals = result.token_ids.tolist()
                        for j, (anchor, mem) in enumerate(group):
                            future = self.branch_target.submit(
                                doc["tokens"][:anchor + 1] + proposals[j], anchor, k,
                            )
                            requests += 1
                            reused_source_rounds += int(bool(mem.valid.any()))
                            pending.append((doc_id, anchor, proposals[j], result.hidden[j:j+1].detach(), future))
                active = []
                # Submit independent branches for every document before waiting;
                # the two vLLM replicas batch these requests across trainer ranks.
                for doc_id, anchor, proposal, draft_hidden, future in pending:
                    doc = docs[doc_id]
                    pre_norm = future.result().to(device=device, dtype=dtype)[None]
                    scores = self.verifier_norm(pre_norm)
                    desired = self.verifier_lm_head(scores).argmax(-1)
                    pair = successor(doc["tokens"], anchor, proposal, draft_hidden, scores, desired)
                    accepted_sum += pair.accepted
                    if not pair.aligned:
                        # Crossing the end is a completed chain, not a disagreement.
                        if pair.position < len(doc["tokens"]):
                            unaligned_pairs += 1
                        continue
                    if pair.position + 1 < len(doc["tokens"]) and doc["mask"][pair.position + 1]:
                        active.append((doc_id, pair.position, pair.memory))
                        aligned_pairs += int(bool(pair.memory.valid.any()))
                if not active:
                    break

        prompt_loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        label_count = 0
        for doc in docs:
            positions = [anchor for anchor, _ in doc["blocks"]]
            labels = response_labels(doc["tokens"], doc["mask"], positions, k, device)
            count = int((labels != -100).sum())
            if not count:
                raise ValueError("Prompt has no supervised clean labels")
            label_count += count
            tokens = torch.tensor(doc["tokens"], device=device)
            for offset in range(0, len(positions), self.blocks_per_forward):
                group = positions[offset:offset + self.blocks_per_forward]
                memories = [mem for _, mem in doc["blocks"][offset:offset + len(group)]]
                pos = torch.tensor(group, device=device)
                result = self.training_blocks(
                    doc["aux"][:, :max(group)], tokens[pos], pos,
                    combine_memory(memories, dtype, device), beta,
                )
                targets = labels[offset:offset + len(group)]
                prompt_loss_sum = prompt_loss_sum + block_ce_sum(result.logits, targets) / (count + 1e-5)
                correct += ((result.token_ids == targets) & (targets != -100)).sum()
        # DDP averages rank gradients. Scale local PROMPT SUMS, not rank means.
        # For 6 ranks and batch32 the 5/6-prompt assignments get exact weighting.
        loss = prompt_loss_sum * self.world_size / self.global_prompt_batch
        for parameter in self.retrace.parameters():
            loss = loss + parameter.reshape(-1)[0] * 0.0
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite stored-chain CE loss")

        def scalar(value):
            return torch.as_tensor(value, device=device, dtype=torch.float32).detach()

        counters = {
            "loss": (prompt_loss_sum.detach(), len(docs)),
            "clean_accuracy": (correct.detach(), label_count),
            "accepted_per_source": (accepted_sum, requests),
            "conditioned_pair_fraction": (aligned_pairs, requests),
            "unaligned_pair_fraction": (unaligned_pairs, requests),
            "conditioned_positions": (conditioned_positions, 1),
            "source_pairs": (requests, 1),
            "clean_labels": (label_count, 1),
            "beta": (beta, 1),
            "global_prompts": (len(docs), int(self.rank == 0)),
            "branch_requests_global": (requests, int(self.rank == 0)),
            "trained_blocks_global": (trained_blocks, int(self.rank == 0)),
            "source_chains_global": (source_chains, int(self.rank == 0)),
            "conditioned_block_fraction": (conditioned_blocks, trained_blocks),
            "recurrent_source_fraction": (reused_source_rounds, requests),
        }
        return None, loss, {f"{name}_{suffix}": scalar(value)
                            for name, values in counters.items()
                            for suffix, value in zip(("sum", "total"), values)}
