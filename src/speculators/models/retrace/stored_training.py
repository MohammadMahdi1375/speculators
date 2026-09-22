"""DFlash's stored-sequence objective with fresh, verified ReTrace round pairs.

This is an explicitly different training sampler from full online trajectories:
sample an unconditioned source round, verify its real proposed branch, and train
the next conditioned round only where the committed prefix matches the stored
sequence. No clean future hidden states are relabelled as rejected-path states.
"""

from dataclasses import dataclass

import torch

from speculators.losses import compound_loss, resolve_loss_config
from speculators.model import SpeculatorModel

from .core import ReTraceDraftModel as RuntimeReTraceDraftModel
from .memory import ReTraceMemory, carry_suffix


@dataclass
class Pair:
    next_anchor: int
    accepted: int
    aligned: bool
    memory: ReTraceMemory


def verified_pair(tokens, position, proposal, hidden, scores, desired):
    """Align genuine verification memory only to the same committed token prefix.

    ``scores[:, j]`` scores proposal j (one-token causal shift already applied).
    Fully accepted blocks have no rejected memory, so no second block is needed.
    """
    k = len(proposal)
    desired_ids = desired[0].tolist()
    accepted = next((i for i in range(k) if proposal[i] != desired_ids[i]), k)
    accepted_tensor = torch.tensor([accepted], device=hidden.device)
    memory = carry_suffix(hidden, scores, accepted_tensor).detached()
    next_anchor = position + accepted + 1
    aligned = False
    if accepted < k and next_anchor < len(tokens):
        committed = proposal[:accepted] + [desired_ids[accepted]]
        aligned = tokens[position + 1 : next_anchor + 1] == committed
    return Pair(next_anchor, accepted, aligned, memory)


def empty_memory(count, k, h, device, dtype):
    zeros = torch.zeros(count, k, h, device=device, dtype=dtype)
    return ReTraceMemory(
        zeros,
        torch.zeros_like(zeros),
        torch.zeros(count, k, device=device, dtype=torch.bool),
    )


@SpeculatorModel.register("retrace_stored_training")
class ReTraceDraftModel(RuntimeReTraceDraftModel):
    """Training-only adapter; exported architecture remains ReTraceDraftModel.

    Importing this module adds a training-only registry alias; it does not
    replace the runtime's "retrace" registration. Exported configs retain
    algorithm="retrace" and the identical parameter/state-dict schema.
    """

    def configure_stored(
        self, target, *, pairs_per_batch=16, blocks_per_forward=4, seed=42, rank=0
    ):
        if min(pairs_per_batch, blocks_per_forward) < 1:
            raise ValueError("Positive pair and block counts required")
        self.branch_target = target
        self.pairs_per_batch = pairs_per_batch
        self.blocks_per_forward = blocks_per_forward
        self.stored_seed, self.stored_rank = seed, rank
        self.step_provider = lambda: 0
        self.config.training_objective = "stored_pair_kl"
        self.loss_config = resolve_loss_config("kl_div", "eager")

    def _documents(
        self,
        input_ids,
        loss_mask,
        document_ids,
        hidden_states,
        verifier_last_hidden_states,
    ):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Expected stock DFlash packed batch [1, sequence]")
        ids = input_ids[0].tolist()
        masks = loss_mask[0].bool().tolist()
        docs = document_ids[0].tolist()
        documents, candidates = [], []
        start = 0
        while start < len(ids):
            end = start + 1
            while end < len(ids) and docs[end] == docs[start]:
                end += 1
            if docs[start] >= 0:
                tokens, mask = ids[start:end], masks[start:end]
                doc = {
                    "tokens": tokens,
                    "mask": mask,
                    "aux": hidden_states[:, start:end],
                    "final": verifier_last_hidden_states[:, start:end],
                }
                doc_id = len(documents)
                documents.append(doc)
                # Source and next round need clean labels inside ONE response.
                # Each source has K verified proposals. Boundary tokens remain
                # context; no anchors/labels cross a masked span or document.
                for pos in range(1, len(tokens) - self.block_size):
                    if all(mask[pos : pos + self.block_size + 1]):
                        candidates.append((doc_id, pos))
            start = end
        return documents, candidates

    def forward(
        self,
        input_ids,
        hidden_states,
        verifier_last_hidden_states,
        loss_mask,
        document_ids,
        position_ids=None,
        **_,
    ):
        if not hasattr(self, "branch_target"):
            raise RuntimeError("Configure the stored-response training adapter first")
        documents, candidates = self._documents(
            input_ids,
            loss_mask,
            document_ids,
            hidden_states,
            verifier_last_hidden_states,
        )
        if not candidates:
            raise ValueError("No valid stored response blocks in this packed batch")
        step = int(self.step_provider())
        generator = torch.Generator().manual_seed(
            self.stored_seed + 1_000_003 * step + 97 * self.stored_rank
        )
        order = torch.randperm(len(candidates), generator=generator).tolist()
        chosen = [candidates[i] for i in order[: self.pairs_per_batch]]
        for doc in documents:
            doc["positions"], doc["memories"] = [], []
        pending = []
        device, dtype = input_ids.device, self.embed_tokens.weight.dtype
        k, h = self.block_size - 1, self.hidden_size
        beta = self.config.retrace_beta_max * min(
            1.0, step / max(1, self.config.retrace_warmup_steps)
        )
        if self.config.retrace_warmup_steps == 0:
            beta = self.config.retrace_beta_max
        # All requests across all documents are submitted before waiting. vLLM
        # schedules independent full-prefix prefills on the server NPUs.
        # The stock Trainer wraps the entire forward in BF16 autocast. Its
        # weight cache must not retain casts made by this no-grad proposal
        # pass: reusing them below detaches trainable backbone projections,
        # leaving DDP gradient buckets incomplete. Preserve the caller's
        # autocast precision and disable caching only for proposal collection.
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.get_autocast_dtype(device.type),
                enabled=torch.is_autocast_enabled(device.type),
                cache_enabled=False,
            ),
        ):
            for doc_id, doc in enumerate(documents):
                positions = [pos for idx, pos in chosen if idx == doc_id]
                for offset in range(0, len(positions), self.blocks_per_forward):
                    group = positions[offset : offset + self.blocks_per_forward]
                    pos = torch.tensor(group, device=device)
                    tokens = torch.tensor(doc["tokens"], device=device)
                    result = self.training_blocks(
                        doc["aux"], tokens[pos], pos, None, beta
                    )
                    proposals = result.token_ids.tolist()
                    for j, anchor in enumerate(group):
                        proposal = proposals[j]
                        future = self.branch_target.submit(
                            doc["tokens"][: anchor + 1] + proposal, anchor, k
                        )
                        pending.append(
                            (
                                doc,
                                anchor,
                                proposal,
                                result.hidden[j : j + 1].detach(),
                                future,
                            )
                        )
                        doc["positions"].append(anchor)
                        doc["memories"].append(empty_memory(1, k, h, device, dtype))
            accepted_sum = conditioned = aligned_pairs = unaligned_pairs = 0
            for doc, anchor, proposal, draft_hidden, future in pending:
                pre_norm = future.result().to(device=device, dtype=dtype)[None]
                scores = self.verifier_norm(pre_norm)
                desired = self.verifier_lm_head(scores).argmax(-1)
                pair = verified_pair(
                    doc["tokens"], anchor, proposal, draft_hidden, scores, desired
                )
                accepted_sum += pair.accepted
                usable = pair.aligned and bool(pair.memory.valid.any())
                if usable:
                    # At least one label after the new (correction) anchor.
                    usable = (
                        pair.next_anchor + 1 < len(doc["tokens"])
                        and doc["mask"][pair.next_anchor + 1]
                    )
                if usable:
                    doc["positions"].append(pair.next_anchor)
                    doc["memories"].append(pair.memory)
                    conditioned += int(pair.memory.valid.sum())
                    aligned_pairs += 1
                elif pair.accepted < k and not pair.aligned:
                    unaligned_pairs += 1

        # Count clean, response-only labels BEFORE reducing block chunks.
        denominator = 0
        for doc in documents:
            n = len(doc["tokens"])
            valid = []
            for pos in doc["positions"]:
                valid.append(
                    [
                        pos + j < n and all(doc["mask"][pos : pos + j + 1])
                        for j in range(1, k + 1)
                    ]
                )
            doc["valid"] = valid
            denominator += sum(sum(row) for row in valid)
        if not denominator:
            raise ValueError("No clean response labels")
        total_loss = torch.zeros((), device=device)
        label_correct = torch.zeros((), device=device)
        for doc in documents:
            for offset in range(0, len(doc["positions"]), self.blocks_per_forward):
                positions = doc["positions"][offset : offset + self.blocks_per_forward]
                pos = torch.tensor(positions, device=device)
                tokens = torch.tensor(doc["tokens"], device=device)
                items = doc["memories"][offset : offset + len(positions)]
                # FP16 stored memories must be concatenated outside BF16
                # autocast, then cast only for network math.
                with torch.autocast(device_type=device.type, enabled=False):
                    memory = ReTraceMemory(
                        torch.cat([m.draft.to(dtype=dtype) for m in items]),
                        torch.cat([m.target.to(dtype=dtype) for m in items]),
                        torch.cat([m.valid for m in items]),
                    )
                result = self.training_blocks(
                    doc["aux"], tokens[pos], pos, memory, beta
                )
                # Proposal j predicts token pos+j+1, scored by target pos+j.
                target_index = (pos[:, None] + torch.arange(k, device=device)).clamp(
                    max=len(doc["tokens"]) - 1
                )
                with torch.no_grad():
                    target_logits = self.verifier_lm_head(
                        self.verifier_norm(doc["final"][0, target_index])
                    )
                valid = torch.tensor(
                    doc["valid"][offset : offset + len(positions)],
                    device=device,
                    dtype=torch.bool,
                )
                # Reuse DFlash's loss functions and exact exp(-j/4) weighting.
                # Slots here exclude the anchor, so j starts at 0.
                slot = torch.arange(k, device=device).repeat(len(positions))[None]
                loss, _ = compound_loss(
                    result.logits.reshape(1, -1, self.draft_vocab_size),
                    target_logits.reshape(1, -1, self.draft_vocab_size),
                    valid.reshape(1, -1),
                    slot,
                    self.loss_config,
                    decay_fn=lambda index, **_: torch.exp(-index.float() / 4.0),
                )
                weight = (valid.sum() + 1e-5) / (denominator + 1e-5)
                total_loss = total_loss + loss * weight
                label_correct += (
                    (result.token_ids == target_logits.argmax(-1)) & valid
                ).sum()
        # Even an all-accepted or entirely unaligned batch must participate in
        # DDP for every conditioning parameter (zero gradient, not unused).
        for param in self.retrace.parameters():
            total_loss = total_loss + param.reshape(-1)[0] * 0.0
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Nonfinite stored-response KL loss")

        def scalar(value):
            return torch.as_tensor(float(value), device=device)

        metrics = {
            "loss_sum": total_loss.detach(),
            "loss_total": scalar(1),
            "clean_accuracy_sum": label_correct.detach(),
            "clean_accuracy_total": scalar(denominator),
            "accepted_per_source_sum": scalar(accepted_sum),
            "accepted_per_source_total": scalar(len(pending)),
            "conditioned_pair_fraction_sum": scalar(aligned_pairs),
            "conditioned_pair_fraction_total": scalar(len(pending)),
            "unaligned_pair_fraction_sum": scalar(unaligned_pairs),
            "unaligned_pair_fraction_total": scalar(len(pending)),
            "conditioned_positions_sum": scalar(conditioned),
            "conditioned_positions_total": scalar(1),
            "source_pairs_sum": scalar(len(pending)),
            "source_pairs_total": scalar(1),
            "clean_labels_sum": scalar(denominator),
            "clean_labels_total": scalar(1),
            "beta_sum": scalar(beta),
            "beta_total": scalar(1),
        }
        return None, total_loss, metrics
