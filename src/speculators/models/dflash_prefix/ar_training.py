"""Training-only causal next-token objective for the shared DFlash backbone.

For an anchor at a, the clean inputs are x[a:a+B-1] and the labels are
x[a+1:a+B]. Queries see only earlier/equal clean inputs in their own block
and frozen target features strictly BEFORE a. The shifted labels, residual
path and target-feature mask must all agree to prevent answer leakage.
"""

import math

import torch
import torch.distributed as dist
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from speculators.models.dflash.attention import create_anchor_block_mask_mod


def validate_ar_settings(settings):
    """Validate optional settings; older run metadata means AR is disabled."""
    weight = settings.get("ar_loss_weight", 0.0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("ar_loss_weight must be finite and nonnegative")
    for name, default, minimum in (
        ("ar_pretrain_steps", 0, 0),
        ("ar_max_anchors", 64, 1),
        ("ar_logit_chunk_size", 128, 1),
    ):
        value = settings.get(name, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if settings.get("ar_pretrain_steps", 0) and weight == 0:
        raise ValueError("AR pretraining requires a positive joint ar_loss_weight")
    if type(settings.get("ar_logit_checkpointing", True)) is not bool:
        raise ValueError("ar_logit_checkpointing must be boolean")


def create_ar_block_mask_mod(
    document_ids, total_seq_len, anchors, proposals, sliding_window=None
):
    """Causal clean blocks, isolated from one another and from future features."""
    base, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids,
        total_seq_len,
        anchors,
        proposals,
        sliding_window=sliding_window,
        sliding_window_non_causal=False,
    )

    def mask(batch, head, query, key):
        # Upstream full attention is deliberately bidirectional. Enforce
        # causality here even when sliding_window is None.
        causal = (key < total_seq_len) | (key - total_seq_len <= query)
        return base(batch, head, query, key) & causal

    return mask, q_len, kv_len


def ar_backbone_hidden(  # noqa: PLR0917
    model, hidden_states, input_ids, document_ids, block_indices, position_ids=None
):
    """Return next-token predictor states [1, anchors*(B-1), H].

    This uses existing modules only: no new serving weights, no target forward,
    and no clean-to-parallel attention edges. Blocks may overlap in the packed
    sequence, but their clean streams remain independent in every draft layer.
    """
    device = hidden_states.device
    total = input_ids.shape[1]
    proposals = block_indices.shape[1] - 1
    anchors = block_indices[:, 0]
    predictor_indices = block_indices[:, :-1].reshape(-1)
    if position_ids is None:
        position_ids = torch.arange(total, device=device).unsqueeze(0)
    positions = torch.cat((position_ids, position_ids[:, predictor_indices]), dim=1)
    rotary = model.rotary_emb(hidden_states, positions)
    clean = model.embed_tokens(input_ids[:, predictor_indices])
    context = model.hidden_norm(model.fc(hidden_states))
    masks = {}
    for window in ({None} if model.uses_full_attn else set()) | (
        {model.sliding_window} if model.uses_sliding_window_attn else set()
    ):
        mask, q_len, kv_len = create_ar_block_mask_mod(
            document_ids.squeeze(0),
            total,
            anchors,
            proposals,
            window,
        )
        masks[window] = model._create_mask_fn(  # noqa: SLF001
            mask,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
    for index, layer in enumerate(model.layers):
        window = model.sliding_window if index in model.sliding_window_indices else None
        clean = layer(
            hidden_states=clean,
            target_hidden=context,
            attention_mask=masks[window],
            position_ids=positions,
            use_cache=False,
            position_embeddings=rotary,
        )
    return model.norm(clean)


def ddp_token_mean(local_sum, local_count):
    """Scale a local differentiable sum for DDP's gradient averaging.

    All ranks participate, including ranks with zero valid AR labels. Reporting
    uses the original local numerator/count, not this gradient-scaled quantity.
    """
    count = local_count.detach().clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return local_sum * world / count.clamp_min(1)


def causal_ar_loss(  # noqa: PLR0917
    model, hidden_states, input_ids, document_ids, indices, valid, position_ids=None
):
    """Standard CE on reference next tokens, pooled over valid tokens/ranks.

    Use an evenly spaced subset of the already sampled anchor slots, without
    additional anchor-sampling RNG. Cap
    the auxiliary workload and checkpoint chunked vocabulary projections to avoid
    retaining another full [anchors, B-1, vocabulary] logits tensor for backward.
    """
    settings = model.joint_settings
    blocks = indices.reshape(-1, model.block_size)
    count = min(blocks.shape[0], settings.get("ar_max_anchors", 64))
    rows = torch.div(
        torch.arange(count, device=indices.device) * blocks.shape[0],
        count,
        rounding_mode="floor",
    )
    blocks, valid = blocks[rows], valid[rows].reshape(-1)
    hidden = ar_backbone_hidden(
        model, hidden_states, input_ids, document_ids, blocks, position_ids
    ).reshape(-1, model.norm.weight.numel())
    labels = input_ids[0, blocks[:, 1:].reshape(-1)]

    def project_and_score(states, targets, mask):
        logits = model.lm_head(states).float()
        per_token = functional.cross_entropy(logits, targets, reduction="none")
        loss_sum = (per_token * mask).sum()
        correct = (logits.detach().argmax(-1).eq(targets) & mask).float().sum()
        return loss_sum, correct

    sums, correct = [], []
    chunk = settings.get("ar_logit_chunk_size", 128)
    recompute = (
        model.training
        and torch.is_grad_enabled()
        and settings.get("ar_logit_checkpointing", True)
    )
    for start in range(0, labels.numel(), chunk):
        args = (
            hidden[start : start + chunk],
            labels[start : start + chunk],
            valid[start : start + chunk],
        )
        numerator, accuracy = (
            checkpoint(project_and_score, *args, use_reentrant=False)
            if recompute
            else project_and_score(*args)
        )
        sums.append(numerator)
        correct.append(accuracy)
    numerator = torch.stack(sums).sum()
    denominator = valid.float().sum()
    metrics = {
        "ar_ce_sum": numerator.detach(),
        "ar_ce_total": denominator.detach(),
        "ar_reference_acc_sum": torch.stack(correct).sum(),
        "ar_reference_acc_total": denominator.detach().clone(),
    }
    loss = (
        ddp_token_mean(numerator, denominator)
        if model.training
        else numerator / denominator.clamp_min(1)
    )
    return loss, metrics
