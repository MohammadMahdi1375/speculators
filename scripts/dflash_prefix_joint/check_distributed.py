#!/usr/bin/env python3
"""Tiny distributed training preflight; no real model or dataset is loaded."""
import argparse
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace


def check_agreement(model, *, phase, step, local, gradients=False):
    """Compare copies only, without repairing or rounding the live tensors."""
    import torch
    import torch.distributed as dist
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    values = [p.grad if gradients else p.detach() for _, p in named]
    flat = torch.cat([(torch.full_like(p, float("nan")) if value is None else value.detach())
                      .reshape(-1).float() for (_, p), value in zip(named, values)])
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    mismatch = flat.ne(reference) | ~torch.isfinite(flat) | ~torch.isfinite(reference)
    count = int(mismatch.sum().item())
    bad, offset = [], 0
    if count:
        for (name, p), value in zip(named, values):
            end = offset + p.numel()
            changed = int(mismatch[offset:end].sum().item())
            if changed and len(bad) < 8:
                finite = bool(torch.isfinite(flat[offset:end]).all() & torch.isfinite(reference[offset:end]).all())
                error = float((flat[offset:end] - reference[offset:end]).abs().max().item()) if finite else None
                bad.append(dict(name=name, dtype=str(p.dtype), elements_changed=changed,
                                max_abs_error=error, finite=finite, gradient_missing=value is None))
            offset = end
    report = dict(rank=dist.get_rank(), step=step, phase=phase, gradients=gradients,
                  pass_local=count == 0, elements_changed=count, first_differences=bad)
    with (local / "phases.jsonl").open("a") as stream:
        stream.write(json.dumps(report, allow_nan=False) + "\n")
    passed = torch.tensor(int(count == 0), device=flat.device, dtype=torch.int32)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if not passed.item():
        (local / "failure.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("JOINT_DISTRIBUTED_MISMATCH " + json.dumps(report, allow_nan=False), flush=True)
        raise RuntimeError(f"Distributed disagreement at step={step}, phase={phase}; see {local}/failure.json")


def exercise(rank, world_size, directory, device, settings):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from check_training import make_fixture
    from train_entry import install_hooks
    from speculators.losses import resolve_loss_config
    from speculators.models.dflash_prefix.joint_guard import bf16_replay_copy
    from speculators.models.dflash_prefix.joint_validation import checkpoint_parameters
    from speculators.train.checkpointer import SingleGPUCheckpointer
    local = Path(directory) / f"rank_{rank}"
    local.mkdir(parents=True, exist_ok=False)
    s = dict(settings, block_size=settings["block_size"], delay_steps=0, head_warmup_steps=1,
        focus_start_steps=0, focus_warmup_steps=1,
        anchor_chunk_size=settings.get("anchor_chunk_size", 2), guard_mode="project",
        guard_start_steps=0, guard_every=1, guard_replay_anchors=4, guard_add_anchors=4,
        guard_chunk_size=settings.get("guard_chunk_size", 2), guard_backtracks=3)
    model, batch = make_fixture(local, s, device)
    batch["loss_config"] = resolve_loss_config("ce", "eager")
    batch["hidden_states"].add_(rank * .125)
    wrapped = DistributedDataParallel(model)
    check_agreement(model, phase="ddp_initialization", step=0, local=local)
    optimizers = [torch.optim.AdamW([p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("prefix_head.")], lr=1e-4),
                  torch.optim.AdamW(model.prefix_head.parameters(), lr=3e-4)]

    class Trainer:
        def setup_optimizer(self):
            pass
        def run_training(self):
            pass
        def val_epoch(self, epoch):
            pass
    stub = SimpleNamespace(Trainer=Trainer, build_optimizers=None)
    install_hooks(SimpleNamespace(build_draft_model=None, create_train_val_loaders=None), stub, s)
    trainer = Trainer()
    trainer.model, trainer.optimizers, trainer.rank = wrapped, optimizers, rank
    active = 0
    for step in range(1, 5):
        model.joint_step = trainer.global_step = step
        wrapped.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            _, loss, _ = wrapped(**batch)
        loss.backward()
        check_agreement(model, phase="after_backward", step=step, local=local, gradients=True)
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.)
        check_agreement(model, phase="after_clipping", step=step, local=local, gradients=True)
        f = model.selector_guard.replay.data
        if rank == 1 and step % 2:
            model.selector_guard.replay.data = None
        else:
            with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                chosen = bf16_replay_copy(model.prefix_head).greedy(f["hidden"], f["candidates"], f["unary"], f["anchors"], embedding_weight=model.embed_tokens.weight)
            # Synthetic labels deliberately exercise a nonempty guard. They
            # are confined to this smoke fixture, never the real training run.
            f["target"], f["reference"] = chosen.clone(), chosen.clone()
            f["valid"].fill_(True)
        trainer._optimizers_step()
        active += int(model.joint_metrics["guard_protected_positions_sum"].item() > 0)
        check_agreement(model, phase="after_optimizer_and_guard", step=step, local=local)
        if step == 2:
            # Validation used to perform the same unsafe live dtype swap.
            # Exercise actual disk export/replay, then TWO more DDP backwards.
            root = Path(directory) / "export_checkpoints"
            if rank == 0:
                SingleGPUCheckpointer(root).save_checkpoint(model, optimizers, 0)
            dist.barrier()
            with checkpoint_parameters(model, root / "0") as (replay_model, _):
                with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    replay_model(**batch)
            del replay_model
            check_agreement(model, phase="after_saved_checkpoint_validation", step=step, local=local)
    if active != 4:
        raise RuntimeError("Distributed guard was not exercised at every smoke step")
    report = dict(rank=rank, world_size=world_size, device=str(device), optimizer_steps=4,
                  nonempty_guard_steps=active, empty_replay_rank_exercised=world_size > 1,
                  bit_exact_rank_agreement=True, gradients_checked_every_step=True,
                  backward_after_saved_validation=True, diagnostic_only=True, preflight_version=2)
    report.update(backbone_checkpointing=s.get("backbone_checkpointing", True),
                  selector_checkpointing=s.get("selector_checkpointing", True),
                  anchor_chunk_size=s["anchor_chunk_size"], guard_chunk_size=s["guard_chunk_size"])
    (local / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if rank == 0:
        print("JOINT_DISTRIBUTED_SMOKE " + json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=300))
    try:
        exercise(dist.get_rank(), dist.get_world_size(), args.output,
                 f"npu:{local_rank}", json.loads(args.settings.read_text()))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
