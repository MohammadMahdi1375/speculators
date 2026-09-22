"""Online ReTrace training from real speculative trajectories.

Supports a local frozen HF target or the existing vLLM file connector, and
gradient averaging across torchrun workers. This is an explicit verification-
path soft-CE recipe; exact equivalence to the paper's training is not claimed.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from .data import read_prompts
from .rollout import generate, rollout, soft_ce_loss, target_greedy
from .runtime import add_model_args, device_for, setup


def average_gradients(parameters, count, device, distributed):
    total = torch.tensor(float(count), device=device)
    if distributed:
        dist.all_reduce(total)
    if total.item() == 0:
        return 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        if distributed:
            dist.all_reduce(gradient)
        gradient.div_(total)
        parameter.grad = gradient
    return int(total.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--data", "--data-path", required=True)
    parser.add_argument("--output", "--save-path", required=True)
    parser.add_argument("--max-prompts", type=int, default=40000)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--grad-accum-rounds", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-greedy", action="store_true")
    parser.add_argument(
        "--disable-conditioning",
        action="store_true",
        help="DFlash backbone ablation with identical initialization",
    )
    args = parser.parse_args()
    for field in (
        "max_steps",
        "epochs",
        "grad_accum_rounds",
        "prompt_length",
        "response_length",
        "save_every",
        "num_layers",
        "gamma",
        "lr",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive")
    if (
        not 0 <= args.lr_warmup_ratio < 1
        or args.block_size < 2
        or args.retrace_warmup_steps < 0
    ):
        parser.error("Invalid warmup or block size")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not args.vllm_endpoint:
        parser.error(
            "Multiple workers require --vllm-endpoint; local HF target is a one-device reference"
        )
    device = device_for(
        f"npu:{local_rank}" if args.device.startswith("npu") else args.device
    )
    if world > 1:
        dist.init_process_group("hccl" if device.type == "npu" else "gloo")
    output = Path(args.output)
    torch.manual_seed(args.seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    prompts, provenance = read_prompts(
        args.data,
        len(tokenizer),
        args.max_prompts,
        args.prompt_length,
        "train",
        args.seed,
    )
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "train_command.txt").exists():
            raise ValueError(
                "Use a new output directory; --draft initializes weights without resuming optimizer/trajectory state"
            )
        repo = Path(__file__).resolve().parents[4]
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
        ).stdout.strip()
        versions = {}
        for name in (
            "torch",
            "torch-npu",
            "transformers",
            "speculators",
            "vllm",
            "vllm-ascend",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        (output / "train_command.txt").write_text(
            json.dumps(
                {
                    "argv": sys.argv,
                    "args": vars(args),
                    "commit": sha,
                    "versions": versions,
                    "world_size": world,
                },
                indent=2,
            )
        )
        (output / "data_manifest.json").write_text(json.dumps(provenance, indent=2))
        source_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        }
        (output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
        patch = subprocess.run(
            ["git", "-C", str(repo), "diff", "HEAD"], text=True, capture_output=True
        ).stdout
        (output / "speculators.patch").write_text(patch)
    if world > 1:
        dist.barrier()
    print(f"Rank {rank}: initializing independent ReTrace model", flush=True)
    draft, target, eos = setup(args, device)
    draft.config.retrace_enabled = not args.disable_conditioning
    if args.check_greedy:
        draft.eval()
        budget = min(32, args.response_length)
        expected = target_greedy(target, draft, prompts[0], budget, eos)
        actual, _ = generate(target, draft, prompts[0], budget, eos)
        if actual != expected:
            raise RuntimeError("Initial greedy parity failed")
        print(f"Rank {rank}: greedy parity passed", flush=True)
    draft.train()
    parameters = [
        parameter for parameter in draft.parameters() if parameter.requires_grad
    ]
    print(
        json.dumps(
            {
                "rank": rank,
                "algorithm": "retrace",
                "initialization": "weights" if args.draft else "scratch",
                "conditioning": draft.config.retrace_enabled,
                "proposals_per_round": draft.block_size - 1,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "target_frozen": True,
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay, foreach=False
    )
    step = 0
    started = time.perf_counter()

    def beta():
        warmup = draft.config.retrace_warmup_steps
        return draft.config.retrace_beta_max * (min(step / warmup, 1) if warmup else 1)

    progress = {"prompts_started": 0, "prompts_completed": 0}

    def rounds():
        for epoch in range(args.epochs):
            indices = list(range(len(prompts)))
            random.Random(args.seed + epoch).shuffle(indices)
            for index in indices[rank::world]:
                progress["prompts_started"] += 1
                for event in rollout(
                    target,
                    draft,
                    prompts[index],
                    args.response_length,
                    eos,
                    beta,
                    train=True,
                ):
                    yield event, epoch
                progress["prompts_completed"] += 1

    iterator = rounds()
    exhausted = False
    last_saved = -1

    def save():
        nonlocal last_saved
        if rank == 0:
            dest = output / f"step_{step:06d}"
            dest.mkdir(exist_ok=False)
            draft.save_pretrained(dest)
            torch.save(optimizer.state_dict(), dest / "optimizer.pt")
            (dest / "training_state.json").write_text(
                json.dumps(
                    {"step": step, "args": vars(args), "world_size": world}, indent=2
                )
            )
            (output / "latest.txt").write_text(str(dest.resolve()) + "\n")
            print(f"Checkpoint: {dest}", flush=True)
        if world > 1:
            dist.barrier()
        last_saved = step

    log = (output / "metrics.jsonl").open("a") if rank == 0 else None
    try:
        while step < args.max_steps:
            # Sums are aggregated across rounds AND ranks, unlike the old
            # last-round-only logger. Exhausted ranks contribute zero gradients.
            beta_used = beta()
            sums = torch.zeros(7, device=device, dtype=torch.float32)
            for _ in range(args.grad_accum_rounds):
                if exhausted:
                    break
                try:
                    event, epoch = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                loss, tv = soft_ce_loss(event, args.gamma)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite ReTrace loss")
                loss.backward()
                sums += sums.new_tensor(
                    [
                        float(loss.detach()),
                        tv,
                        event.accepted,
                        event.proposed,
                        event.committed,
                        event.conditioned,
                        1,
                    ]
                )
                del loss, event
            total = average_gradients(parameters, int(sums[-1]), device, world > 1)
            if total == 0:
                break
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            warmup = max(1, int(args.lr_warmup_ratio * args.max_steps))
            scale = min((step + 1) / warmup, 1.0)
            if step >= warmup:
                scale = 0.5 * (
                    1
                    + math.cos(
                        math.pi * (step - warmup) / max(1, args.max_steps - warmup)
                    )
                )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if world > 1:
                dist.all_reduce(sums)
            coverage = torch.tensor(
                list(progress.values()), device=device, dtype=torch.long
            )
            if world > 1:
                dist.all_reduce(coverage)
            if rank == 0:
                values = sums.tolist()
                row = {
                    "step": step,
                    "loss": values[0] / total,
                    "tv": values[1] / total,
                    "accepted_per_round": values[2] / total,
                    "acceptance_rate": values[2] / max(1, values[3]),
                    "committed_per_round": values[4] / total,
                    "conditioned_positions_per_round": values[5] / total,
                    "rounds": total,
                    "beta": beta_used,
                    "prompts_started": int(coverage[0]),
                    "prompts_completed": int(coverage[1]),
                    "lr": args.lr * scale,
                    "grad_norm": float(norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step % args.save_every == 0:
                save()
        if step == 0:
            raise RuntimeError(
                "No trainable rounds; inspect response length and prompt/EOS boundaries"
            )
        if last_saved != step:
            save()
    finally:
        iterator.close()
        if log:
            log.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
