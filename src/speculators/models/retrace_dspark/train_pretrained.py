"""Fine-tune a pretrained DSpark + ReTrace drafter on complete prompt batches.

BF16 forwards, FP32 trainable parameters/AdamW states, detached FP16 memory,
clean block CE/TV and confidence loss, and exact global prompt-batch accounting. Distributed workers
replicate parameters and sum prompt gradients over HCCL/Gloo; this is not FSDP.
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
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from .clean_training import backward_trajectory, collect_trajectory
from .pretrained import sha256
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


def epoch_batches(size, epochs, batch_size, seed):
    for epoch in range(epochs):
        order = list(range(size))
        random.Random(seed + epoch).shuffle(order)
        for offset in range(0, size, batch_size):
            yield epoch, order[offset : offset + batch_size]


def learning_rate(step, total, peak, warmup_ratio):
    warmup = max(1, round(total * warmup_ratio))
    if step < warmup:
        return peak * (step + 1) / warmup
    return (
        peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    )


def provenance():
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
    root = Path(__file__).resolve().parents[4]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "versions": versions,
        "speculators_commit": commit,
        "source_sha256": {
            p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
        },
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "distributed_strategy": "replicated FP32 master parameters; summed prompt gradients",
        "paper_implementation_choices": {
            "anchors": "live speculative round boundaries",
            "labels": "hard tokens from completed greedy target-verified continuation",
            "prompt_reduction": "mean of per-prompt DSpark CE/TV/BCE; valid-token denominator",
            "memory": "detached FP16 draft and target states; one round",
            "gamma": "4.0, inherited DSpark default",
            "target_distributions": "clean completed continuation; not rejected-prefix logits",
            "teacher_forced_markov_inputs": "clean previous-token IDs at p+j",
            "confidence": "DSpark distribution overlap target, no gradient into target",
            "scope": "ReTrace extension to DSpark; not a paper Table 2 reproduction",
            "author_retrace_dspark_training_code_used": False,
        },
    }


def read_pool(path, vocabulary, max_length):
    prompts, keys = [], set()
    with Path(path).open() as handle:
        for line, text in enumerate(handle, 1):
            row = json.loads(text)
            ids = row["input_ids"]
            if (
                not isinstance(ids, list)
                or not 0 < len(ids) <= max_length
                or any(type(x) is not int or not 0 <= x < vocabulary for x in ids)
            ):
                raise ValueError(f"Invalid prepared prompt at line {line}")
            key = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            if key != row.get("sha256") or key in keys:
                raise ValueError("Prompt pool contains modified tokens or duplicates")
            prompts.append(ids)
            keys.add(key)
    if not prompts:
        raise ValueError("Empty prompt pool")
    return prompts


def rng_state(device):
    result = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if device.type == "npu":
        result["accelerator"] = torch.npu.get_rng_state(device).cpu()
    elif device.type == "cuda":
        result["accelerator"] = torch.cuda.get_rng_state(device).cpu()
    return result


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if device.type == "npu":
        torch.npu.set_rng_state(state["accelerator"], device)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"], device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--global-prompt-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="0 runs every configured epoch"
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Save and pause after this update without changing the LR schedule; resume with the same epoch settings",
    )
    parser.add_argument("--blocks-per-forward", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Optimizer-step save interval; 0 saves at the end of each epoch",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-conditioning", action="store_true")
    parser.add_argument("--heartbeat-prompts", type=int, default=1)
    args = parser.parse_args()
    if not args.draft and not args.resume:
        parser.error(
            "Pretrained initialization is mandatory: pass --draft from the importer"
        )
    for field in (
        "prompt_length",
        "response_length",
        "global_prompt_batch",
        "epochs",
        "blocks_per_forward",
        "gamma",
        "lr",
        "heartbeat_prompts",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive")
    if (
        min(args.max_steps, args.stop_after, args.weight_decay, args.save_every) < 0
        or not 0 <= args.lr_warmup_ratio < 1
    ):
        parser.error("Invalid max steps, save interval, weight decay or LR warmup")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_num_threads(1)
    device_name = (
        f"{args.device.split(':')[0]}:{local_rank}"
        if args.device.startswith(("npu", "cuda"))
        else args.device
    )
    device = device_for(device_name)
    if device.type == "npu":
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if len(visible) != world or torch.npu.device_count() < world:
            raise ValueError(
                "Visible NPU count differs from requested worker count; no fallback device mapping"
            )
        print(
            f"Rank {rank}: npu:{local_rank} maps to requested NPU {visible[local_rank]}",
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        backend = (
            "hccl"
            if device.type == "npu"
            else "nccl"
            if device.type == "cuda"
            else "gloo"
        )
        dist.init_process_group(backend, timeout=timedelta(hours=2))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    prompts = read_pool(args.pool, len(tokenizer), args.prompt_length)
    steps_per_epoch = math.ceil(len(prompts) / args.global_prompt_batch)
    save_interval = args.save_every or steps_per_epoch
    planned = steps_per_epoch * args.epochs
    total_steps = min(planned, args.max_steps) if args.max_steps else planned
    signature = {
        **{
            k: getattr(args, k)
            for k in (
                "target",
                "prompt_length",
                "response_length",
                "global_prompt_batch",
                "epochs",
                "lr",
                "lr_warmup_ratio",
                "weight_decay",
                "gamma",
                "seed",
                "disable_conditioning",
                "dtype",
            )
        },
        "pool_sha256": sha256(args.pool),
        "world_size": world,
        "total_steps": total_steps,
        "target_backend": "vllm" if args.vllm_endpoint else "local",
        "blocks_per_forward": args.blocks_per_forward,
    }
    output = Path(args.output)
    resumed = None
    if args.resume:
        resumed = json.loads((Path(args.resume) / "training_state.json").read_text())
        if resumed["signature"] != signature:
            raise ValueError(
                "Resume configuration/prompt pool/world size differs from the checkpoint"
            )
        if int(resumed["step"]) >= total_steps:
            print(
                "This checkpoint already completed all configured updates.", flush=True
            )
            if world > 1:
                dist.destroy_process_group()
            return
        pointer = output / "latest.txt"
        if (
            pointer.exists()
            and Path(pointer.read_text().strip()).resolve()
            != Path(args.resume).resolve()
        ):
            raise ValueError(
                "Resume the latest checkpoint, or use a new output directory for a branch"
            )
        if (output / "metrics.jsonl").exists():
            records = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if records and records[-1]["step"] > resumed["step"]:
                raise ValueError(
                    "Metrics extend past this checkpoint; use a new output directory with --resume and --pool"
                )
        args.draft = args.resume
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "train_command.json").exists() and not resumed:
            raise ValueError(
                "Use a new output directory or --resume an exact checkpoint"
            )
        run = {
            "argv": sys.argv,
            "args": vars(args),
            "signature": signature,
            **provenance(),
        }
        destination = output / (
            "resume_command.json" if resumed else "train_command.json"
        )
        destination.write_text(json.dumps(run, indent=2) + "\n")
        root = Path(__file__).resolve().parents[4]
        (output / "speculators.patch").write_text(
            subprocess.run(
                ["git", "-C", str(root), "diff", "HEAD"], capture_output=True, text=True
            ).stdout
        )
    if world > 1:
        dist.barrier()
    print(f"Rank {rank}: loading pretrained drafter and frozen target", flush=True)
    draft, target, eos = setup(args, device)
    if (
        draft.config.training_objective != "dspark_clean_block"
        or not draft.config.pretrained_fingerprint
    ):
        raise ValueError(
            "This trainer requires a checkpoint created by retrace_dspark.pretrained"
        )
    draft.config.retrace_enabled = not args.disable_conditioning
    parameters = [p for p in draft.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay, foreach=False
    )
    start_step = int(resumed["step"]) if resumed else 0
    if resumed:
        optimizer.load_state_dict(
            torch.load(
                Path(args.resume) / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        restore_rng(
            torch.load(
                Path(args.resume) / f"rng_rank_{rank}.pt",
                map_location="cpu",
                weights_only=True,
            ),
            device,
        )
    # Compare this identity separately because it is learned from the loaded model.
    if (
        resumed
        and resumed.get("pretrained_fingerprint") != draft.config.pretrained_fingerprint
    ):
        raise ValueError("Resume pretrained identity differs")
    amp = lambda: (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "initialization": "pretrained_dspark",
                "source": draft.config.pretrained_source,
                "aux_hidden_state_layer_ids": draft.target_layer_ids,
                "global_prompt_batch": args.global_prompt_batch,
                "prompts": len(prompts),
                "optimizer_updates": total_steps,
                "full_epochs": args.epochs,
                "checkpoint_schedule": "epoch" if args.save_every == 0 else "steps",
                "checkpoint_every_steps": save_interval,
                "block_size": draft.block_size,
                "trainable_parameters": sum(p.numel() for p in parameters),
            }
        ),
        flush=True,
    )
    latest_saved = -1

    def save(step):
        nonlocal latest_saved
        stage, final = output / f".step_{step:06d}.partial", output / f"step_{step:06d}"
        if rank == 0:
            if stage.exists() or final.exists():
                raise ValueError(f"Checkpoint destination already exists: {final}")
            stage.mkdir()
            state = {
                name: tensor.detach()
                .to(
                    device="cpu",
                    dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
                )
                .contiguous()
                for name, tensor in draft.state_dict().items()
                if name not in draft._keys_to_ignore_on_save
            }
            draft.save_pretrained(stage, state_dict=state)
            del state
            # Preserve FP32 master tensors too: BF16 export alone is not an exact resume.
            master = {
                name: p.detach().cpu()
                for name, p in draft.named_parameters()
                if p.requires_grad
            }
            torch.save(master, stage / "master_parameters.pt")
            torch.save(optimizer.state_dict(), stage / "optimizer.pt")
            (stage / "training_state.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "signature": signature,
                        "pretrained_fingerprint": draft.config.pretrained_fingerprint,
                    },
                    indent=2,
                )
                + "\n"
            )
        if world > 1:
            dist.barrier()
        torch.save(rng_state(device), stage / f"rng_rank_{rank}.pt")
        if world > 1:
            dist.barrier()
        if rank == 0:
            stage.replace(final)
            pointer = output / "latest.partial"
            pointer.write_text(str(final.absolute()) + "\n")
            pointer.replace(output / "latest.txt")
            print(f"Checkpoint: {final}", flush=True)
        if world > 1:
            dist.barrier()
        latest_saved = step

    if resumed:
        master = torch.load(
            Path(args.resume) / "master_parameters.pt",
            map_location="cpu",
            weights_only=True,
        )
        with torch.no_grad():
            for name, parameter in draft.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(master.pop(name))
        if master:
            raise ValueError("Unexpected FP32 master parameters in resumed checkpoint")
        del master
    log = (output / "metrics.jsonl").open("a", buffering=1) if rank == 0 else None
    step = start_step
    started = time.perf_counter()
    try:
        for batch_index, (epoch, batch) in enumerate(
            epoch_batches(
                len(prompts), args.epochs, args.global_prompt_batch, args.seed
            )
        ):
            if batch_index < start_step:
                continue
            if step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            warmup = draft.config.retrace_warmup_steps
            beta = draft.config.retrace_beta_max * (
                min(step / warmup, 1) if warmup else 1
            )
            local = [0.0] * 9
            for number, index in enumerate(batch[rank::world], 1):
                tick = time.perf_counter()
                draft.eval()
                with amp():
                    trace = collect_trajectory(
                        target, draft, prompts[index], args.response_length, eos, beta
                    )
                draft.train()
                with amp():
                    values = backward_trajectory(
                        draft,
                        trace,
                        beta,
                        blocks_per_forward=args.blocks_per_forward,
                        gamma=args.gamma,
                    )
                local = [
                    a + b
                    for a, b in zip(
                        local,
                        [
                            values["loss"],
                            values["labels"],
                            values["rounds"],
                            trace.accepted,
                            trace.proposed,
                            trace.committed,
                            values["conditioned"],
                            trace.target_calls,
                            1,
                        ],
                        strict=True,
                    )
                ]
                if number % args.heartbeat_prompts == 0:
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "update": step + 1,
                                "prompt_in_local_batch": number,
                                "rounds": values["rounds"],
                                "prompt_seconds": round(time.perf_counter() - tick, 2),
                            }
                        ),
                        flush=True,
                    )
                del trace
            total = average_gradients(parameters, int(local[-1]), device, world > 1)
            if total != len(batch):
                raise RuntimeError(
                    "Distributed prompt accounting differs from the global batch"
                )
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            rate = learning_rate(step, total_steps, args.lr, args.lr_warmup_ratio)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            step += 1
            values = torch.tensor(
                local,
                device=device,
                dtype=torch.float64 if device.type == "cpu" else torch.float32,
            )
            if world > 1:
                dist.all_reduce(values)
            if rank == 0:
                values = values.tolist()
                elapsed = time.perf_counter() - started
                row = {
                    "step": step,
                    "planned_steps": total_steps,
                    "epoch": epoch + 1,
                    "global_prompts": total,
                    "prompt_visits_completed": epoch * len(prompts)
                    + min(
                        (
                            batch_index
                            % math.ceil(len(prompts) / args.global_prompt_batch)
                            + 1
                        )
                        * args.global_prompt_batch,
                        len(prompts),
                    ),
                    "loss": values[0] / total,
                    "clean_labels": int(values[1]),
                    "rounds": int(values[2]),
                    "accepted_per_round": values[3] / max(1, values[2]),
                    "committed_per_round": values[5] / max(1, values[2]),
                    "conditioned_positions": int(values[6]),
                    "target_calls": int(values[7]),
                    "beta": beta,
                    "lr": rate,
                    "gradient_norm": float(norm),
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": elapsed
                    / (step - start_step)
                    * (total_steps - step),
                }
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % save_interval == 0:
                save(step)
            if args.stop_after and step >= args.stop_after:
                break
        if step != latest_saved:
            save(step)
        if rank == 0:
            (output / "completed.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "all_configured_epochs_completed": step == planned,
                        "early_step_limit": args.max_steps or None,
                        "paused_after": args.stop_after or None,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        if log:
            log.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
