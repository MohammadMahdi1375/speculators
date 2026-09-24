"""Fine-tune a pretrained DFlash + ReTrace drafter on complete prompt batches.

BF16 forwards, FP32 trainable parameters/AdamW states, detached FP16 memory,
clean block CE, and exact global prompt-batch accounting. Distributed workers
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

from .batched_training import collect_batched_trajectories
from .clean_training import backward_trajectory, collect_trajectory
from .execution import make_prompt_queue, static_prompt_groups, validate_prompt_coverage
from .performance import peak_memory_gib, synchronize, validate_resume_signature
from .resume import check_checkpoint, training_signature
from .pretrained import sha256
from .runtime import add_model_args, device_for, setup
from .utils import average_gradients


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
    root = Path(__file__).resolve().parents[5]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "versions": versions,
        "speculators_commit": commit,
        "source_sha256": {
            p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
        },
        "recipe_revision": "paper_recipe_20260924",
        "rollout_sampling": "greedy (author training temperature unspecified)",
        "anchor_sampling": "all actual verification-round boundaries (author sampler unspecified)",
        "distributed_strategy": "replicated FP32 master parameters; summed prompt gradients",
        "paper_implementation_choices": {
            "anchors": "live speculative round boundaries",
            "labels": "hard tokens from completed greedy target-verified continuation",
            "prompt_reduction": "mean of per-prompt CE; per-prompt valid-token denominator",
            "memory": "detached FP16 draft and target states; one round",
            "gamma": "4.0, inherited DFlash default; not specified in ReTrace appendix",
            "exact_author_training_code_available": False,
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
    parser.add_argument("--migrate-to-vllm", action="store_true")
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
    parser.add_argument("--blocks-per-forward", type=int, default=4)
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=1,
        help="Concurrent local prompt rollouts; >1 requires cached local-target execution",
    )
    parser.add_argument(
        "--performance-mode", choices=("reference", "cached"), default="reference"
    )
    parser.add_argument("--trace-storage", choices=("cpu", "device"), default="cpu")
    parser.add_argument("--profile-performance", action="store_true")
    parser.add_argument(
        "--work-distribution", choices=("static", "dynamic"), default="static"
    )
    parser.add_argument(
        "--allow-performance-change",
        action="store_true",
        help="Explicitly migrate resume execution settings; recipe/optimizer/pool changes remain forbidden",
    )
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
    if args.migrate_to_vllm and not (args.resume and args.vllm_endpoint):
        parser.error("--migrate-to-vllm requires a checkpoint and --vllm-endpoint")
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
        "rollout_batch_size",
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
    if args.rollout_batch_size > 1 and (
        args.vllm_endpoint or args.performance_mode != "cached"
    ):
        parser.error(
            "Batched rollouts require --performance-mode cached and the local target"
        )
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_num_threads(1)
    device_name = (
        f"{args.device.split(':')[0]}:{local_rank}"
        if args.device.startswith(("npu", "cuda"))
        else args.device
    )
    device = device_for(device_name)
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
    signature = training_signature(
        args, world, total_steps, "vllm" if args.vllm_endpoint else "local"
    )
    output = Path(args.output)
    resumed = None
    execution_changes = {}
    if args.resume:
        resumed = check_checkpoint(args.resume, world)
        execution_changes = validate_resume_signature(
            resumed["signature"], signature, args.allow_performance_change, args.migrate_to_vllm
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
            "resume_execution_changes": execution_changes,
            "resume_note": "Optimizer, FP32 masters, step and retained ranks' RNG restored. Migration can change worker assignment and target rounding; no bitwise-equivalence claim.",
            **provenance(),
        }
        destination = output / (
            "resume_command.json" if resumed else "train_command.json"
        )
        destination.write_text(json.dumps(run, indent=2) + "\n")
        root = Path(__file__).resolve().parents[5]
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
        draft.config.training_objective != "clean_block_ce"
        or not draft.config.pretrained_fingerprint
    ):
        raise ValueError(
            "This trainer requires a checkpoint created by retrace.pretrained"
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
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False)
        if args.dtype == "bfloat16"
        else nullcontext()
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "initialization": "pretrained_dflash",
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
                "performance_mode": args.performance_mode,
                "trace_storage": args.trace_storage,
                "blocks_per_forward": args.blocks_per_forward,
                "rollout_batch_size": args.rollout_batch_size,
                "attention_backend": args.attention_backend,
                "target_norm": args.target_norm,
                "work_distribution": args.work_distribution,
                "resume_execution_changes": execution_changes,
            }
        ),
        flush=True,
    )
    if args.work_distribution == "dynamic":
        if draft.config.transformer_layer_config.attention_dropout != 0:
            raise ValueError(
                "Dynamic scheduling requires the inspected dropout-free drafter"
            )
        queue = make_prompt_queue(output, rank, world, device)
    else:
        queue = None
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
            if args.profile_performance:
                synchronize(device)
            step_started = time.perf_counter()
            rollout_seconds, backward_seconds = 0.0, 0.0
            warmup = draft.config.retrace_warmup_steps
            beta = draft.config.retrace_beta_max * (
                min(step / warmup, 1) if warmup else 1
            )
            local = [0.0] * 11
            groups = (
                queue.groups(batch, step + 1, args.rollout_batch_size)
                if queue is not None
                else static_prompt_groups(batch, rank, world, args.rollout_batch_size)
            )
            claimed_positions, prompts_completed = [], 0
            for batch_positions, indices in groups:
                claimed_positions.extend(batch_positions)
                tick = time.perf_counter()
                draft.eval()
                with amp():
                    if args.rollout_batch_size > 1:
                        collected = collect_batched_trajectories(
                            target,
                            draft,
                            [prompts[index] for index in indices],
                            args.response_length,
                            eos,
                            beta,
                            trace_storage=args.trace_storage,
                        )
                        traces = collected.traces
                        local[9] += collected.target_forwards
                        local[10] += collected.draft_forwards
                        del collected
                    else:
                        trace = collect_trajectory(
                            target,
                            draft,
                            prompts[indices[0]],
                            args.response_length,
                            eos,
                            beta,
                            performance_mode=args.performance_mode,
                            trace_storage=args.trace_storage,
                        )
                        traces = [trace]
                        local[9] += trace.target_calls
                        local[10] += len(trace.positions)
                if args.profile_performance:
                    synchronize(device)
                replay_started = time.perf_counter()
                rollout_seconds += replay_started - tick
                draft.train()
                for offset, trace in enumerate(traces):
                    with amp():
                        values = backward_trajectory(
                            draft,
                            trace,
                            beta,
                            blocks_per_forward=args.blocks_per_forward,
                            gamma=args.gamma,
                            optimized=args.performance_mode == "cached",
                        )
                    additions = [
                        values["loss"],
                        values["labels"],
                        values["rounds"],
                        trace.accepted,
                        trace.proposed,
                        trace.committed,
                        values["conditioned"],
                        trace.target_calls,
                        1,
                    ]
                    for index, value in enumerate(additions):
                        local[index] += value
                    number = prompts_completed + offset + 1
                    if number % args.heartbeat_prompts == 0:
                        print(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "update": step + 1,
                                    "prompt_in_local_batch": number,
                                    "global_batch_position": batch_positions[offset],
                                    "pool_index": indices[offset],
                                    "rounds": values["rounds"],
                                    "rollout_microbatch": len(traces),
                                    "microbatch_seconds_so_far": round(
                                        time.perf_counter() - tick, 2
                                    ),
                                }
                            ),
                            flush=True,
                        )
                if args.profile_performance:
                    synchronize(device)
                backward_seconds += time.perf_counter() - replay_started
                prompts_completed += len(indices)
                del trace, traces
            compute_finished = time.perf_counter()
            wait_seconds = 0.0
            if args.profile_performance:
                synchronize(device)
                if world > 1:
                    dist.barrier()
                synchronize(device)
                wait_seconds = time.perf_counter() - compute_finished
            sync_started = time.perf_counter()
            if queue is not None:
                validate_prompt_coverage(
                    claimed_positions, len(batch), device, world > 1
                )
            total = average_gradients(parameters, int(local[8]), device, world > 1)
            if args.profile_performance:
                synchronize(device)
            optimization_started = time.perf_counter()
            gradient_sync_seconds = optimization_started - sync_started
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
            if args.profile_performance:
                synchronize(device)
            optimization_seconds = time.perf_counter() - optimization_started
            step += 1
            values = torch.tensor(
                local,
                device=device,
                dtype=torch.float64 if device.type == "cpu" else torch.float32,
            )
            if world > 1:
                dist.all_reduce(values)
            performance = None
            if args.profile_performance:
                synchronize(device)
                performance = torch.tensor(
                    [
                        rollout_seconds,
                        backward_seconds,
                        optimization_seconds,
                        time.perf_counter() - step_started,
                        peak_memory_gib(device),
                        wait_seconds,
                        gradient_sync_seconds,
                        compute_finished - step_started,
                        local[8],
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                if world > 1:
                    worker_performance = [
                        torch.empty_like(performance) for _ in range(world)
                    ]
                    dist.all_gather(worker_performance, performance)
                else:
                    worker_performance = [performance]
                worker_performance = torch.stack(worker_performance)
                performance = worker_performance[:, :8].amax(dim=0)
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
                    "target_forwards": int(values[9]),
                    "draft_forwards": int(values[10]),
                    "rollout_batch_size": args.rollout_batch_size,
                    "attention_backend": args.attention_backend,
                    "target_norm": args.target_norm,
                    "work_distribution": args.work_distribution,
                    "beta": beta,
                    "lr": rate,
                    "gradient_norm": float(norm),
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": elapsed
                    / (step - start_step)
                    * (total_steps - step),
                }
                if performance is not None:
                    row.update(
                        dict(
                            zip(
                                (
                                    "rollout_seconds_max",
                                    "backward_seconds_max",
                                    "optimizer_seconds_max",
                                    "step_seconds",
                                    "peak_memory_gib_max",
                                    "worker_wait_seconds_max",
                                    "gradient_sync_seconds_max",
                                    "compute_seconds_max",
                                ),
                                performance.tolist(),
                                strict=True,
                            )
                        )
                    )
                    row["performance_mode"] = args.performance_mode
                    row["blocks_per_forward"] = args.blocks_per_forward
                    row["rank_performance"] = [
                        dict(
                            rank=r,
                            rollout_seconds=p[0],
                            backward_seconds=p[1],
                            optimizer_seconds=p[2],
                            step_seconds=p[3],
                            peak_memory_gib=p[4],
                            worker_wait_seconds=p[5],
                            gradient_sync_seconds=p[6],
                            compute_seconds=p[7],
                            prompts=int(p[8]),
                        )
                        for r, p in enumerate(worker_performance.tolist())
                    ]
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
        if hasattr(target, "close"):
            target.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
