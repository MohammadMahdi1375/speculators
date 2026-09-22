"""Stored ReTrace updates with packed metric sums and bounded rank diagnostics.

The update, resume and checkpoint ordering follows the pinned stock Trainer.
Only this adapter uses the loop: shared DFlash/DSpark training is unchanged.
"""

import faulthandler
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from speculators.train.recovery import BatchRecoveryCoordinator
from speculators.train.trainer import MIN_STEP_PCT, _should_sync_recovery, _StepTimer
from speculators.train.utils import normalize_counted_metrics

metric_logger = logging.getLogger("speculators.metrics")
METRIC_KEYS = tuple(
    f"{prefix}_{suffix}"
    for prefix in (
        "loss",
        "clean_accuracy",
        "accepted_per_source",
        "conditioned_pair_fraction",
        "unaligned_pair_fraction",
        "conditioned_positions",
        "source_pairs",
        "clean_labels",
        "beta",
        "error_records",
    )
    for suffix in ("sum", "total")
)


def sum_training_metrics(metrics, device, distributed):
    """One FP32 all-reduce, fixed ordering, with a collective schema-error flag.

    Values are detached copies; the logging reduction cannot mutate the loss.
    SUM/total normalization remains the stock Trainer's normalization. The
    error flag ensures a local schema error is reported on every participant.
    """
    valid = set(metrics) == set(METRIC_KEYS) and all(
        isinstance(value, torch.Tensor) and value.numel() == 1
        for value in metrics.values()
    )
    if valid:
        packed = torch.stack(
            [
                metrics[key].detach().to(device=device, dtype=torch.float32).reshape(())
                for key in METRIC_KEYS
            ]
            + [torch.zeros((), device=device)]
        )
    else:
        packed = torch.zeros(len(METRIC_KEYS) + 1, device=device)
        packed[-1] = 1
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.cpu().tolist()  # Complete the collective before the next step.
    if values[-1]:
        raise RuntimeError(
            "Stored ReTrace metric schema mismatch on at least one rank; "
            f"local keys={sorted(metrics)}, local schema valid={valid}"
        )
    return dict(zip(METRIC_KEYS, values[:-1], strict=True))


def check_gradients(model):
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(
            "Stored ReTrace has unused trainable parameters after backward: "
            + ", ".join(missing)
        )


class RankTrace:
    """First few local updates only; no network operations or extra NPU syncs."""

    def __init__(self, trainer, epoch):
        self.trainer, self.epoch = trainer, epoch
        self.limit = max(0, int(os.environ.get("RETRACE_STORED_TRACE_STEPS", "4")))
        self.enabled = False
        self.handle = self.stacks = None
        if self.limit:
            root = Path(trainer.config.save_path).parent / "rank_diagnostics"
            root.mkdir(parents=True, exist_ok=True)
            stem = (
                f"rank_{trainer.rank}_epoch_{epoch}_pid_{os.getpid()}_{time.time_ns()}"
            )
            self.handle = (root / f"{stem}.jsonl").open("w", buffering=1)
            self.stacks = (root / f"{stem}.stacks.log").open("w", buffering=1)

    def begin(self, index):
        self.enabled = 0 <= index < self.limit
        if self.enabled:
            faulthandler.dump_traceback_later(60, repeat=True, file=self.stacks)
            self.mark("fetch_begin")

    def mark(self, stage, **details):
        if self.enabled:
            record = {
                "rank": self.trainer.rank,
                "local_rank": self.trainer.local_rank,
                "epoch": self.epoch,
                "global_step": self.trainer.global_step,
                "stage": stage,
                "time": time.time(),
                **details,
            }
            self.handle.write(json.dumps(record) + "\n")
            print("ReTrace rank progress: " + json.dumps(record), flush=True)

    def end(self):
        if self.enabled:
            faulthandler.cancel_dump_traceback_later()
        self.enabled = False

    def close(self):
        self.end()
        for handle in (self.handle, self.stacks):
            if handle is not None:
                handle.close()


def train_stored_epoch(trainer, epoch):
    """Keep optimizer/scheduler/save semantics; isolate metric communication."""
    trainer.model.train()
    if hasattr(trainer.train_loader.batch_sampler, "set_epoch"):
        trainer.train_loader.batch_sampler.set_epoch(epoch)
    num_steps = len(trainer.train_loader)
    skip_steps = trainer._prepare_resume_skip(epoch)
    remaining_steps = len(trainer.train_loader)
    step_interval = (
        max(1, round(num_steps * trainer.config.checkpoint_freq))
        if trainer.config.checkpoint_freq < 1
        else None
    )
    timer = _StepTimer()
    recovery = BatchRecoveryCoordinator("training")
    trace = RankTrace(trainer, epoch)
    try:
        # Trace before fetching too: a stalled Arrow/vLLM request is distinct
        # from a worker stuck in backward or a logging collective.
        trace.begin(0)
        t_before_fetch = time.perf_counter()
        for local_step_rel, batch in enumerate(trainer.train_loader, 1):
            local_step = local_step_rel + skip_steps
            timer.reset(trainer.global_step % trainer.config.log_freq == 0)
            timer.mark_value("start", t_before_fetch)
            will_stop = (
                trainer.config.max_steps is not None
                and trainer.global_step + 1 >= trainer.config.max_steps
            )
            trace.mark("fetch_done")
            recovery.consume(
                batch,
                synchronize=_should_sync_recovery(
                    local_step_rel,
                    remaining_steps,
                    will_stop=will_stop,
                ),
            )
            gpu_batch = {
                key: value.to(trainer.local_rank, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(
                trainer.device_type, dtype=trainer.config.hidden_states_dtype
            ):
                timer.mark("fetch")
                trace.mark("forward_begin")
                _, loss, metrics = trainer.model(
                    **gpu_batch, **(trainer.config.train_call_kwargs or {})
                )
            timer.mark("fwd")
            trace.mark("forward_done")
            trainer._optimizers_zero_grad()
            trace.mark("backward_begin")
            loss.backward()
            check_gradients(trainer.model)
            trace.mark("backward_returned", missing_gradients=0)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
            metrics["error_records_sum"] = torch.tensor(
                batch["error_records"], dtype=torch.int32, device=loss.device
            )
            metrics["error_records_total"] = torch.tensor(
                1.0 if trainer.rank == 0 else 0.0, device=loss.device
            )
            timer.mark("bwd")
            trace.mark("backward_synced")
            trainer._optimizers_step()
            current_lrs = {
                type(opt).__name__: opt.param_groups[0]["lr"]
                for opt in trainer.optimizers
            }
            trainer._schedulers_step()
            timer.mark("opt")
            trace.mark("optimizer_done")
            if timer.enabled:
                num_tokens = int((gpu_batch["document_ids"] != -1).sum().item())
                profile = timer.profile(num_tokens)
                trace.mark("metrics_all_reduce_begin")
                metrics = sum_training_metrics(
                    metrics, loss.device, trainer.is_distributed
                )
                trace.mark("metrics_all_reduce_done")
                world_size = dist.get_world_size() if trainer.is_distributed else 1
                metrics = normalize_counted_metrics(metrics, world_size)
                lr_info = (
                    current_lrs
                    if len(current_lrs) > 1
                    else next(iter(current_lrs.values()))
                )
                metric_logger.info(
                    {
                        "train": metrics,
                        "profile": profile,
                        "epoch": epoch,
                        "lr": lr_info,
                        "global_step": trainer.global_step,
                    },
                    extra={"step": trainer.global_step},
                )
            trace.mark("update_done")
            trace.end()
            trainer.global_step += 1
            if (
                trainer.config.max_steps is not None
                and trainer.global_step >= trainer.config.max_steps
            ):
                break
            if (
                step_interval is not None
                and not trainer.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * MIN_STEP_PCT
            ):
                trainer.maybe_save_checkpoint(epoch, local_step=local_step)
            trace.begin(local_step_rel)
            t_before_fetch = timer.now() or time.perf_counter()
    except BaseException as error:
        trace.mark("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        trace.close()
