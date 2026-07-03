import logging
import warnings
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import (
    BaseCheckpointer,
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.utils import apply_fully_sharded

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    is_distributed: bool = False
    local_rank: int = 0
    train_call_kwargs: dict = {}
    val_call_kwargs: dict = {}
    scheduler_type: Literal["linear", "cosine", "none"] = "linear"
    scheduler_warmup_steps: int | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5
    checkpoint_freq: int = 1
    save_steps: int = 0
    save_best: bool = False
    hidden_states_dtype: torch.dtype = torch.bfloat16
    log_freq: int = 1


class Trainer:
    def __init__(
        self,
        model: SpeculatorModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.config = config
        self.local_rank = config.local_rank
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_distributed = config.is_distributed
        self.resume_from_checkpoint = config.resume_from_checkpoint
        checkpointer_class = (
            DistributedCheckpointer if self.is_distributed else SingleGPUCheckpointer
        )
        self.checkpointer: BaseCheckpointer = checkpointer_class(self.config.save_path)

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()
        self.tp_gather = self._detect_tp_gather()
        if self.tp_gather:
            self.train_loader.dataset.defer_generation = True
            if self.val_loader is not None:
                self.val_loader.dataset.defer_generation = True
        if self.local_rank == 0:
            root_logger.info(
                f"[trainer] TP gather/scatter = {'ON' if self.tp_gather else 'OFF'}"
            )

    def setup_trainer(self):
        if self.checkpointer.previous_epoch != -1:
            root_logger.info(f"Found checkpoint at {self.checkpointer.prev_path}.")
            self.current_epoch = self.checkpointer.previous_epoch + 1
            if self.resume_from_checkpoint:
                root_logger.info(f"Resuming training on {self.current_epoch} epoch.")
            else:
                root_logger.warning(
                    "`resume_from_checkpoint` is False, starting "
                    "training from scratch. This will overwrite the "
                    f"existing checkpoints in {self.checkpointer.path}."
                )
                self.current_epoch = 0
        else:
            root_logger.info("No previous checkpoint found. Starting from scratch.")
            self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            saved = self.checkpointer.load_best_val_loss()
            if saved is not None:
                self.best_val_loss = saved
                root_logger.info(
                    f"Restored best_val_loss={self.best_val_loss:.6f} from checkpoint"
                )

    def setup_model(self):
        # Verify model is compatible with training infrastructure
        SpeculatorModel.verify_training_compatible(self.model)

        self.model.to(self.config.hidden_states_dtype)  # type: ignore[arg-type]
        load_checkpoint = (
            self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1
        )

        if not self.is_distributed:
            # Single device case
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if load_checkpoint:
                self.checkpointer.load_model_state_dict(self.model)
            return

        # Distributed case
        #
        # Fresh scratch training: avoid the rank-0 full-state broadcast.
        # In the 2-node co-located DSV4 run, this Gloo object-broadcast
        # path can fail as 'Connection closed by peer'. All ranks are
        # re-seeded before draft model initialization in scripts/train.py.
        apply_fully_sharded(self.model)

        if load_checkpoint:
            self.checkpointer.load_model_state_dict(self.model)

        dist.barrier()

    def setup_optimizer(self):
        # Setup optimizer
        self.opt = torch.optim.AdamW(self.model.named_parameters(), lr=self.config.lr)
        last_epoch = -1
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.opt)
            last_epoch = self.checkpointer.previous_epoch

        # Setup scheduler
        if self.config.scheduler_type == "none":
            self.scheduler = None
            return

        # Compute defaults if None
        scheduler_warmup_steps = (
            self.config.scheduler_warmup_steps
            or (self.config.num_epochs * len(self.train_loader)) // 100
        )
        scheduler_total_steps = self.config.scheduler_total_steps or (
            self.config.num_epochs * len(self.train_loader)
        )

        if self.config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                last_epoch=last_epoch,
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                num_cycles=self.config.scheduler_num_cosine_cycles,
                last_epoch=last_epoch,
            )

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_scheduler_state_dict(self.scheduler)

    def train_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        train_loader = self.train_loader
        if self.local_rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        for batch in train_loader:
            if self.tp_gather:
                batch = self._tp_gather_generate_scatter(batch)
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            _draft_tokens, loss, metrics = self.model(
                **gpu_batch, **self.config.train_call_kwargs
            )

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            current_lr = self.opt.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Save a checkpoint every N optimizer steps (0 disables).
            _save_steps = getattr(self.config, "save_steps", 0)
            if _save_steps and self.global_step > 0 and self.global_step % _save_steps == 0:
                if (not self.is_distributed) or self.local_rank == 0:
                    root_logger.info(
                        f"Saving step checkpoint at global_step={self.global_step}"
                    )
                self.checkpointer.save_checkpoint(self.model, self.opt, self.global_step)
                if self.scheduler is not None:
                    self.checkpointer.save_scheduler_state_dict(
                        self.scheduler, self.global_step
                    )

            if self.global_step % self.config.log_freq == 0:
                if self.is_distributed:
                    for v in metrics.values():
                        dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)

                metrics = {k: v.item() for k, v in metrics.items()}
                metric_logger.info(
                    {
                        "train": metrics,
                        "epoch": epoch,
                        "lr": current_lr,
                        "global_step": self.global_step,
                    },
                    extra={"step": self.global_step},
                )
            self.global_step += 1

    def _detect_tp_gather(self) -> bool:
        import os as _os
        forced = _os.environ.get("DFLASH_TP_GATHER")
        if forced is not None:
            return forced not in ("", "0", "false", "False", "no")
        try:
            engine = getattr(self.train_loader.dataset, "engine", None)
            if engine is None:
                return False
            tp = engine.llm_engine.vllm_config.parallel_config.tensor_parallel_size
            return int(tp) > 1
        except Exception:
            return False

    def _tp_gather_generate_scatter(self, batch):
        import os as _os
        import time as _time
        from safetensors.torch import load_file
        from vllm import SamplingParams

        dataset = self.train_loader.dataset
        engine = dataset.engine
        seed = getattr(dataset, "seed", 42)
        hsd = self.config.hidden_states_dtype
        transform = getattr(dataset, "transform", None)
        collate = self.train_loader.collate_fn

        lengths = [int(x) for x in batch["lengths"].tolist()]
        ids_flat = batch["input_ids"].reshape(-1)
        lm_flat = batch["loss_mask"].reshape(-1)

        local_tokens, local_lossmask = [], []
        off = 0
        for L in lengths:
            local_tokens.append([int(t) for t in ids_flat[off:off + L].tolist()])
            local_lossmask.append(lm_flat[off:off + L].clone())
            off += L

        world = dist.get_world_size()
        gathered = [None] * world
        dist.all_gather_object(gathered, local_tokens)

        counts = [len(gathered[r]) for r in range(world)]
        global_tokens = [t for r in range(world) for t in gathered[r]]

        my_rank = dist.get_rank()
        my_offset = sum(counts[:my_rank])
        my_count = counts[my_rank]

        prompts = [{"prompt_token_ids": t} for t in global_tokens]
        sp = SamplingParams(max_tokens=1, temperature=0.0, seed=seed)

        outs = engine.generate(prompts, sp, use_tqdm=False) if prompts else []

        # Important:
        # Every rank calls engine.generate() with the full global prompt list.
        # On each node, that can create local hidden-state files for all prompts.
        # We must delete all paths returned by this local generate() call, not only
        # this rank's my_outs slice, otherwise /dev/shm/hidden_states grows forever.
        all_out_paths = []
        for o in outs:
            kvp = getattr(o, "kv_transfer_params", None)
            path = kvp.get("hidden_states_path") if isinstance(kvp, dict) else None
            if path:
                all_out_paths.append(path)

        my_outs = outs[my_offset:my_offset + my_count]

        samples = []
        for j, o in enumerate(my_outs):
            kvp = getattr(o, "kv_transfer_params", None)
            path = kvp.get("hidden_states_path") if isinstance(kvp, dict) else None
            if not path:
                raise RuntimeError(f"Missing hidden_states_path in kv_transfer_params: {kvp}")

            # The hidden-state connector writes to /dev/shm. Wait briefly for the
            # file to become visible before loading.
            for _attempt in range(200):
                if _os.path.exists(path):
                    break
                _time.sleep(0.05)
            else:
                parent = _os.path.dirname(path)
                try:
                    existing = sorted(_os.listdir(parent))[:30]
                except Exception as exc:
                    existing = [f"<could not list {parent}: {exc}>"]
                raise FileNotFoundError(
                    f"Hidden-state file was not found before load: {path}. "
                    f"First files in directory: {existing}"
                )

            loaded = load_file(path)
            hs = loaded["hidden_states"]
            seq = int(loaded["token_ids"].shape[0])

            sample = {
                "hidden_states": hs.flatten(1).to(hsd),
                "input_ids": loaded["token_ids"],
                "verifier_last_hidden_states": hs[:, -1].to(hsd),
                "loss_mask": local_lossmask[j],
                "lengths": torch.tensor([seq], dtype=torch.long),
                "position_ids": torch.arange(seq, dtype=torch.long),
            }

            if transform:
                sample = transform(sample)

            samples.append(sample)

        # All ranks must finish loading their own slice before any rank deletes
        # generated hidden-state files. This avoids the previous FileNotFoundError.
        if dist.is_initialized():
            dist.barrier()

        # Delete all files created by this rank's local engine.generate() call.
        # Multiple ranks on the same node may try to delete the same files; that is
        # fine because FileNotFoundError is ignored.
        if getattr(self.config, "on_generate", "delete") == "delete":
            deleted = 0
            for path in all_out_paths:
                try:
                    if path and _os.path.exists(path):
                        _os.remove(path)
                        deleted += 1
                except FileNotFoundError:
                    pass

            if _os.environ.get("DFLASH_HS_DEBUG", "0") == "1":
                print(
                    f"[DFLASH_HS_CLEANUP] rank={my_rank} "
                    f"my_count={my_count} outs={len(outs)} "
                    f"all_out_paths={len(all_out_paths)} deleted={deleted}",
                    flush=True,
                )

        if dist.is_initialized():
            dist.barrier()

        return collate(samples)

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.local_rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        val_metrics: dict[str, float] = {}
        num_batches = len(val_loader)
        for batch in val_loader:
            if self.tp_gather:
                batch = self._tp_gather_generate_scatter(batch)
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            _draft_tokens, _loss, metrics = self.model(
                **gpu_batch, **self.config.val_call_kwargs
            )

            if self.is_distributed:
                for m in metrics.values():
                    dist.all_reduce(m, op=dist.ReduceOp.AVG)

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        val_metrics = {f"{k}_epoch": v / num_batches for k, v in val_metrics.items()}
        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )
        return val_metrics

    def maybe_save_checkpoint(self, epoch: int, val_metrics: dict | None):
        if (
            self.config.save_best
            and val_metrics is not None
            and "loss_epoch" in val_metrics
        ):
            if val_metrics["loss_epoch"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss_epoch"]
                root_logger.info(
                    f"Saving new best checkpoint at epoch {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )
                self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
                if self.scheduler is not None:
                    self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)
                self.checkpointer.save_val_metrics(epoch, val_metrics)
                self.checkpointer.update_best_symlink(epoch)
                root_logger.info(
                    f"Updated checkpoint_best -> {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )
                # Keep ONLY the best checkpoint folder + best_checkpoint symlink
                self.checkpointer.cleanup_keep_only_best(best_epoch=epoch)

        elif epoch == 0 or (epoch + 1) % self.config.checkpoint_freq == 0:
            root_logger.info(
                f"Saving checkpoint to {self.checkpointer.path / str(epoch)}"
            )
            self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
            if self.scheduler is not None:
                self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)
            if val_metrics is not None:
                self.checkpointer.save_val_metrics(epoch, val_metrics)
            root_logger.info(
                f"Checkpoint saved to {self.checkpointer.path / str(epoch)}"
            )
            if (
                val_metrics is not None
                and "loss_epoch" in val_metrics
                and val_metrics["loss_epoch"] < self.best_val_loss
            ):
                self.best_val_loss = val_metrics["loss_epoch"]
                root_logger.info(
                    f"Updating new best checkpoint symlink at epoch {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )
                self.checkpointer.update_best_symlink(epoch)
                root_logger.info(
                    f"Updated checkpoint_best -> {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )

    def run_training(self):
        n_epochs = self.config.num_epochs
        for epoch in range(self.current_epoch, n_epochs):
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
            self.train_epoch(epoch)
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            val_metrics = None

            if self.val_loader is None:
                root_logger.warning("No val loader, skipping validation epoch")
            else:
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                val_metrics = self.val_epoch(epoch)
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_save_checkpoint(epoch, val_metrics)

            if self.is_distributed:
                dist.barrier()
