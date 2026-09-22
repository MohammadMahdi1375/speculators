"""Train stored-response ReTrace with the existing DFlash data loader/Trainer."""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from hs_connectors.transfer import FileTransfer

from speculators.train.data import ArrowDataset
from speculators.train.dataloader import _limit_worker_threads, _setup_dataloader
from speculators.train.distributed import (
    get_rank,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig

from .config import ReTraceSpeculatorConfig
from .stored_loop import train_stored_epoch
from .stored_target import BranchTarget
from .stored_training import ReTraceDraftModel


class StoredTrainer(Trainer):
    """Use stock updates and saves; add epoch wall time and global memory coverage."""

    def train_epoch(self, epoch):
        self.coverage = 0
        begin = time.perf_counter()
        start_step = self.global_step
        train_stored_epoch(self, epoch)
        elapsed = time.perf_counter() - begin
        count = torch.tensor(self.coverage, device=self.local_rank, dtype=torch.long)
        if self.is_distributed:
            dist.all_reduce(count)
        if int(count) == 0:
            raise RuntimeError(
                "No usable conditioned pairs in the entire epoch. This is not a "
                "successful ReTrace run. Inspect stored-response alignment before continuing."
            )
        if self.rank == 0:
            Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
            record = {
                "epoch": epoch + 1,
                "updates": self.global_step - start_step,
                "wall_seconds": elapsed,
                "seconds_per_update": elapsed / max(1, self.global_step - start_step),
                "conditioned_pairs_global": int(count),
                "first_epoch_includes_cold_feature_io": epoch == 0,
            }
            with (Path(self.config.save_path) / "epoch_timing.jsonl").open(
                "a"
            ) as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


class JSONMetrics(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def emit(self, record):
        if get_rank() == 0 and isinstance(record.msg, dict):
            with self.path.open("a") as handle:
                handle.write(json.dumps(record.msg, default=str) + "\n")


def load_model(checkpoint):
    config = ReTraceSpeculatorConfig.from_pretrained(checkpoint)
    config.transformer_layer_config._attn_implementation = "sdpa"
    config.training_objective = "stored_pair_kl"
    model = ReTraceDraftModel.from_pretrained(
        checkpoint, config=config, dtype=torch.float32, local_files_only=True
    )
    # Training weights/master parameters remain FP32. Only frozen shared
    # embedding/head/norm use BF16; autocast handles draft network operations.
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = json.loads(Path(args.run_config).read_text())
    output = Path(run["output"])
    random.seed(run["seed"])
    torch.manual_seed(run["seed"])
    _limit_worker_threads()
    maybe_setup_distributed()
    setup_root_logger()
    setup_metric_logger([], run_name="retrace_stored", output_dir=str(output / "logs"))
    logging.getLogger("speculators.metrics").addHandler(
        JSONMetrics(output / "metrics.jsonl")
    )
    model = load_model(run["initial_checkpoint"])
    teacher = BranchTarget(
        run["endpoint"],
        run["served_model"],
        len(model.target_layer_ids),
        model.hidden_size,
        concurrency=run["request_concurrency"],
        timeout=run["request_timeout"],
    )
    try:
        model.configure_stored(
            teacher,
            pairs_per_batch=run["pairs_per_batch"],
            blocks_per_forward=run["blocks_per_forward"],
            seed=run["seed"],
            rank=get_rank(),
        )
        dataset = ArrowDataset(
            max_len=run["tokens_per_worker"],
            datapath=run["stored_data"],
            transfer=FileTransfer(output / "clean_features"),
            vllm_endpoint=run["endpoint"],
            on_missing="generate",
            on_generate="cache" if run["cache_clean"] else "delete",
            train_ratio=1.0,
            split="train",
            transform=None,
            hidden_states_dtype=torch.bfloat16,
            model=run["served_model"],
            request_timeout=run["request_timeout"],
            max_retries=2,
            generation_validation_retries=1,
            max_consecutive_generation_failures=3,
        )
        dataset.data.set_format("torch")
        loader = _setup_dataloader(
            dataset,
            total_seq_len=run["tokens_per_worker"],
            hidden_size=model.hidden_size,
            num_workers=run["data_workers"],
            num_target_layers=len(model.target_layer_ids),
            prefetch_factor=2,
        )
        if len(loader) == 0:
            raise ValueError(
                "Not enough packed batches for all workers; use more records"
            )
        trainer = StoredTrainer(
            model,
            TrainerConfig(
                num_epochs=run["epochs"],
                save_path=str(output / "checkpoints"),
                lr=run["lr"],
                weight_decay=0.01,
                scheduler_type="cosine",
                scheduler_warmup_ratio=0.05,
                checkpoint_freq=1,
                hidden_states_dtype=torch.bfloat16,
                log_freq=1,
                fsdp_shard=False,
                gradient_checkpointing=False,
                resume_from_checkpoint=args.resume,
            ),
            loader,
        )
        model.step_provider = lambda: trainer.global_step

        def count_pairs(_module, _args, result):
            # CPU scalar extraction also occurs in the standard metric logger.
            trainer.coverage += int(result[2]["conditioned_pair_fraction_sum"])

        hook = model.register_forward_hook(count_pairs)
        if get_rank() == 0:
            print(
                f"Stored-response training: {len(dataset):,} records; "
                f"{len(loader)} packed updates/epoch; {run['epochs']} epochs; "
                f"{run['pairs_per_batch']} sampled source pairs/worker/update",
                flush=True,
            )
            print(
                "Target backend: vLLM. Complete responses come from Arrow. "
                "Checkpoint cadence: each completed epoch.",
                flush=True,
            )
        trainer.run_training()
        hook.remove()
    finally:
        teacher.close()
        maybe_destroy_distributed()


if __name__ == "__main__":
    main()
