"""Exact global prompt batches; packed features without sample truncation.

With 32 prompts and six trainers, rank assignments rotate between five and six
prompts. The model weights local prompt sums by world_size / global_batch so
DDP's mean reduction gives exactly the global mean, despite uneven assignments.
"""

import math

import torch
from torch.utils.data import DataLoader, Sampler

from speculators.train.data import ArrowDataset, CollateFn
from speculators.train.dataloader import _worker_init_fn
from speculators.train.distributed import get_dp_rank, get_dp_size


class PromptBatchSampler(Sampler):
    def __init__(self, size, global_batch=32, *, rank=0, replicas=1, seed=42):
        if size < global_batch or global_batch < replicas or size % global_batch:
            raise ValueError("Dataset must contain complete global prompt batches; batch >= ranks")
        if not 0 <= rank < replicas:
            raise ValueError("Invalid distributed rank")
        self.size, self.global_batch = size, global_batch
        self.rank, self.replicas, self.seed = rank, replicas, seed
        self.epoch = self.skip = 0

    def set_epoch(self, epoch):
        self.epoch, self.skip = int(epoch), 0

    def skip_batches(self, count):
        self.skip = int(count)

    def __len__(self):
        return self.size // self.global_batch - self.skip

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for update in range(self.skip, self.size // self.global_batch):
            group = order[update * self.global_batch : (update + 1) * self.global_batch]
            # Rotate the ranks assigned the extra prompts, without duplicating rows.
            slot = (self.rank - update) % self.replicas
            yield group[slot :: self.replicas]


class ExactCollate:
    def __init__(self, hidden_size, layers, maximum_tokens):
        self.hidden_size, self.layers = hidden_size, layers
        self.maximum_tokens = maximum_tokens

    def __call__(self, batch):
        # Failing a worker raises through torchrun. Never silently train a smaller
        # prompt batch because of a missing feature or an exhausted retry budget.
        if not batch or any(not isinstance(item, dict) for item in batch):
            raise RuntimeError("Missing clean features in an exact prompt batch")
        length = sum(int(item["input_ids"].numel()) for item in batch)
        if length <= 0 or length > self.maximum_tokens:
            raise ValueError("Exact prompt batch exceeds configured feature bound")
        result = CollateFn(length, self.hidden_size, self.layers, torch.bfloat16)(batch)
        if result["error_records"] or result["input_ids"].numel() != length:
            raise RuntimeError("The exact prompt batch lost or truncated a record")
        result["prompt_count"] = len(batch)
        return result


class LoopbackArrowDataset(ArrowDataset):
    def _setup_client(self):
        # The launcher binds a local-only target. Preserve CANN paths, but do not
        # route this local connection through an inherited authenticated proxy.
        import httpx
        from openai import OpenAI

        if self.client is None:
            client = OpenAI(
                base_url=self.vllm_endpoint,
                api_key="EMPTY",
                max_retries=0,
                timeout=self.request_timeout,
                http_client=httpx.Client(trust_env=False, timeout=self.request_timeout),
            )
            try:
                models = client.models.list().data
                if not any(model.id == self.model for model in models):
                    raise RuntimeError("Clean-feature server has the wrong served model")
                self.transfer.setup()
                self.client = client
            except BaseException:
                client.close()
                raise


def make_loader(dataset, *, global_batch, seed, hidden_size, layers, workers, row_limit):
    ranks, rank = get_dp_size(), get_dp_rank()
    sampler = PromptBatchSampler(len(dataset), global_batch, rank=rank, replicas=ranks, seed=seed)
    kwargs = {}
    if workers:
        kwargs.update(prefetch_factor=2, persistent_workers=True, multiprocessing_context="spawn")
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=ExactCollate(hidden_size, layers, math.ceil(global_batch / ranks) * row_limit),
        worker_init_fn=_worker_init_fn,
        **kwargs,
    )
