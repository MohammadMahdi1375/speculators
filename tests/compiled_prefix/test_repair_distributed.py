"""Exercise the pilot through two real DDP backward/optimizer iterations."""

import datetime
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from test_training import tiny_batch
from test_repair import compiled_checkpoint
from speculators.models.dflash_prefix.repair_training import build_repair_model


def _worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank,
        world_size=2, timeout=datetime.timedelta(seconds=45),
    )
    try:
        model = build_repair_model(
            root / "compiled", root / "relocated_target",
            {"objective": "renewal"},
        )
        model.train()
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=3e-4,
        )
        frozen = {name: parameter.detach().clone() for name, parameter in
                  model.named_parameters() if not parameter.requires_grad}
        # One rank has no valid labels: gradients must remain connected and
        # all-reduction must finish on the subsequent iteration too.
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, loss, _ = ddp(**tiny_batch(model, valid=rank == 0 or iteration == 1))
            assert torch.isfinite(loss)
            loss.backward()
            for parameter in model.parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None
                    assert torch.isfinite(parameter.grad).all()
            optimizer.step()
        for name, value in frozen.items():
            torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
        for parameter in model.parameters():
            if parameter.requires_grad:
                other = parameter.detach().clone()
                dist.broadcast(other, src=0)
                torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        (root / f"rank_{rank}_passed").write_text("passed\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("RUN_REPAIR_DDP_TEST") != "1",
    reason="Opt-in: requires a host permitting local Gloo TCP sockets",
)
def test_two_process_ddp_with_empty_label_rank(tmp_path):
    compiled_checkpoint(tmp_path)
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert all((tmp_path / f"rank_{rank}_passed").exists() for rank in range(2))
