#!/usr/bin/env python3
"""CPU DDP regression runner using PyTorch's in-memory test process group.

This tests DDP's actual reducer/hooks without opening TCP sockets. It is not
an HCCL or multi-process hardware test; the launcher still requires that test.
Run in a fresh subprocess because the threaded backend changes global state.
"""
import argparse
import datetime
from pathlib import Path
import sys
import threading
import time
import traceback

import torch
import torch.distributed as dist
from torch.testing._internal.distributed.multi_threaded_pg import (
    ProcessLocalGroup, _install_threaded_pg, _uninstall_threaded_pg,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, choices=(2, 5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-gradient-drift", action="store_true")
    parser.add_argument("--no-recompute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts/dflash_prefix_joint"))
    import check_distributed
    import check_training
    torch.set_num_threads(1)
    # Fixture construction seeds PyTorch's process-global CPU generator.
    # Serialize construction; real forwards use different data per rank.
    lock = threading.Lock()
    original_fixture = check_training.make_fixture
    def fixture(*a, **kw):
        with lock:
            return original_fixture(*a, **kw)
    check_training.make_fixture = fixture
    original_check = check_distributed.check_agreement
    def checked(model, **kwargs):
        if (args.inject_gradient_drift and dist.get_rank() == 1
                and kwargs["phase"] == "after_backward" and kwargs["step"] == 2):
            with torch.no_grad():
                model.fc.weight.grad.flatten()[0].add_(1.)
        return original_check(model, **kwargs)
    check_distributed.check_agreement = checked
    settings = dict(block_size=16, rank=32, heads=4, top_k=16,
                    compiled_logit_cap=8., loss_alpha=.25, dense_floor=.25,
                    head_lr=3e-4, head_weight_decay=.01)
    if args.no_recompute:
        settings.update(backbone_checkpointing=False, selector_checkpointing=False,
                        anchor_chunk_size=32, guard_chunk_size=16)
    store, errors = dist.HashStore(), []
    def worker(rank):
        try:
            with torch.autograd.set_multithreading_enabled(False):
                dist.init_process_group("threaded", store=store, rank=rank,
                    world_size=args.ranks, timeout=datetime.timedelta(seconds=50))
                check_distributed.exercise(rank, args.ranks, args.output, "cpu", settings)
                dist.destroy_process_group()
        except BaseException as exc:
            errors.append(traceback.format_exc())
            ProcessLocalGroup.exception_handle(exc)
    torch._C._distributed_c10d._set_thread_isolation_mode(True)
    _install_threaded_pg()
    try:
        threads = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(args.ranks)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 60
        for thread in threads:
            thread.join(max(0., deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("In-memory DDP regression timed out")
        if errors:
            raise RuntimeError("\n".join(errors))
        print(f"JOINT_CPU_DDP_REGRESSION_PASS ranks={args.ranks}", flush=True)
    finally:
        _uninstall_threaded_pg()
        torch._C._distributed_c10d._set_thread_isolation_mode(False)


if __name__ == "__main__":
    main()
