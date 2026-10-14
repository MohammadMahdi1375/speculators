"""Real two-rank DDP forward/backward plus explicitly reduced guard updates."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest


ROOT = Path(__file__).resolve().parents[2]


def worker(rank, directory):
    import sys
    sys.path.insert(0, str(ROOT / "scripts/dflash_prefix_joint"))
    from check_distributed import exercise
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "rendezvous"),
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60))
    s = dict(block_size=8, rank=16, heads=2, top_k=16, compiled_logit_cap=8.,
        loss_alpha=.25, dense_floor=.25, head_lr=3e-4, head_weight_decay=.01)
    try:
        exercise(rank, 2, directory, "cpu", s)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_JOINT_DDP_TEST") != "1",
                   reason="Requires local TCP sockets; also covered by the automatic A2 HCCL preflight")
def test_ddp_full_model_guard_and_empty_rank(tmp_path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        report = json.loads((tmp_path / f"rank_{rank}/result.json").read_text())
        assert report["nonempty_guard_steps"] == 4 and report["bit_exact_rank_agreement"]


@pytest.mark.parametrize("ranks,no_recompute", [(2, False), (5, False), (5, True)])
def test_actual_ddp_hooks_survive_guard_and_disk_validation_in_memory(tmp_path, ranks, no_recompute):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("threaded_ddp_runner.py")),
        "--ranks", str(ranks), "--output", str(tmp_path)] + (["--no-recompute"] if no_recompute else []),
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(ranks):
        report = json.loads((tmp_path / f"rank_{rank}/result.json").read_text())
        assert report["bit_exact_rank_agreement"] and report["gradients_checked_every_step"]
        assert report["backward_after_saved_validation"] and report["nonempty_guard_steps"] == 4
        assert report["selector_checkpointing"] != no_recompute


def test_diagnostic_detects_drift_before_optimizer_without_repairing_it(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("threaded_ddp_runner.py")),
        "--ranks", "2", "--output", str(tmp_path), "--inject-gradient-drift"],
        capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    report = json.loads((tmp_path / "rank_1/failure.json").read_text())
    assert report["phase"] == "after_backward" and report["step"] == 2
    assert report["first_differences"][0]["name"] == "fc.weight"
    assert report["first_differences"][0]["max_abs_error"] > .99
