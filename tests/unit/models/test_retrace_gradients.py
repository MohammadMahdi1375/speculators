import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from speculators.models.retrace.train import average_gradients


def worker(rank, init_file, output):
    dist.init_process_group(
        "gloo", init_method="file://" + init_file, rank=rank, world_size=2
    )
    try:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = (
            torch.tensor([2.0, 4.0]) if rank == 0 else torch.tensor([9.0, 12.0])
        )
        assert (
            average_gradients([parameter], [1, 3][rank], torch.device("cpu"), True) == 4
        )
        torch.testing.assert_close(parameter.grad, torch.tensor([2.75, 4.0]))
        parameter.grad = torch.tensor([3.0, 6.0]) if rank == 0 else None
        assert (
            average_gradients([parameter], [2, 0][rank], torch.device("cpu"), True) == 2
        )
        torch.testing.assert_close(parameter.grad, torch.tensor([1.5, 3.0]))
        assert average_gradients([parameter], 0, torch.device("cpu"), True) == 0
        if rank == 0:
            torch.save(parameter.grad, output)
    finally:
        dist.destroy_process_group()


def test_uneven_and_exhausted_ranks_average_by_actual_rounds(tmp_path):
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip(
            "This execution environment prohibits TCP sockets; run the Gloo test on the host"
        )
    try:
        mp.spawn(
            worker,
            args=(str(tmp_path / "init"), str(tmp_path / "result.pt")),
            nprocs=2,
            join=True,
        )
    except mp.ProcessRaisedException as error:
        if "gloo/transport/tcp/device.cc" in str(
            error
        ) and "Operation not permitted" in str(error):
            pytest.skip(
                "Environment denies Gloo TCP device setup; run this test on the host"
            )
        raise
    torch.testing.assert_close(
        torch.load(tmp_path / "result.pt", weights_only=True), torch.tensor([1.5, 3.0])
    )


def test_reduction_sums_then_divides_by_actual_round_count(monkeypatch):
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p.grad = torch.tensor([2.0, 4.0])
    # Other worker contributes 3 rounds and summed gradient [9,12].
    contributions = iter([torch.tensor(3.0), torch.tensor([9.0, 12.0])])
    monkeypatch.setattr(dist, "all_reduce", lambda x: x.add_(next(contributions)))
    assert average_gradients([p], 1, torch.device("cpu"), True) == 4
    torch.testing.assert_close(p.grad, torch.tensor([2.75, 4.0]))


def test_exhausted_rank_still_participates_with_zero_gradients(monkeypatch):
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    contributions = iter([torch.tensor(2.0), torch.tensor([3.0, 6.0])])
    monkeypatch.setattr(dist, "all_reduce", lambda x: x.add_(next(contributions)))
    assert average_gradients([p], 0, torch.device("cpu"), True) == 2
    torch.testing.assert_close(p.grad, torch.tensor([1.5, 3.0]))
