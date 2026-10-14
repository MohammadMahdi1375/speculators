#!/usr/bin/env python3
"""Two real CPU DDP reducers reloaded from saved state, using an in-memory test backend."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys
import threading
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.multi_threaded_pg import ProcessLocalGroup, _install_threaded_pg, _uninstall_threaded_pg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts/dflash_prefix_joint"))
    from test_continuation import create_source, environment, cpu_and_hooks, Loader, assert_saved_optimizer_loaded
    from test_joint import helper
    from check_distributed import check_agreement
    from speculators.train.config import TrainConfig
    from speculators.train.trainer import Trainer, TrainerConfig
    from speculators.losses import resolve_loss_config
    from speculators.models.dflash_prefix.joint_continuation import FixedContinuationValidation
    torch.set_num_threads(1)
    source, output = args.output / "source", args.output / "continued"
    args.output.mkdir(parents=True, exist_ok=True)
    monkeypatch = pytest.MonkeyPatch()
    batch = create_source(source, "muon", monkeypatch)
    # Replicated cached fixture: both ranks load exactly the same saved model/optimizer payload.
    shutil.copytree(source / "validation_inputs/rank_0", source / "validation_inputs/rank_1")
    helper("prepare_continuation").prepare(dict(environment(source, output), TRAIN_NPUS="1,2", NUM_TRAIN_NPUS="2"))
    record = json.loads((output / "continuation.json").read_text())
    settings = json.loads((output / "joint_settings.json").read_text())
    flat = TrainConfig.resolve(["--config", str(output / "train_config.yaml")]).flatten()
    cli, tm = cpu_and_hooks(monkeypatch)
    # Production initializes module-level topology per process. Resolve it per
    # thread for this in-memory backend; the real Trainer/DDP code is unchanged.
    import speculators.train.checkpointer as checkpoint_module
    monkeypatch.setattr(tm, "get_rank", dist.get_rank)
    monkeypatch.setattr(tm, "is_distributed", lambda: True)
    monkeypatch.setattr(checkpoint_module, "get_rank", dist.get_rank)
    monkeypatch.setattr(checkpoint_module, "is_distributed", lambda: True)
    helper("continue_entry").install_hooks(cli, tm, settings, record)
    lock, errors = threading.Lock(), []
    store = dist.HashStore()

    def worker(rank):
        try:
            with torch.autograd.set_multithreading_enabled(False):
                dist.init_process_group("threaded", store=store, rank=rank, world_size=2,
                                        timeout=datetime.timedelta(seconds=45))
                with lock:
                    model = cli.build_draft_model(SimpleNamespace(**flat), None, None, None, 64)
                local = args.output / f"rank_{rank}"
                local.mkdir()
                rank_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                rank_batch["hidden_states"].add_(rank * .125)
                kwargs = dict(max_anchors=4, loss_config=resolve_loss_config("ce", "eager"))
                trainer = Trainer(model, TrainerConfig(lr=flat["lr"], muon_lr=flat["muon_lr"], optimizer="muon",
                    num_epochs=8, save_path=flat["save_path"], resume_from_checkpoint=True,
                    scheduler_type="cosine", train_call_kwargs=kwargs, val_call_kwargs=kwargs, log_freq=100), Loader(rank_batch),
                    FixedContinuationValidation(output / f"validation_inputs/rank_{rank}", record["validation_manifests"][f"rank_{rank}"]))
                assert isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
                assert trainer.current_epoch == 5 and trainer.global_step == 10
                assert_saved_optimizer_loaded(trainer.optimizers, source / "checkpoints/4")
                check_agreement(model, phase="reloaded_ddp", step=10, local=local)
                handle = model.register_forward_pre_hook(lambda module, inputs: setattr(module, "joint_step", trainer.global_step))
                checked = [0]
                original_step = trainer._optimizers_step
                def step():
                    check_agreement(model, phase="reloaded_backward", step=trainer.global_step, local=local, gradients=True)
                    original_step()
                    check_agreement(model, phase="after_guard_update", step=trainer.global_step, local=local)
                    checked[0] += 1
                trainer._optimizers_step = step
                # Direct epochs avoid installing main-thread-only signal handlers in this test backend.
                for epoch in (5, 6, 7):
                    trainer.train_epoch(epoch)
                    trainer.maybe_save_checkpoint(epoch)
                    dist.barrier()
                    metrics = trainer.val_epoch(epoch)
                    trainer.maybe_update_best(epoch, metrics)
                    dist.barrier()
                handle.remove()
                assert trainer.global_step == 16 and checked[0] == 6
                (local / "result.json").write_text(json.dumps(dict(pass_test=True, real_ddp_reducer=True,
                    in_memory_backend=True, hccl_test=False, restored_optimizer_state=True,
                    continued_optimizer_steps=checked[0], final_global_step=trainer.global_step,
                    epoch_indices=[5, 6, 7], gradients_agree=True, parameters_agree=True)))
                dist.destroy_process_group()
        except BaseException as exc:
            errors.append(traceback.format_exc())
            ProcessLocalGroup.exception_handle(exc)

    torch._C._distributed_c10d._set_thread_isolation_mode(True)
    _install_threaded_pg()
    try:
        threads = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(50)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("Continuation DDP test timed out")
        if errors:
            raise RuntimeError("\n".join(errors))
        print("JOINT_CONTINUATION_CPU_DDP_PASS ranks=2 epochs=6,7,8 optimizer_and_gradient_checks=True")
    finally:
        _uninstall_threaded_pg()
        torch._C._distributed_c10d._set_thread_isolation_mode(False)
        monkeypatch.undo()


if __name__ == "__main__":
    main()
