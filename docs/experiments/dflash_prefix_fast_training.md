# Faster joint DFlash Prefix training

The joint launcher now defaults to `EXECUTION_PROFILE=fast`. This is an execution
change, with the same model, data, packed token batch, sampled anchors, loss
weights, optimizer, clipping, scheduler, guard frequency and epoch budget.
AR is still enabled explicitly with `AR_LOSS_WEIGHT=0.1`.

## Execution profiles

| Setting | `fast` | `memory` |
| --- | --- | --- |
| `BACKBONE_CHECKPOINTING` | `0` | `1` |
| `SELECTOR_CHECKPOINTING` | `0` | `1` |
| `AR_LOGIT_CHECKPOINTING` | `0` | `1` |
| `ANCHOR_CHUNK_SIZE` | `8` | `8` |
| `DDP_GRADIENT_AS_BUCKET_VIEW` | `1` | `0` |
| `GUARD_REUSE_WORKSPACES` | `1` | `0` |

Explicit environment values override the profile. A standalone invocation of
`prepare_experiment.py` without a profile retains the previous memory defaults.
The launcher supplies `fast` by default. Effective values are printed and saved
in `joint_settings.json` and `resolved_schedule.json`.

The fast path retains the activations needed for backward, avoiding another
decoder, selector and AR vocabulary-projection evaluation. It requires more NPU
memory. DDP gradient bucket views avoid duplicate gradient buffers/copies after
the first iteration. The selector guard reuses separate FP32 gradient and BF16
replay copies, refreshing their weights before every use, including backtracking.
It never changes live DDP parameter storage and retains optimizer rollback.

The training loop also clears completed-step gradients before the next forward,
packs scalar logging metrics into one all-reduce and host transfer per dtype,
and relies on DDP's initialization sync instead of broadcasting every parameter
twice. These changes also apply to the memory profile. Loss gradients are not
compressed or cast by these logging optimizations.

Selector and AR projection chunk sizes remain unchanged. Increasing selector
chunks changed BF16 gradients in a local numerical comparison, so it is not a
default optimization. Packed batch size, anchor counts, guard interval and
validation coverage are not reduced to manufacture a throughput gain.

## Run on A2

Activate the existing environment and run from the updated `speculators` repo:

```bash
AR_LOSS_WEIGHT=0.1 AR_PRETRAIN_STEPS=0 AR_MAX_ANCHORS=64 \
EXECUTION_PROFILE=fast \
BACKBONE_CHECKPOINTING=0 SELECTOR_CHECKPOINTING=0 AR_LOGIT_CHECKPOINTING=0 \
ANCHOR_CHUNK_SIZE=8 DDP_GRADIENT_AS_BUCKET_VIEW=1 GUARD_REUSE_WORKSPACES=1 \
EPOCHS=5 \
bash examples/train/train_dflash_prefix_joint.sh
```

The explicit controls also override older exports in your shell. Existing model,
dataset and `BASE_TRAIN_CONFIG` paths are used. The default allocation is all
eight NPUs: two vLLM servers and six DDP trainers. The existing single-NPU and
HCCL preflights run before full training. `PREFLIGHT_ONLY=1` stops after those
checks. The tiny preflight does not establish full-model memory capacity.

If the full workload runs out of memory, first try
`AR_LOGIT_CHECKPOINTING=1`, retaining the other fast controls. If necessary also
set `BACKBONE_CHECKPOINTING=1`, followed by `SELECTOR_CHECKPOINTING=1`. Keep the
same batch, anchors and loss coefficients. Use a fresh output directory on a
fresh restart; after saved progress use the continuation command below.

Do not compare the old `profile.step_ms` alone: it excludes logging collectives
after the optimizer. Compare wall time over the same optimizer-step range,
after startup, excluding checkpoint/validation boundaries. Record memory peaks
from `JOINT_TRAIN_MEMORY`. Measure before and after the guard activates because
its replay and backtracking contribute work. No A2 speedup or memory figure is
claimed from CPU tests.

## Resume with faster execution

Resume keeps source execution settings unless explicitly overridden. Regular
`EXECUTION_PROFILE` and `AR_LOSS_WEIGHT` exports do not replace the saved recipe.

```bash
RESUME_EXECUTION_PROFILE=fast \
EPOCHS=5 \
bash examples/train/train_dflash_prefix_joint.sh \
  --resume-training /absolute/path/to/run/checkpoints/SELECTED_CHECKPOINT
```

`EPOCHS` is the total target; choose a value beyond the completed progress.
Individual overrides use `RESUME_BACKBONE_CHECKPOINTING`,
`RESUME_SELECTOR_CHECKPOINTING`, `RESUME_AR_LOGIT_CHECKPOINTING`,
`RESUME_ANCHOR_CHUNK_SIZE`, `RESUME_DDP_GRADIENT_AS_BUCKET_VIEW` and
`RESUME_GUARD_REUSE_WORKSPACES`. They do not alter saved AR weight, warmup stage,
optimizer state, data or batch. Changes are recorded in `continuation.json`.
Existing continuation limitations still apply: BF16 exports, RNG and guard
replay prevent bit-exact resumption of older runs.

## FSDP2 decision

This update keeps DDP. FSDP2 shards parameters and gathers them for computation;
it is not automatically faster. This trainer's selector guard inspects and
updates complete FP32 parameters outside the model forward, uses separate
shadow modules and may restore optimizer state. A naive `fully_shard` switch
does not preserve that contract. Correct integration would also require
validation of Muon, clipping, export and continuation with DTensors. The joint
launcher continues rejecting unsupported FSDP configurations.

References:
- [PyTorch FSDP2 contract](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html)
- [DDP gradient bucket views](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)

## Verification scope

Tests compare fast versus recomputed joint AR/parallel loss, every trainable
gradient and one AdamW update in FP32 and BF16, retaining identical chunking.
Guard tests compare cached and fresh copies across updates, including rejection,
parameter and moment rollback, and absence of hooks on live parameters.
Distributed tests exercise actual DDP reducers on two and five ranks, bucket
views, zero-replay ranks, cached guard copies and backward after disk validation.
CPU in-memory process groups do not substitute for A2/HCCL or full-model tests.
These are numerical/correctness checks, not a measured convergence guarantee.
