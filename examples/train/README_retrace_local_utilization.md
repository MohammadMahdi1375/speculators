# Local ReTrace execution with continuous rollout slots

This optional execution path is for the **non-vLLM** job on logical NPUs 8–15.
It leaves the existing local and vLLM launchers available and adds:

```
speculators/examples/train/train_retrace_dflash_local_fast.sh
```

## What changes

| Execution | Previous local default | This mode |
| --- | --- | --- |
| Concurrent rollouts per worker | 1 | 2 |
| Taking more prompts | After a complete prompt/cohort and its replay | Refill free slots while other trajectories remain live |
| Trace storage | CPU, copied every round | NPU; FP16 rejected memory retained |
| Training blocks per forward | 4 | 16 |
| Target/draft K/V | Existing serial/cohort caches | Reusable slot buffers with independent row positions |
| Host copies of verification results | Separate proposal, acceptance and correction transfers | One combined transfer per batched round |

Only active rows enter a transformer forward. Target and draft cache rows are
compacted on retirement; decode does not gather/copy every complete cache on
every round. Completed clean contexts are copied to private trace storage before
a slot is reused. Padding and rejected positions remain causally masked.

The optimizer update still contains **32 prompts across eight workers**. Workers
claim individual prompts atomically from that update, and every prompt must
appear exactly once before gradient averaging and the optimizer step. No prompt
from a future update is consumed; the draft weights remain fixed throughout the
current update. There are no stale rollout weights or cross-request memories.

Pretrained DFlash-b16 initialization, frozen Qwen3-4B target, 40K prompt pool,
eight epochs, LR schedule, ReTrace conditioning, clean CE labels and per-prompt
loss normalization remain the supplied paper_recipe choices. This optimization
does not resolve the author-code/protocol gaps documented with that recipe and
does not establish Table 2 reproduction.

## Continue the current local paper_recipe run

Let the old job save a checkpoint and stop that specific job before launching
another on the same devices. A source update cannot save unsaved in-memory
updates from an already running Python process.

Set `RETRACE_OLD` to the run directory containing `checkpoints/latest.txt`:

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python
export RETRACE_NPUS=8,9,10,11,12,13,14,15
export RETRACE_OLD=/absolute/path/to/the/current/local/paper_recipe/run
```

Direct continuation, in a new output directory:

```bash
unset RETRACE_OUTPUT
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_local_fast.sh" \
  --resume-run "$RETRACE_OLD" --allow-performance-change
```

The same checkpoint, prompt pool, optimizer moments, FP32 master parameters,
retained rank RNG states and global step are restored. Backend and worker count
must remain local and eight respectively. This does not reset epochs, warmup or
the learning-rate schedule. BF16 batching and different work assignments may
change numerical results; bitwise equivalence is not promised.

Epoch-end saving remains the default. With 40,000 prompts and batch 32, epoch
checkpoints are `checkpoints/step_001250` through `checkpoints/step_010000`.
Normal completion and `--stop-after N` also save a completed checkpoint.

## Measure the difference on the same checkpoint

```bash
export RETRACE_OUTPUT="$RETRACE_ROOT/output/retrace_local_compare_$(date +%Y%m%d_%H%M%S)"
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_local_fast.sh" \
  --compare-run "$RETRACE_OLD"
```

This runs the checkpoint's recorded execution settings and the new settings
sequentially, each for six additional updates, excluding two warmup updates
from the timing summary. Both start from the identical saved model, optimizer
and prompt position. It preserves the full training schedule. The two branches
may take different BF16 greedy paths; logical rounds and clean label counts are
reported so changes in workload remain visible.

Read `summary.json`, then continue the faster measured branch:

```bash
bash "$RETRACE_OUTPUT/resume_best.sh"
```

The original run is preserved. The selected branch retains its six completed
comparison updates. A short sample does not guarantee a sustained speedup; the
comparison excludes startup/checkpoint I/O. An incomplete run without a saved
checkpoint is rejected rather than silently restarting from pretrained weights.

## Start fresh if no checkpoint needs preserving

```bash
unset RETRACE_OUTPUT
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_local_fast.sh" --smoke
```

The smoke uses short prompts/responses and three updates. It verifies startup,
not full-run utilization. Omit `--smoke` for the full pretrained run. Use
`--stop-after 3` on a fresh full run to time three full-size updates and save a
continuation point without shortening the learning-rate schedule.

## Memory and utilization

The posted snapshot showed around 22 GB of 64 GB allocated per device. That is
memory occupancy, not arithmetic utilization. Extra useful concurrency can
help, while simply allocating more memory cannot. The percentage in one
`npu-smi` snapshot is also not a sustained utilization measurement.

Watch these fields in `checkpoints/metrics.jsonl`:

- `step_seconds`: optimize this at the same global prompt count.
- `mean_active_rollouts_per_draft_forward`: actual useful rows per draft call.
- `worker_wait_seconds_max`: remaining imbalance at the update boundary.
- `peak_memory_gib_max`: measured allocation peak, not total device usage.
- `rank_performance`: work and wait time for each rank.

Two active slots leave a pool of unclaimed prompts for load balancing. Four
slots across eight workers immediately claim all 32 prompts and can worsen the
tail when response lengths differ. Setting 8 or 16 slots without changing the
global batch can leave entire workers without prompts. Increasing global batch
size would change this training experiment and is not done here.

You can compare `RETRACE_ACTIVE_ROLLOUTS=4` explicitly if the default does not
help. `RETRACE_BLOCKS_PER_FORWARD=8` reduces replay memory if necessary. There
is no automatic OOM retry that silently drops or repeats prompts. Keep the
setting only if measured step time improves. This CPU-validated implementation
has no measured Ascend speedup or utilization percentage yet.

## Added source files

All new Python files live under `src/speculators/models/retrace/paper_recipe`:

- `continuous_training.py`: reusable caches and continuous slot refill.
- `utilization_train.py`: unchanged optimizer/recipe with the new collector.
- `utilization_launch.py`: local NPU launch and checkpoint preflight.
- `utilization_resume.py`: shared resume checks and execution-only migration.
- `utilization_compare.py`: paired timings and saved continuation scripts.

The installer also adds this README and the training bash. Original DFlash,
DSpark, ReTrace, vLLM and vLLM-Ascend implementation files are not overwritten.
