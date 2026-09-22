# ReTrace + DFlash training performance patch

This patch supplies optional acceleration for the **existing pretrained DFlash
online training recipe**. It does not replace your responses with offline labels,
change the prompt pool, shorten responses, alter the rejection rule, or promise
Table 2 reproduction. It changes only the independent ReTrace DFlash training
package and adds a benchmark launcher under `speculators/examples/train`.

There is **no measured Ascend speedup yet**. CPU tests establish correctness
properties, not NPU performance. Use the paired benchmark below before selecting
the execution mode for the remaining training.

## Changes

* Cache the draft's feature projection and each layer's context K/V projections
  during a single no-gradient rollout. Each committed context token is projected
  once. Query/mask tokens are recomputed normally. Cache contents are discarded
  before backward, the next prompt, and any optimizer update. The frozen target
  retains its existing incremental cache and its existing attention backend.
* Optionally keep detached FP16 rejected states and clean context on the training
  device until replay. The lifetime is one prompt; traces do not accumulate across
  a global batch. More device memory is required than the CPU-storage path.
* Replay 16 blocks per forward in the benchmark instead of the default four.
  This reduces the number of replay chunks by approximately four, not necessarily
  total training time by four. Every recorded block and label is still trained.
* In cached mode, omit context invisible to each replay chunk, combine memory
  transfers/casts, and avoid synchronizing loss/validity counters for every block
  chunk. A nonfinite loss still aborts before an optimizer update.
* Report synchronized rollout, backward, optimizer and complete-update timings,
  plus peak allocated device memory. Max-rank phase times need not sum to the
  update duration because workers can wait for different amounts of time.

The default remains `--performance-mode reference --trace-storage cpu`; installing
the patch alone does not silently change the execution mode of a new run. Cached
execution and different chunk sizes preserve the intended mathematics but can
change BF16 rounding, proposal choices, and the subsequent training trajectory.
They are not a promise of bitwise-identical resumed weights.

## 1. Preserve a completed checkpoint and stop the old DFlash job

At 40,000 prompts and global batch 32, epoch checkpoints occur every 1,250
updates. Wait for `Checkpoint: .../step_001250` (or a later checkpoint) before
stopping if you want to keep the completed work. Ctrl+C does not save an extra
checkpoint. Leave the DSpark job on NPUs 8–15 running.

Use the same Conda environment/runtime settings in which your DFlash run works.
Set `RETRACE_OLD` to the DFlash full-run output directory, **not** to its checkpoint
subdirectory, a smoke run, or a DSpark run:

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python
export RETRACE_OLD="$RETRACE_TRAIN_OUTPUT"

test -f "$RETRACE_OLD/prompts.jsonl"
cat "$RETRACE_OLD/checkpoints/latest.txt"
```

If `RETRACE_TRAIN_OUTPUT` is no longer set to the original DFlash run, assign
`RETRACE_OLD` its actual absolute path instead. The benchmark validates its prompt
pool hash, checkpoint state, and number of workers.

## 2. Install

Download `install_retrace_dflash_performance.py` into `$RETRACE_ROOT`, then run:

```bash
cd "$RETRACE_ROOT"
"$RETRACE_PYTHON" install_retrace_dflash_performance.py --root "$RETRACE_ROOT" --apply
```

The installer uses only Python's standard library. It verifies the pinned
speculators commit and known source hashes, reviews all files before modifying
any, saves originals under `retrace-performance-backups`, and supports a no-op
repeat installation. Omit `--apply` for a preview. Unknown edits are rejected;
there is no force-overwrite option. Original models, vLLM, vLLM-Ascend, datasets,
checkpoints and DSpark training files are untouched. Stop/restart DFlash Python
processes to load changes. Do not rerun an old bundle migration over this update.

## 3. Measure full-length training on the same saved checkpoint

```bash
export RETRACE_PERF_NPUS=0,1,2,3,4,5,6,7
export RETRACE_PERF_OUTPUT="$RETRACE_ROOT/output/retrace_dflash_perf_$(date +%Y%m%d_%H%M%S)"

bash "$RETRACE_ROOT/speculators/examples/train/retrace_dflash_performance.sh" \
  --run "$RETRACE_OLD" \
  --output "$RETRACE_PERF_OUTPUT" \
  --steps 6 \
  --warmup-updates 2 \
  --blocks 16
```

The two branches are run sequentially on NPUs 0–7. Both restore the **same saved
model, FP32 master parameters, AdamW state, RNG state, global batch, prompt order,
response length, and learning-rate schedule**. Only execution settings differ:

| Branch | Projections | Trace storage | Blocks per replay forward |
|---|---|---|---|
| `reference` | Original full-context path | CPU | Checkpoint's original setting |
| `cached` | Incremental draft context cache | Training device | 16 |

The first two updates in each branch are excluded from the timing average. Four
full-length updates per branch remain for comparison. This is a short estimate;
response lengths, acceptance and system load can change later. At your reported
71 seconds/update, the reference branch alone takes roughly seven minutes,
plus loading and checkpoint writing. No hardware-runtime estimate is assigned
to the cached branch before it is measured.

The original run is read-only. Both benchmark branches save a complete checkpoint
at their planned pause. Extra checkpoint storage is needed. If device allocation
fails, rerun into a new benchmark output directory with `--blocks 8`; do not
silently shorten the prompts or responses to obtain an apparently faster result.

Read:

```bash
cat "$RETRACE_PERF_OUTPUT/summary.json"
```

Key fields:

* `measured_update_speedup`: reference mean seconds/update divided by cached mean.
* `estimated_remaining_days_at_measured_rate`: remaining updates from the cached
  branch checkpoint times its measured update duration; excludes startup/save I/O.
* `mean_rollout_seconds_max`, `mean_backward_seconds_max`, and
  `mean_optimizer_seconds_max`: where time is being spent.
* `peak_memory_gib`: maximum allocated device memory in that branch.

Illustrative arithmetic for an **eight-day remaining baseline**, not measured
or predicted improvements from this patch:

| Measured speedup, if observed | Approximate remaining training |
|---|---|
| 1.5x | 5.3 days |
| 2x | 4 days |
| 3x | 2.7 days |
| 4x | 2 days |

If the target verification loop dominates, draft-side caching alone will have
a limited effect. The timing report will show whether batched target execution
or fused attention is the next useful optimization. Those changes are not
included or claimed to be validated by this patch.

## 4. Continue the measured cached branch

If the measured improvement is useful, run the exact resume script generated by
the benchmark:

```bash
bash "$RETRACE_PERF_OUTPUT/resume_fast.sh"
```

It restores the cached branch's **post-benchmark checkpoint**, so its six updates
are retained. It completes the original plan (10,000 total updates for your
40K/32/eight-epoch run). Checkpoints continue at epoch boundaries in:

```text
$RETRACE_PERF_OUTPUT/cached/checkpoints/
```

The benchmark's planned-pause checkpoint is additional to epoch checkpoints.
`metrics.jsonl` and `latest.txt` remain in that directory. Full resume output goes
to the terminal; `trainer.log` in each benchmark branch contains its benchmark
phase. The original prompt pool must remain available at its original path.

Alternatively, the existing pretrained full-training Bash now accepts:

```text
--performance-mode cached --trace-storage device --blocks-per-forward 16
--allow-performance-change --resume /absolute/path/to/a/completed/checkpoint
--pool /absolute/path/to/the/original/prompts.jsonl
```

Use a new output directory for that alternative if the old metrics extend beyond
the selected checkpoint. `--allow-performance-change` is an explicit migration
of only these three execution settings; it cannot change the dataset, target,
learning rate, global batch, epochs, precision, world size, or objective. The
optimizer, FP32 masters, RNG and step are restored, and changes are recorded in
`resume_command.json`. Do not shorten `--max-steps` for benchmarking; that changes
the learning-rate schedule. The supplied benchmark uses `--stop-after` instead.

## If no epoch checkpoint has been saved

The benchmark stops with a clear error; it does not pretend to recover unsaved
updates. You may wait for a completed checkpoint, or explicitly pass
`--from-initial` to benchmark/restart from the run's original imported pretrained
drafter. This option discards unsaved training progress and starts its schedule
at update zero. It does not change or delete the old run.

## Files and validation

Modified under `speculators/src/speculators/models/retrace/`:
`clean_training.py`, `train_pretrained.py`, `pretrained_launch.py`.

Added under that directory: `performance.py`, `performance_benchmark.py`.

Also added:
`speculators/examples/train/retrace_dflash_performance.sh`, this README under
`speculators/examples/train/README_retrace_dflash_performance.md`, and
`speculators/tests/unit/models/test_retrace_performance.py`.

Local validation: the ten existing pretrained-training checks passed. Seven
additional checks passed for cached FP32/BF16 outputs, projection reuse, cache
lifetime guards, full trajectories and detached memory, replay loss/gradients,
legacy-checkpoint continuation, resume recipe guards, and benchmark accounting.
These use a tiny CPU Qwen3 fixture. Real Qwen3-4B execution, HCCL behavior, NPU
memory use and speed must be measured with the supplied server benchmark.
