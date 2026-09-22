# Batched ReTrace + pretrained DFlash training

This update is for the already-installed performance patch. It is ready for a
short NPU benchmark; no NPU speedup for this second update has been measured.

## What your measurement established

| Measured phase | Original execution | First cached patch |
|---|---:|---:|
| Mean update | 77.17 s | 68.59 s |
| Maximum-worker rollout time, averaged | 65.74 s | 65.95 s |
| Maximum-worker backward time, averaged | 11.59 s | 2.55 s |

The first patch reduced backward work but left the dominant rollout cost almost
unchanged. Its measured speedup was 1.125x, or about 11.1% less wall time per
update. The reported 7.93 days remaining is consistent with those measurements.
Increasing the replay chunk further would address only a small remaining cost.

The old `optimizer_seconds_max` included gradient synchronization and the time
early workers spent waiting for the slowest worker. It did not measure 50.76
seconds of AdamW arithmetic. The new report separates worker waiting, gradient
synchronization, and clipping/AdamW time. Per-phase maxima can belong to
different workers and must not be added.

## What changes

Each NPU still contributes four prompts to the global batch of 32. With
`--rollout-batch-size 4`, it now executes those four live rollouts concurrently:

* One target forward verifies the next proposal block for every active prompt.
* One draft forward proposes blocks for those prompts together.
* Each prompt retains separate absolute positions, target K/V storage, clean
  context, accepted prefix, and previous-round rejected memory.
* Rejected cache suffixes are overwritten at their actual positions. A causal
  mask prevents stale suffixes or other requests from entering the prefix.
* Finished rows are removed, and their identity stays attached to their traces.
* Gradient replay still trains every recorded block with the existing per-prompt
  clean-label loss and normalization. The optimizer updates after the same 32
  prompts. The target stays frozen.

The prompt pool, generated-response budget, rejection rule, ReTrace conditioning,
global batch, epochs, learning-rate schedule, and pretrained identity are kept.
This does not switch to offline saved-response training. It also does not use a
vLLM server: the target remains the local frozen HF Qwen3 model on each NPU.

Batched matrix operations can change BF16 rounding and occasionally token
choices. This is not a bitwise-equivalence promise or a Table 2 reproduction
claim. Model caches and simultaneous traces use more device memory.

## Install the second update

The previously reported benchmark already saved a complete checkpoint at step
6. Preserve it. If a DFlash training process is currently running on NPUs 0–7,
finish/save or stop that process before launching the new benchmark. Continuing
from step 6 does not retain any later, unsaved updates from a resumed run.
The DSpark job on NPUs 8–15 can continue.

Download `install_retrace_dflash_batching.py` into the root below. Use your working
Conda/runtime environment, including any library-path setup already needed there.

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python

"$RETRACE_PYTHON" "$RETRACE_ROOT/install_retrace_dflash_batching.py" \
    --root "$RETRACE_ROOT" --apply
```

The standard-library installer verifies the pinned source and the first
performance patch, rejects unknown edits before making changes, and backs up
modified files under `retrace-batching-backups`. Repeating the same installation
is a no-op. Omit `--apply` for a preview. Do not rerun older migration/install
bundles over this update.

It updates only the independent ReTrace training package and its launcher/tests.
It does not edit stock DSpark/DFlash model code, vLLM, vLLM-Ascend, data or existing
checkpoints. It defaults to one rollout at a time until batching is requested.

## Benchmark from your saved checkpoint

Use the original full-run directory for its prepared prompt pool and the saved
cached benchmark checkpoint for weights, optimizer, FP32 master parameters and
all eight rank RNG states. These are deliberately different directories:

```bash
export RETRACE_OLD="$RETRACE_ROOT/output/retrace_dflash_pretrained_20260917_014120"
export RETRACE_BATCH_CHECKPOINT="$RETRACE_ROOT/output/retrace_dflash_perf_restart_20260917_235004/cached/checkpoints/step_000006"
export RETRACE_BATCH_NPUS=0,1,2,3,4,5,6,7
export RETRACE_BATCH_OUTPUT="$RETRACE_ROOT/output/retrace_dflash_batched_$(date +%Y%m%d_%H%M%S)"

bash "$RETRACE_ROOT/speculators/examples/train/retrace_dflash_batched_performance.sh" \
    --run "$RETRACE_OLD" \
    --checkpoint "$RETRACE_BATCH_CHECKPOINT" \
    --output "$RETRACE_BATCH_OUTPUT" \
    --steps 6 \
    --warmup-updates 2 \
    --rollout-batch-size 4 \
    --blocks 16 \
&& cat "$RETRACE_BATCH_OUTPUT/summary.json"
```

There is no `--from-initial` here: this continues saved progress. Both branches
restore the same step-6 state, run updates 7–12, and retain the original
10,000-update learning-rate schedule. `serial` is the first patch's cached path
with one prompt at a time. `batched` uses four prompt rollouts together. Both use
16 replay blocks and device-resident traces. They run sequentially on NPUs 0–7.

The first two updates in each branch are warmup and are omitted from the timing
average. Each branch saves at step 12 before exiting. The input checkpoint and
original prompt pool are read-only. These temporary planned-pause checkpoints
are additional to normal epoch-end saves.

Optional `--dry-run` prints the exact commands and checks metadata without
launching training or creating the output directory. If the batched branch runs
out of device memory, retry into a **new** output directory with
`--rollout-batch-size 2`. The global batch remains 32; each NPU then processes
two successive groups of two prompts.

Read these report fields:

* `measured_update_speedup`: serial-cached update time divided by batched update
  time, on the same planned updates.
* `estimated_remaining_days_at_measured_rate`: remaining updates times the
  measured batched update time, excluding startup and checkpoint I/O.
* `mean_target_forwards`: actual model forward calls, summed across workers.
* `mean_target_calls`: logical per-request target evaluations. This count still
  includes every request; it is not the number of batched model invocations.
* `mean_rollout_seconds_max`, `mean_backward_seconds_max`,
  `mean_worker_wait_seconds_max`, `mean_gradient_sync_seconds_max`,
  `mean_optimizer_seconds_max`: separate timing fields.
* `peak_memory_gib`: maximum allocated device memory for that branch.

Do not compare the new update-7–12 timing directly with the old update-3–6 timing
and call that an isolated speedup: the prompts and trained states differ. The
paired serial branch is included to make the comparison meaningful. A short
benchmark cannot guarantee the same gain later as lengths and acceptance change.

## Continue the selected branch

If batching provides a useful measured gain:

```bash
bash "$RETRACE_BATCH_OUTPUT/resume_batched.sh"
```

This resumes the batched branch at step 12, keeps its six new updates, and
continues the original epoch and learning-rate schedule. Checkpoints are saved
at epoch boundaries, starting at total step 1,250 for the current 40K/32 recipe:

```text
$RETRACE_BATCH_OUTPUT/batched/checkpoints/
```

If batching is not faster, `resume_serial.sh` instead resumes the paired serial
branch, also at step 12. Do not start both scripts on the same NPUs. The generated
resume scripts write full-training output to the terminal; each branch's
`trainer.log` covers its benchmark phase. Keep the original prompt pool available.

The existing full-training launcher also accepts `--rollout-batch-size 4` with
`--performance-mode cached`, `--trace-storage device`, and an explicit
`--allow-performance-change` when resuming. Changing the rollout microbatch is
allowed; changing the data, global batch, world size, target or training schedule
during a resume is still rejected.

## Files and validation

Modified files under `speculators/src/speculators/models/retrace/`:

* `performance.py`: recognize rollout batch size as an explicit execution-only
  resume setting; old checkpoints default to one.
* `train_pretrained.py`: concurrent collection, unchanged per-prompt replay,
  actual model-call counters, and separated distributed phase timing.
* `pretrained_launch.py`: forward the optional rollout batch setting.
* `clean_training.py`: concatenate FP16 stored memory outside autocast, then
  convert to network dtype; this also supports the BF16 CPU regression test.

Added files:

* `src/speculators/models/retrace/batched_training.py`
* `src/speculators/models/retrace/batched_benchmark.py`
* `examples/train/retrace_dflash_batched_performance.sh`
* `examples/train/README_retrace_dflash_batched_training.md`
* `tests/unit/models/test_retrace_batched_training.py`

All 27 CPU checks passed: ten new batching checks plus seventeen existing
pretrained/performance checks. The new checks cover unequal prompt lengths,
cache rollback, finished-row removal, real trajectories and rejected memory,
loss/gradients, EOS, all-accepted blocks/bonus tokens, final-length truncation,
BF16 with FP32 trainable parameters, checkpoint continuation and epoch saves,
optimizer step continuity, and benchmark/resume configuration guards.

The CPU multi-process test could not initialize Gloo because this execution
environment denied its socket operation. HCCL, real Qwen3-4B NPU memory use and
throughput have not been validated here. This is why the deliverable includes a
short measured comparison instead of claiming a particular speedup or launching
another unattended eight-day run.
