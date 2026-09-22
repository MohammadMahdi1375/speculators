# DFlash + ReTrace: attention and worker scheduling update

This update corrects avoidable execution restrictions in the existing pretrained
DFlash + ReTrace trainer. It enables the stock DFlash/Transformers SDPA attention
path, adds optional fused target RMSNorm, and assigns prompts dynamically within
each optimizer update. It preserves the saved training recipe and resumes the
optimizer, FP32 master parameters, update number, and learning-rate schedule.

**Status:** 27 CPU regression tests passed, including real checkpoint resume and
multi-process prompt claims. Ascend operator behavior, HCCL execution, and speed
still require the supplied short check on your machine. No measured speedup is
claimed in this package. This is a maintained implementation of the published
method, not a certified reproduction of Table 2.

## What was corrected

| Existing restriction | Change |
|---|---|
| ReTrace forced eager attention, including inside draft forward calls | An explicit `--attention-backend sdpa` now reaches target, draft, cached proposals, and packed training blocks. The original causal and block masks remain in effect. |
| Every worker received four fixed prompts regardless of completion time | `--work-distribution dynamic` uses a shared CPU queue for the same 32-prompt update. A worker that finishes takes another unclaimed prompt. |
| Frozen Qwen3 norms used sequences of small operations | `--target-norm npu` uses `torch_npu.npu_rms_norm` with the original frozen weights and epsilon. Trainable draft norms retain their original implementation. |
| Device utilization could not be inferred from aggregate phase maxima | Profiled updates now record each rank's compute, wait, prompt count, and memory. |

The queue never crosses an optimizer-update boundary. All workers use fixed
parameters until every prompt in that global batch has contributed. Coverage is
checked for omissions and duplicates. Gradients are summed and divided by the
**global prompt count**, including when workers process different numbers of
prompts or a worker has none. Attention dropout must be zero in dynamic mode.

The optimized preset keeps cached draft context, device-resident traces, 16
training blocks per forward, and rollout microbatch **1**. Your previous batch-4
comparison did not improve update time, so batch 4 is not the new default.
FP16 detached memory and BF16 forward/FP32 optimizer behavior are preserved.

Transformers 5.14.1's SDPA adapter converts zero/negative-infinity attention masks
to the boolean form needed by Ascend's optimized attention path. Backend
selection itself is not evidence that every shape uses a fused kernel. The NPU
check compares forward results and q/k/v gradients for prefill, verification,
draft, and packed-block masks before launching either comparison branch.

## Method and reproduction limits

The paper specifies pretrained DFlash-b16, a frozen target, detached FP16
one-round states, exclusion of the accepted prefix and first rejection, and
causally aligned target scoring states. Correction and gated residual fusion
follow Equations 7–9; the value projection starts at zero and beta warms over 50
updates. The draft and conditioner share the prediction objective. Appendix B–D
specifies 40K prompts, seed 42, batch 32, eight epochs, 512/1024 token limits,
BF16, AdamW with weight decay 0.01, and cosine LR 5e-5 with 5% warmup. It reports
32 A800 GPUs with FSDP. [ReTrace paper, Equations 5–10 and Appendix B–D](https://arxiv.org/pdf/2608.29748)

Our current trainer collects a complete greedy target-verified continuation with
fixed draft weights, retains real rejected states, and replays its live round
anchors against clean continuation labels. The anchor selection, greedy training
rollouts, per-prompt reduction, and collection/replay schedule are implementation
choices; the paper does not give enough detail to identify them as the authors'
exact loop. Gamma 4 is inherited from the DFlash loss implementation. This update
changes execution, not those choices. The run uses eight replicated Ascend
workers rather than the authors' hardware and FSDP configuration.

The saved Arrow responses are not substituted for target verification states:
clean-response states do not contain the model's response to an actual rejected
draft path. Reusing only those states would train a different conditioning
distribution. Offline or sampled two-round training would need its own explicit
protocol and quality comparison.

The old 9-hour DFlash epoch processed stored full sequences with a different
target-feature pipeline. This trainer still performs sequential live verification
rounds, so this update does **not** imply a 9-hour epoch or a particular speedup.
Full-continuation collection remains a material runtime cost. There is no claim
that this is the fastest attainable architecture.

## Install on your existing tree

Download `install_retrace_dflash_execution.py` into:

```text
/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/
```

Use the conda shell in which torch_npu already imports successfully. Keep any
working libstdc++/LD_PRELOAD environment fix. Do not run a competing job on the
same selected devices. The installer does not stop jobs or save their in-memory
progress; a running old process must be restarted to use this update.

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python

"$RETRACE_PYTHON" "$RETRACE_ROOT/install_retrace_dflash_execution.py" \
  --root "$RETRACE_ROOT"

"$RETRACE_PYTHON" "$RETRACE_ROOT/install_retrace_dflash_execution.py" \
  --root "$RETRACE_ROOT" --apply
```

The installer expects your previous cached/batched training update and pinned
speculators commit `e376ed8724f42e1c7a01df9f3ef870600ffb03e5`. It checks all source
hashes before writing. Unrecognized edits stop installation without replacing
them. Original files are backed up under `retrace-execution-backups/`.

## Check and continue your saved run

These commands use the last confirmed complete **serial** checkpoint, step 12.
If you have a newer complete checkpoint, set `RETRACE_EXEC_CHECKPOINT` to that
directory instead. An old process's unsaved updates cannot be recovered by
installing this patch.

```bash
export RETRACE_EXEC_NPUS=0,1,2,3,4,5,6,7
export RETRACE_EXEC_RUN="$RETRACE_ROOT/output/retrace_dflash_pretrained_20260917_014120"
export RETRACE_EXEC_CHECKPOINT="$RETRACE_ROOT/output/retrace_dflash_batched_20260918_013945/serial/checkpoints/step_000012"
export RETRACE_EXEC_OUTPUT="$RETRACE_ROOT/output/retrace_dflash_execution_$(date +%Y%m%d_%H%M%S)"

bash "$RETRACE_ROOT/speculators/examples/train/retrace_dflash_execution_check.sh" \
  --run "$RETRACE_EXEC_RUN" \
  --checkpoint "$RETRACE_EXEC_CHECKPOINT" \
  --output "$RETRACE_EXEC_OUTPUT" \
  --steps 4 --warmup-updates 1 --blocks 16

cat "$RETRACE_EXEC_OUTPUT/summary.json"
```

This runs an operator check, then four complete updates in each branch from the
same checkpoint and pool. The first update in each branch is excluded from the
timing summary. With step 12 as input, both branches save step 16. The original
checkpoint is read-only. Each branch retains the original 10,000-update schedule;
the check does not restart its learning-rate or beta warmup.

If `fast_mode_faster` is true and the check completes, continue its saved progress:

```bash
bash "$RETRACE_EXEC_OUTPUT/resume_fast.sh"
```

Otherwise the completed baseline can be continued with:

```bash
bash "$RETRACE_EXEC_OUTPUT/resume_baseline.sh"
```

The measured speedup is baseline mean update time divided by fast mean update
time. The ETA uses the remaining update count and the measured mean; it excludes
startup and checkpoint writes and may change with sequence lengths and acceptance.
Numerical backend and reduction-order changes can alter BF16 trajectories; this
is not a bitwise equivalence or native-inference parity test.

If fused RMSNorm is unsupported or its guard fails, retain SDPA and dynamic work
assignment by repeating into a **new** comparison output with
`--target-norm reference`. There is no silent fallback or ignored failed check.

## Full Bash for a new pretrained run

The installed full launcher is:

```text
speculators/examples/train/train_retrace_dflash_pretrained_full.sh
```

After the device check above, a new run can be launched with:

```bash
RETRACE_TRAIN_OUTPUT="$RETRACE_ROOT/output/retrace_dflash_sdpa_$(date +%Y%m%d_%H%M%S)" \
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_pretrained_full.sh"
```

This new-run command imports
`/home/n84449292/m84379596/Huggingface/Qwen3-4B-DFlash-b16` and prepares a 40K
prompt pool from your local Arrow dataset. It runs the operator guard before
training. **Use `resume_fast.sh` above to retain step-12 progress**, rather than
starting this new run. The old `retrace_qwen3_4b_perfectblend_online.sh` is the
earlier server-based launcher and is not the recommended pretrained path.

Defaults remain NPUs 0–7, global prompt batch 32, eight epochs, and checkpoints
at each epoch end. For 40,000 prompts this means steps 1250, 2500, ..., 10000.
The comparison additionally saves at its planned pause. Continuation checkpoints
go to `RETRACE_EXEC_OUTPUT/fast/checkpoints/`; new-run checkpoints go to
`RETRACE_TRAIN_OUTPUT/checkpoints/`. `latest.txt` points to a completed checkpoint.
Increasing rollout microbatch or global batch merely to fill memory is not part
of this preset.

## Source files

All paths below are relative to `speculators/`.

| File | Role |
|---|---|
| `src/speculators/models/retrace/execution.py` | SDPA dispatch, frozen-target NPU norms, atomic prompt queue and coverage checks |
| `src/speculators/models/retrace/execution_check.py` | Device attention/mask/gradient and RMSNorm checks |
| `src/speculators/models/retrace/execution_benchmark.py` | Saved-checkpoint comparison, summary and continuation scripts |
| `src/speculators/models/retrace/core.py` | Stop forcing eager inside proposal/training calls |
| `src/speculators/models/retrace/runtime.py` | Apply explicit target/draft execution settings |
| `src/speculators/models/retrace/performance.py` | Cached attention dispatch and resume-signature compatibility |
| `src/speculators/models/retrace/batched_training.py` | SDPA-compatible batched execution for optional later tuning |
| `src/speculators/models/retrace/train_pretrained.py` | Work queue, exact prompt accounting, rank-level timings |
| `src/speculators/models/retrace/pretrained_launch.py` | Forward execution settings and run the device guard |
| `examples/train/train_retrace_dflash_pretrained_full.sh` | Updated full pretrained training preset |
| `examples/train/retrace_dflash_execution_check.sh` | Bounded NPU comparison entry point |
| `examples/train/README_retrace_dflash_execution.md` | These instructions |
| `tests/unit/models/test_retrace_execution.py` | Attention, gradients, queue, resume, and recipe regression coverage |

Stock DFlash and DSpark source files, ReTrace-DSpark, vLLM, vLLM-Ascend, model
weights, datasets, and existing checkpoint directories are not modified by this
installer. Native serving is not certified by the training comparison.

Validation in this workspace used torch/Transformers on CPU. The suite covered
SDPA forward and gradient agreement, clean-label replay with nonzero
conditioning, prefix padding, cache rollback and row removal, BF16 with FP32
masters, concurrent FileStore claims, unequal-worker gradient normalization,
resume from an older checkpoint schema, optimizer step counters, and epoch-end
saves. Shell syntax and Python error/import lint checks passed. No NPU training
or HCCL collectives were executed here.
