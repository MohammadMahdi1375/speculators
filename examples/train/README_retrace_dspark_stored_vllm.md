# Pretrained DSpark-block7 + ReTrace, stored-response vLLM training

This is the DSpark counterpart of the working DFlash stored-response trainer.
It starts from `/home/n84449292/m84379596/Huggingface/Qwen3-4B-DSpark-block7`,
preserves its pretrained backbone, Markov head and confidence head, and trains
ReTrace conditioning with clean DSpark supervision. Only the three ReTrace
conditioning matrices are newly initialized. The target stays frozen.

This is an experimental adaptation. The ReTrace paper's Table 2 uses DFlash;
DSpark plus stored-response round-pair sampling is a different experiment.
Matching 40K records, eight epochs and a learning rate does not establish
reproduction of that table. Hardware speed and native serving correctness need
measurement on your NPUs.

## Fixed full-run settings

| Setting | Value |
| --- | --- |
| Initial checkpoint | `Qwen3-4B-DSpark-block7` above; never a smoke checkpoint |
| Frozen target | Local Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c` |
| Data | `open_perfectblend.qwen3-4b-rollout.qwen3.seq3072` |
| Selected records | 40,000 eligible stored prompt/response records; seed 42 |
| Epochs | 8 |
| Target server NPUs | 8,9 |
| Trainer NPUs | 10,11,12,13,14,15; six DDP workers |
| Target port | 8523; distinct from the DFlash stored trainer's 8423 |
| Draft block | 7 proposals, including the anchor-slot prediction |
| Target KV page | 128; separate from the draft block size |
| Prompt / response limits | 512 / 1024 tokens; longer prompts skipped, responses truncated |
| Packed token budget | 3072 per trainer per update |
| Source round samples | Up to 16 per trainer per update |
| Blocks per forward | 4 |
| Target request concurrency | 8 per trainer |
| Precision | BF16 autocast, FP32 trainable parameters; detached FP16 memories |
| Optimizer | AdamW, LR 5e-5, weight decay 0.01, cosine, 5% warmup |
| ReTrace beta | 0 to 1 over 50 optimizer updates; restored step on resume |
| Loss | 0.1 CE + 0.9 TV + 1.0 confidence BCE; exp(-j/4), all seven slots |
| Checkpoints | End of every completed epoch |

The stock packed sampler determines updates per epoch from selected lengths;
this is not the old 10,000-update/global-32-prompt schedule. Its final tail may
drop fewer than six records to keep rank batch counts equal. All 40,000 selected
source IDs and the content hash are recorded in `stored_data/stored_manifest.json`.
The 16 sampled anchors need not cover every response position or every packed
document on a single visit. Clean features are re-extracted as needed by vLLM;
no full replacement responses are generated.

## Install and smoke test

Save `install_retrace_dspark_stored_vllm.py` in your `vLLM_NPU_spec_main` directory.
This installer requires the previously installed DSpark pretrained ReTrace
package at the inspected speculators revision. It checks source hashes, backs
up changes, and refuses unknown edits. It includes the C++ runtime bootstrap
needed by your Conda/CANN setup. It changes no vLLM, vLLM-Ascend, DFlash or
shared Trainer source files.

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python

"$RETRACE_PYTHON" "$RETRACE_ROOT/install_retrace_dspark_stored_vllm.py" \
  --root "$RETRACE_ROOT" --apply

export RETRACE_DSPARK_STORED_OUTPUT="$RETRACE_ROOT/output/retrace_dspark_stored_smoke_$(date +%Y%m%d_%H%M%S)"
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dspark_stored_vllm.sh" --smoke
```

Smoke selects 48 records, runs one epoch and samples four source rounds per
worker/update. Both CANN environment scripts are sourced by the Bash wrapper.
It preserves other runtime preloads while selecting the compatible Conda C++
runtime; no package upgrade or backend disabling is used.

Success means a clean exit, finite loss, no missing trainable gradients, a
positive `conditioned_pairs_global`, and a saved `checkpoints/0` checkpoint.
The first few updates produce per-rank diagnostics. An epoch with no usable
conditioned pairs is rejected instead of being labelled successful ReTrace.
The server log is `RUN/vllm_server.log`. This package was tested on CPU, including
BF16 autocast and the actual Arrow loader/Trainer checkpoint-resume path; it
has not been executed on your NPUs from this workspace.

Validation: 50 CPU tests passed; the optional multi-process Gloo test was
skipped. Coverage includes native DSpark loss equivalence, seven-slot alignment,
causal isolation, actual rejected-branch memory, gradients under BF16 autocast,
runtime checkpoint reload, and epoch save/resume through the shared Trainer.
Installer dry-run, hash refusal, backups, rollback, idempotence and imports from
the earlier DSpark package were also checked. This is not an NPU throughput test.

## Full training

After the DSpark smoke succeeds:

```bash
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dspark_stored_full.sh"
```

This starts a fresh full run from the original DSpark checkpoint on NPUs 8–15.
It can coexist with the DFlash run on NPUs 0–7. Ensure 8–15 are free of your
older DSpark training job before launching; the script does not kill other jobs.
To print the exact command without loading Torch or starting processes:

```bash
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dspark_stored_full.sh" --print-command
```

Default output:

```text
$RETRACE_ROOT/output/retrace_dspark_stored_full_TIMESTAMP/
  stored_run.json
  initial/dspark/                  # converted unchanged baseline
  initial/retrace_dspark/          # same pretrained tensors + zero residual
  stored_data/stored_manifest.json
  provenance/                     # vLLM command, revision and source patch
  metrics.jsonl
  vllm_server.log
  resume_stored.sh
  checkpoints/
    train_command.txt
    speculators.patch
    epoch_timing.jsonl
    0/                            # end of epoch 1
    1/                            # end of epoch 2
    ...
    7/                            # end of epoch 8
```

Each epoch checkpoint includes model/config, optimizer, scheduler and training
state. Descriptive `epoch0_end` etc. symlinks also identify epoch boundaries.
The console log is next to the run directory, at `RUN.log`. Use a distinct
`RETRACE_DSPARK_FULL_OUTPUT` environment variable to choose the full run path.
The full launcher ignores inherited smoke output and DFlash output variables.

Resume this protocol using the generated script:

```bash
bash /absolute/path/to/your/retrace_dspark_stored_full_TIMESTAMP/resume_stored.sh
```

Resume restores the latest saved epoch, optimizer/scheduler and global step.
It does not recover unsaved updates or import the old online trainer's optimizer
state. Stock checkpoint weights are serialized in BF16, so restarting is not
a bitwise-identical FP32 optimizer continuation.

## Implementation and limits

1. Import all pretrained DSpark tensors strictly, including Markov W1/W2 and
   confidence projection. Wrong architectures, missing tensors and shape
   mismatches fail. Initial ReTrace residual output is zero.
2. Read stored tokens and masks without decoding/re-tokenizing them. vLLM
   produces clean context features by full-sequence prefill.
3. Sample causal anchors. Generate seven proposals with the current DSpark
   Markov chain. Submit the actual proposed branches to vLLM concurrently.
   Its connector generates only one extra token to export features; that token
   is not used as a training response.
4. Drop accepted proposals and the first rejection. Keep the remaining actual
   branch draft/target states, detached in FP16. A conditioned successor is
   usable only if its committed prefix agrees with the stored sequence. Fully
   accepted or unaligned pairs do not invent rejected memory.
5. Preserve DSpark's slot alignment: slot j predicts token p+j+1, while the
   Markov training input is clean token p+j. Leave the verified anchor embedding
   unchanged; mask slots use memory indices 1 onward. Context attention is
   strictly before the anchor, with no cross-document or cross-block leakage.
6. Train with native DSpark semantics: CE targets are the clean teacher's
   argmax, TV uses its clean distribution, and confidence BCE targets detached
   distributional overlap. Reject-branch features condition the input only.
   They are not used as clean labels after the first rejection.
7. Use the shared Arrow loader and Trainer, with an isolated update loop that
   preserves update/checkpoint ordering and packs logging into one all-reduce.
   Proposal collection disables the autocast weight cache inside no-grad so
   subsequent backward retains trainable parameter gradients.

Only sampled round pairs are verified, avoiding full on-policy continuation
generation. Cached stored responses alone do not contain rejected-path states,
so these short branch queries remain necessary. The scope of speedup depends
on NPU execution, response lengths and pair alignment; no fixed speedup is
promised. Check `epoch_timing.jsonl` after an epoch for a useful run estimate.

This patch supplies training. Saved weights retain the runtime
`ReTraceDSparkDraftModel` schema, but it does not claim that native inference,
speedup over DSpark, or paper Table 2 numbers have been validated.

References: [ReTrace paper](https://arxiv.org/abs/2608.29748),
[pinned DSpark loss and confidence implementation](https://github.com/vllm-project/speculators/blob/e376ed8724f42e1c7a01df9f3ef870600ffb03e5/src/speculators/models/dspark/metrics.py).

## Files installed

Under `speculators/src/speculators/models/retrace_dspark/`:

- `config.py`: allow the separate `dspark_stored_pair` objective metadata.
- `stored_data.py`: select bounded stored records and write provenance.
- `stored_target.py`: concurrent actual-branch vLLM feature requests.
- `stored_training.py`: DSpark-specific causal block replay, Markov proposals,
  rejected-memory alignment and CE/TV/confidence training.
- `stored_loop.py`: packed metric reduction, gradients and bounded diagnostics.
- `stored_train.py`: model/data/Trainer setup and epoch coverage/timing.
- `stored_launch.py`: source import preflight, vLLM supervision, partitioned NPUs.

Under `speculators/examples/train/`:

- `train_retrace_dspark_stored_vllm.sh`: configurable entry point and smoke.
- `train_retrace_dspark_stored_full.sh`: fixed pretrained 40K/eight-epoch run.
- `retrace_dspark_runtime.py`: reuse the existing C++ runtime bootstrap; install
  it if absent and identical to the reviewed version.
- `README_retrace_dspark_stored_vllm.md`: this guide.

Under `speculators/tests/unit/models/`:

- `test_retrace_dspark_stored.py`
- `test_retrace_dspark_stored_metrics.py`
- `test_retrace_dspark_stored_launch.py`
