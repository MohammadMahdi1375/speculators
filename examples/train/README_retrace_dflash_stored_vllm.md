# Stored-response DFlash + ReTrace through vLLM

This is a separate, experimental training protocol. It reuses the stored
responses in the local Open-PerfectBlend Arrow dataset, the stock DFlash data
loader/packed batches, its clean-target KL loss, and the stock epoch trainer.
The frozen Qwen3 target runs in vLLM on NPUs 0–1; DFlash/ReTrace training uses
NPUs 2–7. There is no full target model or autoregressive response-generation
loop in a trainer worker.

The previous full-response online trainer was an implementation choice. It
should not have been described as the only way to train ReTrace. Its hard-token
CE objective also differs from the pinned speculators DFlash default KL loss.
This implementation uses that DFlash KL objective and exponential slot weights.

## What is preserved and what changes

| Component | Stored-response mode |
|---|---|
| Initialization | Strictly imported local `Qwen3-4B-DFlash-b16`; zero-initialized ReTrace value projection |
| Target | Frozen Qwen3-4B, shared vLLM server |
| Clean context and supervision | Actual stored prompt/response token IDs; target features extracted in parallel prefills |
| Rejected memory | Fresh proposed branches evaluated by vLLM; the trainer applies the frozen target norm/head for greedy verification |
| Memory alignment | Drops accepted prefix and first rejection; causal target scoring shift; detached FP16 memory; one round |
| Conditioning | Existing ReTrace correction, gate, value residual, and 50-update beta ramp |
| Objective | Stock DFlash clean-target KL, decay `exp(-j/4)` over proposal slots; frozen head/norm |
| Sampling | Independent source/successor round pairs, not complete online rollouts |
| Batch unit | 3,072 packed stored tokens per trainer, variable prompt count |
| Checkpoints | At every completed epoch, using the stock DFlash trainer |
| Inference | Same `retrace` architecture and tensor schema as existing runtime |

The [ReTrace paper](https://arxiv.org/pdf/2608.29748), Appendix B, specifies a
40K **prompt-only** pool. It does not describe this stored-response round-pair
sampler. This is not a certified reproduction of Table 2. In particular, fixed
responses, pair sampling, prefix-agreement filtering, variable prompt batches,
DDP on six trainer NPUs, and the Ascend/vLLM backend differ from the published
experiment. Similar hyperparameters do not establish identical training.

The pinned [DFlash training path](https://github.com/vllm-project/speculators/blob/e376ed8724f42e1c7a01df9f3ef870600ffb03e5/src/speculators/train/data.py)
sends stored token sequences to the
[vLLM hidden-state client](https://github.com/vllm-project/speculators/blob/e376ed8724f42e1c7a01df9f3ef870600ffb03e5/src/speculators/data_generation/vllm_client.py).
The client requests one output token to export prefill features; this is not a
new full response. This launcher uses the same mechanism for clean sequences
and actual proposed branches.

## Why there are still branch requests

A clean response does not contain target states for rejected tokens that the
drafter actually proposed. After divergence, those states depend on a different
prefix. Substituting the stored clean states would change the conditioning
signal.

For each packed batch, the code selects up to 16 source anchors per trainer,
drafts independent blocks, and submits their actual branch tokens concurrently
to vLLM. It trains the next conditioned block only if the accepted proposals and
target correction match the corresponding stored sequence. Unaligned pairs are
counted; their clean source blocks still receive the ordinary DFlash loss. A
whole epoch without usable conditioned pairs raises an error. This filter can
reduce coverage, especially with stochastic stored responses, and introduces a
sampling bias. Each source round starts without carried memory; the sampler
does not train the full distribution of long recursive conditioning chains.

Clean future states supply supervision only. The attention mask prevents each
draft block from attending to context at/after its own anchor. Packed documents
are split before block drafting, so context and memory cannot cross samples.
No gradients flow through the verification pass or retained states.

## Install and smoke test

Put `install_retrace_dflash_stored_vllm.py` in your existing repository root.
After stopping the previous trainer on NPUs 0–7, run:

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python

"$RETRACE_PYTHON" "$RETRACE_ROOT/install_retrace_dflash_stored_vllm.py" \
  --root "$RETRACE_ROOT" --apply

export RETRACE_STORED_OUTPUT="$RETRACE_ROOT/output/retrace_stored_smoke_$(date +%Y%m%d_%H%M%S)"
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_stored_vllm.sh" \
  --smoke
```

The self-contained installer checks the inspected source hashes before any
writes, creates backups, and does not change DFlash, DSpark, vLLM, vLLM-Ascend,
datasets or previous checkpoint directories. It adds five modules, a launcher,
this README and tests; it adds a distinct objective name to ReTrace's config.
The shell launcher sources your CANN 9.1.0 environment and retains any existing
`LD_PRELOAD` workaround.

Smoke mode selects 48 stored responses, runs one epoch, uses four source pairs
per packed batch, and saves an epoch checkpoint. The target server startup log
is `$RETRACE_STORED_OUTPUT/vllm_server.log`. Training metrics are in
`$RETRACE_STORED_OUTPUT/metrics.jsonl`.

Check `conditioned_pair_fraction` and `conditioned_positions`: they must be
nonzero over the smoke epoch. Check the loss is finite and that the epoch save
completed. A successful CPU test does not replace this NPU/vLLM smoke run.

## Full training

Use a new directory after the smoke test. The full run starts from your public
pretrained DFlash checkpoint again, with a new optimizer; it does not resume the
old online-training protocol or overwrite its results.

```bash
export RETRACE_STORED_OUTPUT="$RETRACE_ROOT/output/retrace_stored_full_$(date +%Y%m%d_%H%M%S)"
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_stored_vllm.sh"
```

Defaults: 40K unique stored prompt/first-response examples, prompt at most 512
tokens, stored response at most 1,024 tokens, eight epochs, AdamW LR 5e-5,
weight decay 0.01, cosine decay, 5% LR warmup, seed 42. Target features and
draft forward math use BF16; trainable parameters are held in FP32. The stock
DFlash checkpointer saves BF16 weights/optimizer tensors, so resume is not a
bitwise FP32-state continuation. Normal inference checkpoint loading is retained.

The official multipack sampler balances token budgets and can drop a final
incomplete distributed batch. An epoch means a pass through that sampler; it
does not mean exactly 1,250 updates as in the previous 32-prompt implementation.
The launcher prints the actual updates per epoch. The branch budget is an
experimental speed/coverage setting, not an author-released sampler setting.

Your paths are the defaults. Optional overrides:

```bash
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_stored_vllm.sh" \
  --server-npus 0,1 --trainer-npus 2,3,4,5,6,7 \
  --pairs-per-batch 32 --request-concurrency 8 --blocks-per-forward 4
```

Increasing pairs gives more conditioning examples and costs more branch prefills.
Increasing `--tokens-per-worker` changes packing/batch size and the optimization
schedule. Increasing `--blocks-per-forward` uses more activation/logit memory.
None of these is a guaranteed speed improvement.

An optional `--prompt-pool OLD_RUN/prompts.jsonl` selects that pool's source row
IDs. It verifies the stored prefix agrees exactly, including any empty-think
prefix already present in the stored response. It refuses mismatches; it never
inserts tokens or regenerates the response. Some source rows may have responses
too short for a round pair, requiring a lower `--max-records` for that pool.

## Checkpoints, storage and timing

Checkpoints are stored at:

```text
$RETRACE_STORED_OUTPUT/checkpoints/0/  # completed epoch 1
$RETRACE_STORED_OUTPUT/checkpoints/1/  # completed epoch 2
...
$RETRACE_STORED_OUTPUT/checkpoints/7/  # completed epoch 8
```

Each directory includes model, optimizer, scheduler and `training_state.json`.
The standard trainer adds `epoch0_end`, `epoch1_end`, etc. symlinks. Resume only
this new protocol, with its original configuration:

```bash
bash "$RETRACE_STORED_OUTPUT/resume_stored.sh"
```

Default clean-feature handling is DFlash's `on_generate=delete`. vLLM computes
the frozen clean features in one prefill when loading a row; the token response
itself stays fixed. With `--cache-clean`, those features are retained for later
epochs. The launcher prints a dataset-specific disk estimate and checks free
space; 40K full hidden-state sequences can occupy hundreds of GiB. Proposed
branch states are never persisted as a cross-update cache.

Timing is recorded in `checkpoints/epoch_timing.jsonl`. No NPU speedup has been
measured for this implementation. Fewer sequential rounds and shared vLLM
batching remove the old execution pattern, but branch prefills still repeat
prefix computation and transfer hidden-state files. Prefix caching and chunked
prefill are disabled to ensure each request exports its full feature sequence.
This is not a zero-copy or incremental branch-KV implementation. Compare time
per epoch, row coverage and conditioning coverage, not just seconds/update,
because update sizes and the training sampler have changed.

## Files and validation

Added under `speculators/`:

- `src/speculators/models/retrace/stored_data.py`
- `src/speculators/models/retrace/stored_target.py`
- `src/speculators/models/retrace/stored_training.py`
- `src/speculators/models/retrace/stored_train.py`
- `src/speculators/models/retrace/stored_launch.py`
- `examples/train/train_retrace_dflash_stored_vllm.sh`
- `examples/train/README_retrace_dflash_stored_vllm.md`
- `tests/unit/models/test_retrace_stored.py`

Updated: `src/speculators/models/retrace/config.py` accepts `stored_pair_kl`.
The original DFlash trainer/model files are imported, not modified. Importing
the training adapter registers a separate training-only alias; the runtime
`retrace` registration and exported model architecture remain unchanged.

Tests cover response preservation, masked/turn boundaries, causal scoring
alignment, first-rejection removal, prefix disagreement, detached memory,
conditioning/backbone gradients, frozen target tensors, stock KL equivalence,
document separation, transfer cleanup, runtime checkpoint reload, and the
actual stock data loader/Trainer epoch save/resume path on a tiny CPU model.
Hardware execution, full training convergence and Table 2 results remain
unvalidated. No speed or quality improvement is promised before measurement.
