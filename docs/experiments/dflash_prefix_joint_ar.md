# Joint AR training for DFlash Prefix Attention

This experiment adds a training-only, causal next-token objective to the shared
DFlash transformer used by `train_dflash_prefix_joint.sh`. It keeps the existing
`compiled_prefix_v3` selector, frozen target, checkpoint tensor names and serving
path. It is an adaptation of joint AR/diffusion training to a small conditional
drafter, not a reproduction of the full Nemotron tri-mode language model.

## Motivation and limits

[Nemotron-Labs-Diffusion, Sections 2.1–2.4](https://arxiv.org/html/2607.05722v1)
distinguishes stronger AR initialization from retaining an AR objective during
diffusion training. It uses `L_AR + 0.3 L_diff`, with causal clean-stream attention
and blockwise noisy-stream attention. Its language-model accuracy gains do not
establish acceptance-length or throughput gains for a separate DFlash drafter.

This repository's selector already conditions on earlier reference tokens during
training. The new objective is different: it directly trains the **shared draft
transformer, context projection and final normalization** as an AR predictor.
The frozen verifier, token embeddings and full-vocabulary output projection stay
frozen. The AR pass does not train a separate auxiliary head that would be thrown
away at inference.

With correct speculative verification, the intended benefit is better agreement
with the fixed target and higher serving throughput. It does not increase the
fixed target's task accuracy. AR CE alone is not evidence of this benefit.

## Objective and alignment

During the joint phase:

```text
loss = existing_parallel_loss
     + scheduled_selector_weight * existing_selector_loss
     + AR_LOSS_WEIGHT * causal_reference_next_token_CE
```

The original parallel loss and its configured weighting are retained. The AR
coefficient multiplies AR, unlike the paper's coefficient on diffusion; `0.1`
below is a starting experiment, not a transferred or validated optimum.

For an anchor at position `a` and block size `B`:

| Path | Input queries | Labels | Query attention | Target feature access |
| --- | --- | --- | --- | --- |
| Existing parallel drafter | `x[a], MASK, ..., MASK` | Existing target-derived labels at proposal slots | Existing DFlash block mask | Same-document positions strictly before `a` |
| New AR auxiliary pass | `x[a], ..., x[a+B-2]` | `x[a+1], ..., x[a+B-1]` | Causal, within the same sampled block | Same-document positions strictly before `a` |

The AR labels are reference tokens in the training data. They need not equal the
target's argmax when the stored rollouts were sampled. `ar_reference_acc` and the
existing `unary_full_acc` therefore measure different agreements.

The clean stream never enters the masked parallel stream. Causality is enforced
in every draft layer, including full-attention layers. A strict attention mask
without shifted labels would be insufficient: a token can leak through its own
embedding and residual connection. Cached target features at or after the anchor
are also forbidden, even during AR teacher forcing. Packed-document boundaries,
padding and the assistant loss mask gate the AR labels.

The AR numerator and valid-token count are pooled across DDP ranks. Local losses
are scaled by world size before DDP averages gradients, so ranks with different
numbers of valid labels do not receive equal weight by accident. Empty ranks
still participate; an entirely empty AR batch contributes a connected zero.
Validation records local sums/counts for the existing metric reducer and needs
no extra per-batch collective.

## Settings

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `AR_LOSS_WEIGHT` | `0` | AR coefficient in the joint phase; zero preserves the old training path |
| `AR_PRETRAIN_STEPS` | `0` | Initial optimizer steps with only the AR objective; requires positive AR weight |
| `AR_MAX_ANCHORS` | `64` | Cap on auxiliary anchors per rank, drawn evenly from already sampled anchor slots |
| `AR_LOGIT_CHUNK_SIZE` | `128` | Number of predictor positions projected to vocabulary per chunk |
| `AR_LOGIT_CHECKPOINTING` | `0` with launcher fast profile; `1` with memory profile | Recompute those projection/loss chunks during backward to reduce memory |

AR pretraining here means warming up this same conditional drafter from its
normal random initialization. It does not copy a pretrained Qwen or Nemotron
transformer's layers into DFlash. Such a weight-transfer experiment needs a
separate mapping and ablation because the context injection and layer layout
are different. Existing target embeddings and output weights are still loaded
by the original code.

For the first `AR_PRETRAIN_STEPS` updates, the AR weight is `1` and both other
loss weights are `0`. The existing parallel/selector computations are still
performed for diagnostics and consistent parameter participation in DDP; warmup
is **not** optimized as a cheaper AR-only execution path. Selector optimizer
updates are delayed, so its weight decay also stays inactive. The configured
selector/guard/focus start steps are relative to the end of AR pretraining and
are saved as absolute steps. Warmup must leave room for joint training within
the existing epoch budget.

Two passes through shared layers require non-reentrant decoder checkpointing
with DDP. The training subclass selects it when AR is enabled and checkpointing
is requested. The original execution controls continue to apply; set
`BACKBONE_CHECKPOINTING=0 SELECTOR_CHECKPOINTING=0 AR_LOGIT_CHECKPOINTING=0`
only when activation memory allows.

The launcher now defaults to the [fast execution profile](dflash_prefix_fast_training.md),
which disables all three recomputation paths while retaining the training recipe.
Use `EXECUTION_PROFILE=memory` for the previous defaults; explicit environment
controls override either profile. Resume requires `RESUME_EXECUTION_PROFILE=fast`
to switch a saved run's execution settings.

No extra verifier forward is required. Training takes an additional drafter
pass plus chunked vocabulary projections. The exact time and memory overhead
must be measured on the full A2 workload. Increasing `AR_MAX_ANCHORS` increases
that cost. Inference performs no AR auxiliary pass and loads no new parameters.

## Running on the existing A2 setup

Keep the model, dataset and base recipe used by the matched baseline. The
launcher already defaults to two vLLM NPUs and six trainer NPUs and uses DDP.
This change does not add FSDP2 support. All eight NPUs are used by those defaults.

From `/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/speculators`, first
exercise the small single-NPU/distributed backward and saved-checkpoint checks:

```bash
AR_LOSS_WEIGHT=0.1 AR_PRETRAIN_STEPS=0 AR_MAX_ANCHORS=64 \
PREFLIGHT_ONLY=1 \
bash examples/train/train_dflash_prefix_joint.sh
```

The launcher creates a fresh timestamped output directory by default. After the
preflight passes, run joint training:

```bash
AR_LOSS_WEIGHT=0.1 AR_PRETRAIN_STEPS=0 AR_MAX_ANCHORS=64 \
EPOCHS=5 \
bash examples/train/train_dflash_prefix_joint.sh
```

An optional separate experiment adds AR warmup inside the same epoch budget:

```bash
AR_LOSS_WEIGHT=0.1 AR_PRETRAIN_STEPS=1000 AR_MAX_ANCHORS=64 \
EPOCHS=5 \
bash examples/train/train_dflash_prefix_joint.sh
```

For Qwen3-8B, set `MODEL` and `DATASET` to the corresponding local paths and use
a matching `BASE_TRAIN_CONFIG`, including its target layer IDs and architecture.
Changing only the target path does not establish a matched 8B recipe.

The preparation step records every AR setting in `joint_settings.json` and
`experiment.json`; `resolved_schedule.json` records them with optimizer-step
counts. Existing command, source-diff and target-server provenance remain active.
Continuation copies AR settings from the source run and restores `global_step`,
so warmup is not restarted. `AR_*` environment overrides **do not change an
existing run's objective** when using `--resume-training`. Start a fresh
experiment to compare a different objective. Old metadata without AR settings
continues to mean AR disabled.

Checkpoints still export as `DFlashPrefixDraftModel`. Use the launcher's printed
serve command and the usual evaluator to measure actual acceptance and speed.

## Evaluation

Run at least these matched experiments before drawing a conclusion:

| Experiment | AR weight | AR pretrain steps |
| --- | --- | --- |
| Existing Prefix Attention baseline | `0` | `0` |
| Joint AR + parallel | `0.05`, `0.1`, `0.3` | `0` |
| AR warmup followed by joint training | best held-out weight | `1000` initially |

Choose weights and warmup on held-out validation, then report final serving
results on untouched evaluation prompts. Compare both equal data/optimizer
budgets and equal training-time budgets: warmup consumes steps and joint AR
adds compute. Keep target, dataset split, seeds, batch budget, block size,
selector/guard settings, decoding configuration and hardware fixed. A second
seed helps distinguish a small gain from training variation.

Track existing `unary_*`, shortlist target recall, selector rescue/damage,
saved-weight reference EAL, and the new `ar_ce`, `ar_reference_acc`, `ar_weight`,
`ar_pretraining`, `parallel_weight`. Reference EAL is an offline proxy. Report
real verified acceptance length, tokens/second, draft latency, verification
latency and peak memory at the same temperatures and concurrency. Do not use
teacher-forced AR accuracy as a substitute for free-running parallel drafting.

If AR CE improves while parallel shortlist recall or verified acceptance falls,
reduce the AR weight or reject the auxiliary objective. The useful claim is
better acceptance or throughput at the same inference architecture and budget;
the shared AR objective itself is already established prior work.

## Correctness checks

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:src:hs_connectors/src \
python -m pytest tests/unit/models/test_dflash_prefix_ar.py tests/joint_prefix
```

The tests cover shifted labels, future-token and target-feature leakage across
multiple layers, SDPA/eager masks, document boundaries, token normalization,
empty masks, gradient flow, the disabled path, checkpointing, warmup transitions,
and actual DDP reducer hooks using PyTorch's in-memory test backend, with and
without recomputation. Set `RUN_JOINT_DDP_TEST=1` to also run the two-process
Gloo tests on a host that permits local sockets. The in-memory backend is not
an HCCL or hardware test; the NPU preflight is required before a full A2 run.
None of these synthetic checks establishes an acceptance-length or speed gain.
