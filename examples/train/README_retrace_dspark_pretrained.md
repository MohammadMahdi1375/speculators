# Pretrained DSpark-block7 + ReTrace on NPUs 8–15

This package fine-tunes **DSpark + ReTrace** from your local checkpoint:

`/home/n84449292/m84379596/Huggingface/Qwen3-4B-DSpark-block7/`

The importer recognizes the public `deepseek-ai/dspark_qwen3_4b_block7` format. It preserves the five-layer DSpark backbone, rank-256 vanilla Markov head and confidence head. Only the three ReTrace conditioning matrices are new. The residual projection starts at zero. There is no randomly initialized drafter fallback.

This is an extension of ReTrace to DSpark. DSpark is not the DFlash backbone used in the ReTrace Table 2 experiment. This training package does not claim to reproduce those Table 2 numbers or to have measured an improvement over DSpark.

## Install

Use a **new terminal** on the same server, leaving your DFlash training process running. Put `install_retrace_dspark_pretrained.py` in:

`/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/`

The installer embeds all required new source files and the Bash launcher. You do not need to download model weights again or unpack an archive.

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_ROOT="$PWD"
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python

"$RETRACE_PYTHON" install_retrace_dspark_pretrained.py --root "$RETRACE_ROOT" --apply
```

Omit `--apply` for a read-only preview. Installation checks speculators commit `e376ed8724f42e1c7a01df9f3ef870600ffb03e5`, checks the stock model dependencies, rejects unknown source edits, and backs up replaced DSpark-ReTrace files. It supports either an absent `retrace_dspark` package or the previously supplied isolated DSpark implementation. Re-running an already applied installer is a no-op.

The installed launcher is:

`speculators/examples/train/retrace_dspark_qwen3_4b_pretrained.sh`

The installer does not change stock `models/dspark`, stock `models/dflash`, your running DFlash `models/retrace` code, vLLM, or vLLM-Ascend. Training uses a local frozen Hugging Face target on each worker and needs no target server. It uses the existing environment without pip installations. Python package registration happens when the independent `retrace_dspark` package is imported.

## First run: three-update NPU smoke training

```bash
cd "$RETRACE_ROOT"
export RETRACE_DSPARK_OUTPUT="$RETRACE_ROOT/output/retrace_dspark_pretrained_smoke_$(date +%Y%m%d_%H%M%S)"

bash speculators/examples/train/retrace_dspark_qwen3_4b_pretrained.sh --smoke
```

Defaults select **physical visibility IDs `8,9,10,11,12,13,14,15`** before Python is started. The eight worker-local devices are named `npu:0` through `npu:7` inside that visibility mask. Every worker prints its requested mapping. No fallback to physical IDs 0–7 is implemented. The smoke run uses 24 prompts, eight prompts per global update, a 256-token prompt limit, a 64-token response limit, and three updates (one short epoch).

Expected successful endpoint:

```text
.../checkpoints/step_000003
Training finished. Checkpoint pointer: .../checkpoints/latest.txt
```

Before workers start, `initial/import_report.json` records source SHA-256 hashes, the source tensor shapes, imported head names, exact tensor equality after dtype conversion, and zero residual initialization. Any missing pretrained backbone/head tensor or mismatched shared embedding/head fails the import instead of silently replacing it.

The NPU device group must be exposed to this shell by your server/container allocation. A visibility error requires correcting that allocation, rather than changing the requested group to 0–7 while DFlash is using it.

## Full training

Start a **new output directory** from the original DSpark checkpoint after the smoke succeeds. The smoke checkpoint is a functional test, not the starting point of the full run.

```bash
export RETRACE_DSPARK_OUTPUT="$RETRACE_ROOT/output/retrace_dspark_pretrained_full_$(date +%Y%m%d_%H%M%S)"

bash "$RETRACE_ROOT/speculators/examples/train/retrace_dspark_qwen3_4b_pretrained.sh"
```

| Setting | Default |
|---|---|
| Frozen target | Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c` |
| Pretrained drafter | Local `Qwen3-4B-DSpark-block7` |
| Physical NPU visibility | `8,9,10,11,12,13,14,15` |
| Worker layout | Eight replicated draft trainers, each with its own frozen target |
| Dataset | Your local `open_perfectblend.qwen3-4b-rollout.qwen3.seq3072` Arrow dataset |
| Prompt pool | 40,000 distinct prompts, deterministic row permutation, seed 42 |
| Thinking mode | Disabled using the empty-think suffix |
| Prompt / new-response limits | 512 / 1024 tokens |
| Draft block | Seven proposals; all seven DSpark outputs are used |
| Global batch | 32 complete prompt trajectories per optimizer update |
| Epochs / total updates | Eight epochs / 10,000 updates |
| Peak learning rate | `5e-5`, 5% linear warmup, cosine decay |
| AdamW | Weight decay 0.01, gradient clipping at 1.0 |
| Forward / master precision | BF16 / FP32 |
| Rejected memory | Detached FP16, one preceding verification round |
| ReTrace correction warmup | Beta ramps from 0 to 1 over 50 updates |
| Training block chunk | Two blocks per gradient forward |
| Checkpoint schedule | End of every epoch, and on normal completion/planned pause |

The Bash script sources both of your CANN 9.1.0 environment scripts. DSpark-specific variables (`RETRACE_DSPARK_BASE`, `RETRACE_DSPARK_NPUS`, `RETRACE_DSPARK_OUTPUT`) prevent inherited DFlash checkpoint/output settings from changing this run. Its torchrun rendezvous uses an independent random local port. Source imports, prompt selection and hash checks happen on CPU before worker launch.

The prepared prompt pool is deterministic with the same settings as the DFlash recipe. To use exactly an existing DFlash **full-run** pool, add `--pool /absolute/path/to/prompts.jsonl`; its `.manifest.json` must also exist and match the 40,000 / 512 settings. Dataset responses are not copied as training continuations: every prompt gets a fresh target-verified rollout with the current drafter.

Logs are written to `$RETRACE_DSPARK_OUTPUT/trainer.log`. Structured metrics are in `checkpoints/metrics.jsonl`; startup/command/source/version provenance is saved with the run. The frozen target is never optimized.

## Epoch checkpoints and resume

For 40,000 prompts and global batch 32, each epoch has **1,250 optimizer updates**. Checkpoints are:

```text
$RETRACE_DSPARK_OUTPUT/checkpoints/step_001250   # epoch 1
$RETRACE_DSPARK_OUTPUT/checkpoints/step_002500   # epoch 2
$RETRACE_DSPARK_OUTPUT/checkpoints/step_003750   # epoch 3
$RETRACE_DSPARK_OUTPUT/checkpoints/step_005000   # epoch 4
$RETRACE_DSPARK_OUTPUT/checkpoints/step_006250   # epoch 5
$RETRACE_DSPARK_OUTPUT/checkpoints/step_007500   # epoch 6
$RETRACE_DSPARK_OUTPUT/checkpoints/step_008750   # epoch 7
$RETRACE_DSPARK_OUTPUT/checkpoints/step_010000   # epoch 8
```

`checkpoints/latest.txt` contains the latest complete checkpoint path. Each checkpoint contains the exported model/config, FP32 master parameters, AdamW state, per-rank RNG state and training progress. All epoch checkpoints are retained; the frozen 4B target is not copied into each checkpoint. Writes use a staging directory and update the latest pointer only after every rank finishes saving.

To resume a stopped run, preserve its output path, prompt pool, world size and training settings:

```bash
export RETRACE_DSPARK_OUTPUT=/absolute/path/to/your/retrace_dspark_pretrained_full_run
export RETRACE_DSPARK_RESUME="$(cat "$RETRACE_DSPARK_OUTPUT/checkpoints/latest.txt")"

bash "$RETRACE_ROOT/speculators/examples/train/retrace_dspark_qwen3_4b_pretrained.sh" \
  --resume "$RETRACE_DSPARK_RESUME"
```

`--stop-after N` saves a planned pause without changing the full learning-rate schedule. `--max-steps N` changes the planned training length and schedule; use it only deliberately. Ctrl-C, process termination or hardware failure does **not** guarantee an immediate checkpoint.

If metrics were written after the most recent saved epoch before an unplanned interruption, use a new output directory and the old prompt pool so existing logs are preserved:

```bash
export RETRACE_DSPARK_OLD=/absolute/path/to/old/full_run
export RETRACE_DSPARK_RESUME="$(cat "$RETRACE_DSPARK_OLD/checkpoints/latest.txt")"
export RETRACE_DSPARK_OUTPUT="$RETRACE_ROOT/output/retrace_dspark_resume_$(date +%Y%m%d_%H%M%S)"

bash "$RETRACE_ROOT/speculators/examples/train/retrace_dspark_qwen3_4b_pretrained.sh" \
  --resume "$RETRACE_DSPARK_RESUME" \
  --pool "$RETRACE_DSPARK_OLD/prompts.jsonl"
```

## Implementation and scope

`ReTraceDSparkDraftModel` subclasses the existing **DSparkDraftModel**. DSpark itself shares backbone components with DFlash upstream; this package still loads and uses DSpark's Markov and confidence heads and its seven-proposal convention.

- Public DeepSpec target decoder-layer IDs `[1,9,17,25,33]` are converted to Hugging Face hidden-state indices `[2,10,18,26,34]`. Already converted speculators DSpark checkpoints retain their existing indices.
- Each live round drafts seven tokens, verifies the actual proposed trajectory, commits the accepted prefix and target correction/bonus, and discards the accepted prefix plus the first rejection from the remembered suffix.
- Detached draft and target scoring states from the remaining rejected suffix are aligned to the next round. Target correction and gated residual fusion condition the masked inputs. The verified anchor input remains unchanged. DSpark output slot `j` predicts token `p+j+1`; its memory is aligned by prediction position, including the deliberate omission of memory slot zero from anchor conditioning.
- Training collects a complete target-verified continuation with fixed drafter weights, then replays actual round boundaries for gradients. The Markov head receives clean previous-token IDs. Draft attention sees context strictly before each anchor and its own query block; it cannot see clean future hidden states.
- The objective is DSpark's **0.1 hard-token CE + 0.9 total-variation loss + confidence BCE**, all with exponential position decay `exp(-j/4)`. Confidence targets are detached distributional overlaps. TV/confidence labels use the clean continuation's frozen target distributions, not logits conditioned on rejected prefixes after the first rejection. This is a documented DSpark adaptation, not an assertion that the ReTrace paper used this exact loss or anchor sampler.
- Full backbone, Markov/confidence heads and new conditioning matrices are fine-tuned. Shared target embeddings/output head and target model stay frozen. Complete-prompt gradients are summed across ranks and divided by the actual global prompt count, including a partial final batch.

Native vLLM/Ascend inference evaluation is separate and is not installed or validated by this training update. The initial unchanged DSpark baseline export is saved under `initial/dspark` for later comparison. No speedup or acceptance improvement has been measured for this new run.

## Installed file inventory

All model files below are under `speculators/src/speculators/models/retrace_dspark/`:

| File | Purpose |
|---|---|
| `__init__.py` | Independent DSpark-ReTrace registration |
| `config.py` | Separate architecture identity, DSpark heads and pretrained provenance |
| `core.py` | DSpark proposal chain, ReTrace fusion and causal multi-block replay |
| `memory.py` | Rejected suffix alignment, detached FP16 storage and gated fusion |
| `target.py` | Frozen target state extraction and cache rollback |
| `runtime.py` | Model loading, tied Qwen3 weights, device placement and target setup |
| `pretrained.py` | Strict local DeepSpec/speculators checkpoint importer and baseline export |
| `prompt_pool.py` | Deterministic Arrow prompt extraction and pool manifest |
| `clean_training.py` | Live rollouts, clean labels, Markov teacher forcing and DSpark objective |
| `train_pretrained.py` | Distributed complete-prompt training, epoch checkpoints and exact resume |
| `pretrained_launch.py` | CPU preparation, dedicated NPU group, independent process lifecycle and logs |

Also installed:

- `speculators/examples/train/retrace_dspark_qwen3_4b_pretrained.sh`
- `speculators/examples/train/README_retrace_dspark_pretrained.md`
- `speculators/tests/unit/models/test_retrace_dspark_pretrained.py`

This installer can supply the training package without any earlier DSpark bundle. No changes to global speculators registration files are required for the module launch path. Previously installed DSpark-ReTrace evaluation files, if any, are retained.

## Sources and validation

Architecture/weight-name and layer-index checks used the [public DSpark checkpoint configuration](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/blob/main/config.json) and [DeepSpec's Qwen3 DSpark model](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py), [feature extraction](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py), and [draft operations](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py). The source commit is pinned in this document and the bundle manifest.

Local validation uses a tiny Qwen3 target and a synthetic checkpoint with the official DSpark field/tensor layout. It checks import equality, missing-head rejection, shared target tensors, zero-residual identity, causal block replay, clean Markov inputs, gradient flow, loss equivalence with stock DSpark, frozen target weights, epoch coverage, and BF16-checkpoint/FP32-master exact resume. Launcher tests check NPU visibility and independence from inherited DFlash variables. These tests do not substitute for loading your real checkpoint and running HCCL training on your Ascend server; use the provided smoke command first.

The optional two-worker Gloo integration test could not run in the development workspace: socket creation returned `Operation not permitted`, including with loopback selected. It is not counted as a passed check. To run that test on a host allowing local distributed sockets, set `RETRACE_TEST_DISTRIBUTED=1` when running the test module. Ascend/HCCL execution and real-checkpoint loading are pending the server smoke run.

Final local validation: **17 checks passed, one optional distributed test skipped** when running against a fresh pinned speculators checkout populated by this installer. Separate installer checks passed for clean installation, dry-run behavior, repeated installation, known previous-version upgrade, backup contents, unknown-edit refusal, dependency guards and preservation of tracked stock sources. Python syntax and Bash syntax checks also passed.
