# ReTrace with a vLLM target and Speculators draft training

The corrected `train_retrace_dflash_paper.sh` uses the existing Speculators
hidden-state extraction server. Its defaults are:

| Setting | Value |
| --- | --- |
| Frozen target | Local Qwen3-4B snapshot, served by vLLM |
| Target devices | Logical NPUs 8,9; DP=2, TP=1 |
| Trainable model | Pretrained Qwen3-4B-DFlash-b16 plus ReTrace |
| Trainer devices | Logical NPUs 10,11,12,13,14,15; six Speculators workers |
| Target endpoint | http://127.0.0.1:8523/v1 |
| Prompt pool | 40,000 distinct prompts from the specified local Arrow dataset |
| Epochs | 8 |
| Global prompt batch | 32, independent of the number of workers |
| Lengths | Up to 512 prompt tokens and 1,024 generated tokens |
| Optimizer | Existing recipe: AdamW, LR 5e-5, weight decay 0.01 |
| Saving | Each epoch end; also normal completion or `--stop-after` |

The old aggregate `RETRACE_NPUS` variable does not override the split. If needed,
use `RETRACE_SERVER_NPUS` and `RETRACE_TRAINER_NPUS`. The parent may see all eight
devices, but each child receives its own visibility mask. Inside a child,
`npu:0` refers to the first device in that child's mask.

## Start

Use your existing environment; run one supervised launcher. It starts the
target server, waits for health, starts the trainer, and cleans up only the
process groups it started.

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_paper.sh" --smoke
```

The smoke run uses 12 prompts, lengths 256/64, batch 4 and three updates. After
it succeeds, launch the full recipe:

```bash
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_paper.sh"
```

Each command chooses a new output directory. You can set `RETRACE_OUTPUT` to a
new path to name it. Full checkpoints are normally at:

```
$RETRACE_OUTPUT/checkpoints/step_001250  # epoch 1
$RETRACE_OUTPUT/checkpoints/step_002500  # epoch 2
...
$RETRACE_OUTPUT/checkpoints/step_010000  # epoch 8
```

`checkpoints/latest.txt` points only to a completed save. `trainer.log` and
`vllm_server.log` are in the run root; metrics are in `checkpoints/metrics.jsonl`.

## Continue a saved local-target paper_recipe run

First let the existing job finish a checkpoint and stop that specific job.
Installing the update cannot make an already running process save its in-memory
updates. Ctrl-C does not create an emergency checkpoint.

```bash
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_paper.sh" \
  --resume-run /absolute/path/to/your/existing/paper_recipe/run \
  --migrate-to-vllm
```

The new run reuses the exact prompt pool and loads the last completed checkpoint.
It restores FP32 master weights, optimizer state, global step and the retained
workers' RNG state. With 8 to 6 workers, rank RNG files 0 through 5 are restored;
the retired ranks' streams are not reused. Worker assignment and vLLM rounding
can change trajectories, so this is not a bitwise-equivalent continuation.

The migration accepts only a local-to-vLLM backend change, a non-increasing
worker count and reference target normalization. Dataset identity, global batch,
loss, learning rate, epochs and total steps must still match. Other explicitly
requested execution changes require `--allow-performance-change`. The original
run is preserved and compatibility is checked before starting a server.

For a later continuation of an existing vLLM run, use `--resume-run` without
`--migrate-to-vllm`. A three-step smoke checkpoint has a different training
schedule and cannot resume the full recipe.

## What this update changes and what is measured

This update changes target placement, shared-feature dtype handling and explicit
checkpoint migration. The recurrent rejected-suffix conditioning and clean CE
objective remain the existing paper_recipe implementation. It does not switch
back to the earlier stored-pair approximation. Frozen shared embeddings, output
head and final norm stay in each trainer for draft inputs and teacher scoring;
the 36-layer target transformer runs in vLLM.

This transport sends the actual full proposal prefix each verification round,
exports hidden states through the existing file connector, validates token/layer
alignment and removes transferred files after reading. Six trainers can submit
requests concurrently. It does not reuse incremental target KV between these
requests. It can therefore be slower than a local cached target despite using
vLLM. Moving inference alone is not an established speed optimization, and this
update makes no Table 2 reproduction or throughput claim. A high-performance
incremental verification/rollback transport is a separate implementation task.

The tiny smoke checks startup and transfer, not full-length throughput. To
measure actual workload cost without changing the full schedule, launch a new
full run with `--stop-after 3`; it saves and pauses after three full-size updates.
Continue from that run using `--resume-run`. Do not treat a short timing sample
as a guaranteed eight-epoch runtime.

CPU validation covers child device masks, launcher defaults, invalid topology,
checkpoint migration restrictions, real HTTP completions/file-transfer contract,
BF16 feature handling with FP32 masters, and actual tiny-model checkpoint/resume.
NPU/vLLM model execution has not been validated in the development workspace.
