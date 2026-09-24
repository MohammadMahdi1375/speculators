# Qwen3-4B ReTrace audit and corrected experimental recipe

This package corrects identifiable differences in the previously supplied implementation. It is **not a certified reproduction of Table 2**. The author training code, selected prompt IDs and full counter implementation were not available for this audit. Native Ascend validation requires your machine.

## Findings

| Area | Previously supplied route | This package |
|---|---|---|
| Training examples | Stored clean responses, sampled source/successor pairs; every source starts without memory; successors disagreeing with the stored prefix are omitted | Prompt-only, actual draft/verification trajectories; recurrent one-round memory, with no stored-response alignment filter |
| Training objective | The recovered `stored_training.py` selects `kl_div` | DFlash-style hard-token cross-entropy, exponentially weighted by distance; no auxiliary loss |
| Effective batch | Token-packed batches with sampled pairs | 32 complete prompt trajectories per optimizer update |
| Stochastic draft proposals | Server scripts force greedy proposals at temperature 1 | Temperature, top-k, top-p sampling; exact proposal probabilities exported to the existing rejection sampler |
| Acceptance reporting | Charts use a mean of request-level acceptance lengths including a bonus | Report draft-only cycle pooling, cycle pooling with bonus, and request means separately |

These training findings describe the recovered delivered files, not a read of your remote filesystem. The installer records hashes and recognizable settings from your installed sources in its backup. It preserves your existing stored trainer, checkpoints, evaluator, baseline DFlash and DSpark code.

The existing memory code already excludes the accepted prefix and first rejected token, aligns draft states with the target states that scored them, detaches the retained suffix, and leaves the anchor untouched. The audit did not find an obvious index shift in that path. This does not prove that every deployed server used the intended checkpoint or conditioning path.

## Important interpretation of your results

Your request counts match Appendix E: GSM8K/MATH500/LCB/Alpaca 128 each, HumanEval 164, AIME25 30, MT-Bench 80. Do not increase or replace subsets merely to obtain higher acceptance.

Your temperature-0 macro request mean is about **6.43**, but your macro mean of cycle-pooled acceptance **including a nominal bonus** is about **6.79**. These are different statistics. The paper's Table 2 reports 6.85; its text describes accepted draft tokens averaged over verification cycles. Public DFlash code also exposes emitted-token counts including the bonus, with end-of-generation clipping. Without the ReTrace counter code, bonus convention and clipping remain unresolved. This package does not select whichever convention happens to look closest.

At temperature 1, the forced-greedy draft is a concrete proposal-distribution difference. Replacing it may change acceptance; improvement is not guaranteed. Greedy versus probabilistic draft sampling does not explain temperature-0 differences.

Your CSV contains blank `greedy_parity`, `comparison_valid` and speedup fields. `measurement_valid=True` validates metric collection, not output parity or speedup relative to the target. Re-run the official DFlash checkpoint and a target-only reference with the same prompt manifest, settings and hardware before making a reproduction claim. Keep length-limited requests in the report. In particular, 15/30 AIME25 requests hit the output limit in your T=0 run.

## Install

Copy `install_retrace_table2_corrections.py` to your project root. In your existing activated environment:

```bash
export RETRACE_ROOT=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export RETRACE_PYTHON=/home/n84449292/m84379596/conda/vllm-dflash2-main/bin/python
cd "$RETRACE_ROOT"
"$RETRACE_PYTHON" install_retrace_table2_corrections.py --root "$RETRACE_ROOT" --apply
```

The installer reviews all destinations before writing, backs up its single edit to the native sampling method, and refuses unrecognized sampling support or conflicting new files. Omitting `--apply` gives a dry run. Installation is idempotent. It does not reset repositories, edit model weights or stop running jobs.

## First: evaluate the current checkpoints with matched sampling

Run these **sequentially on the same idle logical NPU** for timing comparisons. Stop the first server before starting the second. They are independent of `evaluator.py`.

```bash
cd "$RETRACE_ROOT/Evaluator"
DEVICES=0 bash run_server_dflash_table2.sh
```

The official server defaults to port 8200. Connect your existing evaluator to `http://127.0.0.1:8200` and reuse your existing `retrace_eval_prompts.json`.

```bash
DEVICES=0 bash run_server_retrace_dflash_table2.sh
```

The ReTrace server defaults to port 8201 and the checkpoint `output/retrace_dflash_qwen3_4b_8epoch_fullvocab/checkpoints/7` from your latest logs. Override `DRAFT=/absolute/checkpoint` if that is not the checkpoint used for your CSV. The server prints the exact path and config hash.

Both servers use b16 = **one anchor plus 15 proposals**, TP=1, batch=1, BF16, eager/synchronous execution. At T=0 proposals remain greedy. At T=1 the helper applies top-k/top-p to the draft and returns the exact full-vocabulary q to vLLM rejection sampling. It changes neither target logits nor acceptance rules. The opt-in environment variable affects only DFlash, including the ReTrace subclass; existing scripts and DSpark retain their behavior.

For the previously delivered evaluator:

```bash
"$RETRACE_PYTHON" evaluator.py \
  --base-url http://127.0.0.1:8201 \
  --manifest "$RETRACE_ROOT/Evaluator/retrace_eval_prompts.json" \
  --temperatures 0 1 --max-new-tokens 8192 --seed 42 \
  --output "$RETRACE_ROOT/output/retrace_matched_sampling_eval"
```

It already applies no-thinking prompts, T=0/top-p=1/top-k=1 and T=1/top-p=.95/top-k=20. Do not change the existing frozen prompt manifest between methods. The paper does not publish selected row IDs; the delivered evaluator uses seeded selection and first-turn MT-Bench prompts, so exact prompt/turn matching remains unverified.

To inspect all conventions in a CSV:

```bash
"$RETRACE_PYTHON" compare_retrace_table2.py --csv /absolute/path/to/results.csv
```

## Corrected training path

The new script is in `speculators/examples/train/train_retrace_dflash_paper.sh`. It uses a separate `speculators.models.retrace.paper_recipe` module tree; it does not silently turn your fast stored run into a different algorithm.

Defaults:

| Setting | Value |
|---|---|
| Initialization | `/home/n84449292/m84379596/Huggingface/Qwen3-4B-DFlash-b16` |
| Target | Your existing Qwen3-4B snapshot, frozen |
| Data | 40,000 distinct prompts extracted from your local Arrow dataset, seed 42 |
| Prompt / response limits | 512 / 1024 tokens |
| Epochs / global prompt batch | 8 / 32: 1,250 updates per epoch, 10,000 total |
| Optimizer | AdamW, LR 5e-5, weight decay .01, cosine decay, 5% LR warmup |
| ReTrace correction strength | beta ramps from 0 to 1 over 50 updates |
| Precision | BF16 forward, FP32 trainable master parameters and AdamW states; detached FP16 memory |
| Objective | Clean committed-token CE, exp(-distance/4) weights, valid-token denominator per prompt |
| Trainable parameters | Pretrained draft backbone and correction/value/gate matrices; target and shared output layers frozen |
| Saving | Every epoch; also at explicit `--stop-after` pauses |
| Hardware default | Logical NPUs 0–7; local frozen target per worker |

The published settings are in Appendix B. Our greedy training rollouts, live-round anchor selection, prompt deduplication/overlength filtering, and per-prompt reduction are explicit implementation choices where exact author code is missing. The paper uses FSDP on 32 A800 GPUs; this Ascend port uses replicated parameters and sums/averages prompt gradients. Do not describe the hardware/parallelism as identical.

Run a small smoke first:

```bash
RETRACE_OUTPUT="$RETRACE_ROOT/output/retrace_paper_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_paper.sh" --smoke
```

Then start from the official pretrained DFlash checkpoint, with a new output directory:

```bash
export RETRACE_OUTPUT="$RETRACE_ROOT/output/retrace_paper_full_$(date +%Y%m%d_%H%M%S)"
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_paper.sh"
```

Checkpoints are saved at `$RETRACE_OUTPUT/checkpoints/step_001250`, `step_002500`, ... `step_010000`. `latest.txt` points to the last complete checkpoint. Each includes BF16 inference weights, FP32 master parameters, optimizer and per-rank RNG states. Use `DRAFT` pointing to the chosen new checkpoint with the new ReTrace server.

For a full-recipe timing check without changing its LR schedule, replace `--smoke` with `--stop-after 4`. Continue it with the same `RETRACE_OUTPUT`:

```bash
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_paper.sh" \
  --resume "$(cat "$RETRACE_OUTPUT/checkpoints/latest.txt")"
```

Do not resume a stored-pair checkpoint as optimizer state for this recipe. Its objective, batches, examples and update accounting differ. The new route starts from the official checkpoint, as requested.

The optional vLLM target transport is available:

```bash
bash "$RETRACE_ROOT/speculators/examples/train/train_retrace_dflash_paper.sh" \
  --target-backend vllm --server-npus 0,1 --trainer-npus 2,3,4,5,6,7 \
  --target-norm reference --port 8523
```

It verifies the actual evolving draft paths and exports their features; it cannot reuse clean stored-response features for rejected branches. Its transport resends prefixes and is not expected to be faster than local cached verification. No new speed estimate is claimed. The fast stored approach and this recurrent route have different workloads.

## Verification and remaining work

CPU tests cover pretrained tensor preservation, zero-residual identity, suffix alignment, recurrent detached memory, current-round gradients, clean CE, chunk equivalence, epoch accounting, checkpoint resume, sampling probabilities, rejection-distribution arithmetic, and installer idempotence/conflict handling. See `VALIDATION.md` for the executed result.

Not validated here: Qwen3-4B training on your NPUs, native probabilistic vLLM-Ascend end-to-end correctness, held-out gains, exact author prompt selection or Table 2 reproduction. Before a long run, verify the smoke and current-checkpoint sampling change. Before claiming losslessness, compare greedy token IDs with a target-only reference and test stochastic rejection separately; a stochastic run's lack of greedy mismatch is not a distribution test. Report Ascend speedups relative to your own matched target-only run, not the paper's A800 absolute speed.

## Sources

- ReTrace paper, method and Appendices A/B/E: https://arxiv.org/pdf/2608.29748
- DFlash training objective: https://arxiv.org/pdf/2602.06036
- Public DFlash proposal sampler and b16 layout, inspected commit: https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
- Speculators base inspected at `e376ed8724f42e1c7a01df9f3ef870600ffb03e5`; archived earlier ReTrace bundles and your reported logs supplied the adaptation under audit.
