# Validation — 24 September 2026

Executed in an isolated CPU environment using PyTorch 2.10.0+cpu, Transformers 5.14.1, and the downloaded Speculators source at e376ed8724f42e1c7a01df9f3ef870600ffb03e5, augmented with the existing ReTrace registration and this separate recipe.

- 30 tests passed in the recipe/memory/sampling/installer/report suite.
- 3 additional cached-execution tests passed. They compare cached proposals with uncached proposals under nonzero conditioning, reject gradient-mode cache use, and compare complete trajectories plus replay gradients.
- Tests include exact pretrained tensor import, missing-weight rejection, zero-residual identity, clean-label shifting, exclusion of future context, detached suffix memory and request isolation, conditioner/backbone gradients, original CE reduction, equivalent block chunking, deterministic CPU checkpoint resume with FP32 master parameters, and full prompt/epoch accounting.
- Proposal tests compare sampled frequencies with exported q, exercise top-k followed by top-p, preserve greedy/profile behavior, and check the p/q plus residual-distribution identity. These are mathematical/helper tests, not native vLLM-Ascend end-to-end tests.
- Installer tests cover dry-run behavior, backup, preservation of unrelated edits, idempotence, and conflicting-file rejection. Python syntax and all Bash scripts passed syntax checks.

Not run: Qwen3-4B weight loading here, NPU training, CANN operators, HCCL, native probabilistic rejection in your vLLM-Ascend checkout, benchmark evaluation, or Table 2 reproduction. The operator guard and small smoke run are provided for your environment. No runtime speedup or acceptance gain is claimed by these tests.

Extract the self-contained installer with `python install_retrace_table2_corrections.py --extract /tmp/retrace-table2-review` to inspect the source and tests. Existing pretrained-recipe tests operate on tiny randomly initialized Qwen3 fixtures; they do not substitute for evaluation of the full pretrained model.
