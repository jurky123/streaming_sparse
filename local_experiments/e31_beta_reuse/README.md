# E31 — Analytic beta reuse and loss alignment

No training, new QKV, Test evaluation, new cache or deployed module. Uses frozen E29 response and E30 caches plus E28 physical identity mappings.

From repository root using the existing project environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m local_experiments.e31_beta_reuse.run --seed 17
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m local_experiments.e31_beta_reuse.run --seed 29
python -m local_experiments.e31_beta_reuse.report
python -m local_experiments.e31_beta_reuse.audit
python -m local_experiments.e31_beta_reuse.finish
```

Independent seeds can run concurrently. `--limit-groups 1` is a pilot. Completed runs are protected; use a new output version for reruns.

A0 reproduces E30 mass² ridge. A1 normalizes mass² within each appearance of each physical page before pooling contexts. A2 uses A1 weights, replacing only phi4 with the raw existing Key variance feature; no dimensionality or fitted scale search. Current-page beta is the identical original EventOracle across all analytic reuse variants, isolating history changes.

Context donors: same chunk/step different tile; same chunk different step; strictly earlier chunk. Missing historical donor pages retain E29 predicted beta. Matched-support comparisons intersect pages available in all three categories; report coverage rather than treating unequal supports as comparable.

B evaluates quality regression and frozen-response TrueBeta teacher gain distillation loss (Huber + 0.1 KL including no-op, both temperatures0.3). No optimizer or training. Exact teacher yields zero diagnostic loss. Correlations are within-event across seven non-True schemes and within-method across events, not evidence that optimizing a loss necessarily succeeds.

Artifacts: `artifacts/e31_beta_reuse_v1/`. Reports/tables and compact tar exclude full derived fit/panel caches. Original artifacts and model parameters are unchanged.
