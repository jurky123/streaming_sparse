# E30-A — Beta capacity, reuse and set-decision sensitivity

Only A1–A3. No training, no new QKV, no Test access. Use the existing E24/E25/E27/E28/E29 artifacts and project Python environment, from repository root.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m local_experiments.e30_beta_diagnostics.run --seed 17
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m local_experiments.e30_beta_diagnostics.run --seed 29
python -m local_experiments.e30_beta_diagnostics.components --seed 17
python -m local_experiments.e30_beta_diagnostics.components --seed 29
python -m local_experiments.e30_beta_diagnostics.report
python -m local_experiments.e30_beta_diagnostics.audit
python -m local_experiments.e30_beta_diagnostics.finish
```

The two seed runs can execute concurrently. Completed runs are protected against accidental replacement; use a new artifact version for reruns. `--limit-groups 1` is a pilot and does not create a completion marker.

A1: mass-squared weighted ridge, fixed relative regularization; original adaptive five functions. Shared/LOTO history page IDs are inherited from E28 and summary hashes rechecked. LOTO excludes the held event's complete Q64. Current pages retain event Oracle coefficients. All fitted beta variants are teacher diagnostics, not deployable.

A2: candidate union is the initial Medium panel. Teacher replacement recomputes all aggregate state. Candidate/background effects are not additive.

A3: regression errors, exact-quality mass errors, gain ranking, residual-numerator I/B diagnostics, full beta gradient for the baseline proposal, top-page concentration and exact counterfactual corrections. Local derivative attribution is not an exact finite-change decomposition.

Outputs: `artifacts/e30_beta_diagnostics_v1/`. Root reports/CSVs plus protocol, gates and audit files are reviewable; per-event `beta_*.pt` and `panel_*.npz` retain derived tensors and score surfaces. No original data or checkpoint is overwritten. This stage does not launch conditional B training even if the capacity gate passes.
