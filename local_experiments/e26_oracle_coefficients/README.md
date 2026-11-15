# E26-A: Oracle page coefficient diagnostic

Only analytic fits are performed; no E26-B training and no Test access.
Use the E24 Python environment and repository root. Existing E24 data and E25
panels/checkpoints must be available and match the provenance hashes.

```
python -m local_experiments.e26_oracle_coefficients.run --seed 17
python -m local_experiments.e26_oracle_coefficients.run --seed 29
python -m local_experiments.e26_oracle_coefficients.audit
python -m local_experiments.e26_oracle_coefficients.extreme_audit
python -m local_experiments.e26_oracle_coefficients.finish
```

Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and CUDA_VISIBLE_DEVICES per process.
The two seeds can run independently on separate GPUs. `--limit 7` is the fixed
implementation pilot, not a model-selection set. Main results use all 288 dev
instances for each seed. Fixed ridge=1e-6 relative to Gram trace/5, all five
coefficients regularized; no action-label tuning.

Residual fit design is mass*[1,phi], beta fit uses [1,phi]. Original C phi is the
primary design; D phi is a matched secondary control because E25-D changed its
query weights. Oracle coefficients are FP64 and fit to all64 teacher Queries,
then evaluated at the fixed16 probes. They are not deployment parameters or
out-of-query generalization results. Each event has its own coefficient fit,
held fixed across all five E25 counterfactual states.

`Cphi_trueBeta` isolates the residual factorization error. `SVD5_trueBeta` is the
best Frobenius rank5 of each true64x16 response with exact quality correction,
not a decision-optimal bound. All chosen actions are evaluated by true64Query
NMSE, and Set/Independent share identical predicted information.

Large logmass ranges are handled through algebraically equivalent per-page
scaling and direct positive denominator sums, not beta clipping. Saved
`coef_z_scaled` is the original coefficient times exp(logmass_page_scale).
Page maximum target scaling cancels when coefficients are restored. Canonical
teacher data, E25 panel state identity and exact raw64 objective remain fixed.

The compact package excludes original QKV, existing model checkpoints and the
~350MB full fitted coefficients/panels. Their local paths/hashes are preserved.
