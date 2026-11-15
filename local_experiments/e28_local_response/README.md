# E28 analytic local response / cross-query reuse

No neural training, new QKV, test access, or model changes. Frozen E24 C17/C29 and E27 selected adaptive phi. Fixed B16, true beta, M16 decisions and Q64 real outcome, Medium one swap or no-op.

Run from repository root using the existing environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m local_experiments.e28_local_response.run --seed 17
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m local_experiments.e28_local_response.run --seed 29
python -m local_experiments.e28_local_response.audit
python -m local_experiments.e28_local_response.audit_shared
python -m local_experiments.e28_local_response.audit_actions
python -m local_experiments.e28_local_response.report
python -m local_experiments.e28_local_response.supplement
python -m local_experiments.e28_local_response.finish
```

Both targets use identical mass-weighted relative-ridge LS (1e-6 trace/5). Centered fits t directly against mass*[1,phi], avoiding explicit t/m overflow. Local fits mass*deltaO against that same design. Reconstruction couples exact beta to the known projected summary-center correction. No true target clipping or event deletion.

History sharing requires identical ordered token IDs, raw BF16 K/V, and current C summary hashes within trajectory/layer/head. Shared fits all matching occurrences; LOTO excludes the evaluated event's full64queries. Current pages retain event-local Oracle fits in all variants. This is an offline teacher diagnostic, not a Page Encoder result. LOTO may use other steps and future chunks and is not causal deployment validation.

Artifacts: artifacts/e28_local_response_v1/. Original model/data files are not modified.

Before `audit_actions`/`supplement`, run both seeds again with `--uniform-local`. An explicitly separate `--uniform-local` sensitivity run removes the global mass weights from local-deltaO LS. It uses the same fixed ridge, no model selection. This prevents confusing target locality with the original globally weighted regression objective. Primary matched-weight comparisons remain unchanged.
