# E25 — Frozen set-decision diagnostics

Run from repository root using the E24 environment and existing immutable
`artifacts/e24_joint_representatives_v1` cache. No test split is evaluated.

1. Run `python -m local_experiments.e25_decision_diagnostics.run --group A --seed 17`
   for groups A/C, seeds 17/29 (independent GPUs can run simultaneously).
2. Run `python -m local_experiments.e25_decision_diagnostics.audit`.
3. Run `python -m local_experiments.e25_decision_diagnostics.report`.
4. Only if `D_gate.json` passes, run
   `python -m local_experiments.e25_decision_diagnostics.retrain --seed 17`
   and seed29. Reader starts fresh, C summaries frozen, four epochs / 4608 updates.
5. Run step1 for group D, seeds17/29, then regenerate report.

Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. CUDA_VISIBLE_DEVICES selects GPU.
The group A checkpoint is E24 budget-matched A, not its warm-up-only model.

P1 evaluates projected true output error, algebraically equivalent to online-centred
true t/u with orthogonal B; the output-space expression avoids cancellation at
extreme logit scales. P0 follows the exact E24 prediction and no-op rule. All
proposals receive true64-query evaluation. Hindsight is teacher-only.

Counterfactual states are reconstructed using the E24 state-generating rule under
each frozen summary, rather than reusing stale old-summary gain labels. The common
panel is the intersection of legally retained initial Medium pairs across states.
All masks, pairs and scores are stored in per-event compressed NPZ files.

The D gate (both C P1 recover >=50%, >=20pp above proposal+hindsight) was fixed
before ladder results. It governs this diagnostic only; it is not a deployment or
test-release gate. Test stays sealed even if D improves.
