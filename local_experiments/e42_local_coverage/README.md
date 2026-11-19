# E42 fixed local-response coverage experiment

Only training weight changes: half original summary mass squared, half true mass squared normalized to preserve each Query's total old weight. Both are detached. No online changes, no Test access, no raw KV reads.

1. Run `preflight --seed {17,29}`. Each does a discarded 16-step T0 numerical replay, then coverage and fixed head9 checks.
2. Run `preflight --gate`. Stop on failure, do not tune the mixture.
3. Run `core --seed {17,29}` for exactly 2304 T1 updates, final epoch1 fixed.
4. Run `evaluate --seed {17,29} --arm {T0,T1}`. T0 reuses E39 final checkpoints; it is not retrained.
5. Run `report`, `audit --seed {17,29}`, then `finish`.

Module prefix: `python -m local_experiments.e42_local_coverage.`

Full train/development decisions, fixed E29 endpoint diagnostics, all scores, loss/gradient curves and damage concentration are retained in artifacts. Relative metrics keep original denominators; large tail events are not dropped. The original incremental true gains can differ slightly from direct positive sums under cancellation; independent extreme-action checks quantify this without changing the training labels.
