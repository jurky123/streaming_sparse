# E45 frozen E43 replay

Run `python -m local_experiments.e45_frozen_replay.run --seed 17` and seed29, then `python -m local_experiments.e45_frozen_replay.report`. Uses existing E24/E29/E40/E43 artifacts, never trains. Each seed verifies P-to-L exact equivalence across144 states and original E43 matrices across96 states.

The heldout set is48 fixed new background swaps, using second eligible non-Medium endpoints, original Queries and original candidate pool. No Query or trajectory generalization claim. Cross-state preference reuse includes no-op decisions, evaluated against target-state full-Q64 true gains.

`source_snapshot.tar.gz` preserves the precise source dependency closure used, including previously unarchived dependencies. It is not a substitute for missing data or model artifacts.
