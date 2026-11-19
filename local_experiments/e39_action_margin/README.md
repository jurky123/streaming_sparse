# E39 — Fixed-capacity critical-action margin continuation

E39-A closes the original E38 action pairs into prediction, Query coverage and output-projection differences. It uses saved E38 true local responses and E30 contributions, without raw attention recomputation. The same frozen B coordinates are used at M16 and Q64; full gains come from the Q64 cached target.

T0/T1 start from each seed's E29 selected Reader. C summary, adaptive phi, B16, Risk, page identity, M16, Medium and one-swap/no-op remain fixed. Both arms receive the same 2304 train updates (2 × 1152), states and optimizer configuration. Adam is reset identically because E29 did not save its optimizer. T0 retains E29 losses; T1 replaces only the set loss with a cost-augmented margin on predicted top4 plus teacher best and no-op. Teacher actions are training-only. The final epoch1 checkpoint is the main result; epoch0 is reported without model selection. No Test execution.

Run with the existing project environment:

```bash
python -m local_experiments.e39_action_margin.core --seed 17 --mode A
python -m local_experiments.e39_action_margin.core --seed 29 --mode A
python -m local_experiments.e39_action_margin.preflight
python -m local_experiments.e39_action_margin.core --seed 17 --mode train
python -m local_experiments.e39_action_margin.core --seed 29 --mode train
python -m local_experiments.e39_action_margin.report
python -m local_experiments.e39_action_margin.audit
python -m local_experiments.e39_action_margin.finish
```

`artifacts/e39_action_margin_v1/protocol.json` was fixed before execution. No hard-negative-count, loss-weight, architecture, learning-rate or threshold search. Raw training KV is used only offline to reconstruct the same frozen summaries; exact N/Z labels are cached. Training retains E29 numerical scoring in both arms. Evaluation uses the algebraically equivalent separately normalized E30 scorer, checked in preflight/audit.

A fixed 48-event training monitor (sorted training files every24) is evaluated before training and at both epochs, with no additional optimization. Development has 288 events per seed from two correlated video trajectories; seeds and head–tiles are not independent video samples.
