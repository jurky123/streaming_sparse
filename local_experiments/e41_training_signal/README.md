# E41 training signal audit

Frozen E29 audit only. Inputs are E40 frozen summaries and E24 canonical page contributions. No raw KV access, optimizer, parameter updates or Test evaluation.

Run from repository root with the existing runtime Python:

```
python -m local_experiments.e41_training_signal.run --seed 17
python -m local_experiments.e41_training_signal.run --seed 29
python -m local_experiments.e41_training_signal.report
python -m local_experiments.e41_training_signal.audit
```

`protocol.json` fixes the gradient subset before results. Output-gradient analysis covers every training/development Risk state. Parameter gradients cover 48 train Risk states per seed; they cannot establish historic or long-term conflict. The sorted-quartile selection covers all trajectories/layers but happens to select head6 only; full output-gradient analysis covers all heads. Report this restriction rather than extrapolate it.

The original M16 forward path and E40 Q64-cache slicing can have small floating-point differences; record these, while requiring identical executed/oracle action indices. Original E29 loss reductions, train-only fitted scales and BF16 coefficient cast are preserved.

No KL loss is used to update parameters. Its mean-query analytical gradient is only a reference.
