# E27-A: Learned page-conditioned Query directions

Use existing E24 data and E25/E26 panels; no new QKV and no Test evaluation.
From repository root with the existing virtual environment:

```
python -m local_experiments.e27_adaptive_query.stage0
python -m local_experiments.e27_adaptive_query.train --seed 17
python -m local_experiments.e27_adaptive_query.train --seed 29
python -m local_experiments.e27_adaptive_query.analyze --seed 17
python -m local_experiments.e27_adaptive_query.analyze --seed 29
python -m local_experiments.e27_adaptive_query.report
python -m local_experiments.e27_adaptive_query.audit
```

Use OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, and an independent
CUDA_VISIBLE_DEVICES for each seed. Network: existing 712 summary features,
per-component RMS normalization, 64-hidden page encoder producing 8x4 BF16
coefficients, shared128->8 Query linear projection, tanh of their product.
No teacher field enters the network. The true residual and true beta are used
only for variable-projection training and offline Oracle-coefficient evaluation.

Each seed trains on its corresponding frozen C checkpoint: all summaries,
physical pages, basis, Risk theta and raw identities stay fixed. Four epochs,
Adam1e-4, clip1, no sweep. The inner five-variable ridge matches E26. Outer
loss is page-energy-normalized residual squared error with fixed E24 train
zscale floor, a declared weighting choice distinct from unweighted MSE.

Oracle LS coefficients use64 queries, M16 decides, raw64 measures actual gain.
Optional fit16 variant still uses true labels, not an online predictor. Untrained
network control is not eligible for checkpoint selection. Select among four
trained epochs by maximum summed development true set gain; Test stays sealed.

Only after both seed gates pass may E27-B begin. The gate requires at least80%
of P1 gain and at least5 percentage points in Oracle recovery over corresponding
E26 fixed-Cphi/true-beta baseline. Do not change rank/width/budget to pass.

Numerics: current code uses the equivalent incremental gain identity rather than
subtracting two large objectives. Old A runs were reevaluated with recheck.py;
a fresh run already uses the stable path. Recompute E26 controls with
`python -m local_experiments.e27_adaptive_query.baselines --seed 17` (also29)
before report. Original pre-correction records remain in before_stable_gain.

If and only if B_gate.json passes, use coefficients.py --seed17/29 (with a space
between option and value) for conditional training, then coefficients.py
--seed 17 --analyze (also29), report_b.py and audit_b.py. Each B model freezes
its corresponding selected phi and C summary. B_protocol.json specifies the
pre-training loss normalization. finish.py assembles the final report/package.
