# E40 — Reader scale and Query identifiability diagnostic

No training, calibration, new summary statistics, original attention recomputation, new data capture, or Test access. Original E29 Reader and frozen C/phi/B16 are used, not E39 continued models.

The existing E30 development cache supports exact FP32 phi collision checks. It does not contain preactivations, the four original RMS scales, or the training C summaries. After explicit user authorization, `full capture` reads existing KV offline only to rebuild and persist the frozen C summary. True page N/Z/output remain cached. Development phi, predicted beta, true beta and original action identities are replayed exactly. This is not an online probe.

```bash
python -m local_experiments.e40_reader_scale.full --seed 17 --split train --mode capture
python -m local_experiments.e40_reader_scale.full --seed 17 --split development --mode capture
python -m local_experiments.e40_reader_scale.full --seed 17 --split train --mode analyze
python -m local_experiments.e40_reader_scale.full --seed 17 --split development --mode analyze
# Repeat for seed29, then:
python -m local_experiments.e40_reader_scale.report
python -m local_experiments.e40_reader_scale.audit
python -m local_experiments.e40_reader_scale.finish
```

Use the existing project virtualenv with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Sources and scope are recorded in artifacts/e40_reader_scale_v1/protocol.json. Full raw records, reconstructed summaries and diagnostic arrays are local artifacts; the review package contains summaries and code.

Exact collision grouping uses verified raw token/KV plus C-summary identity and all four unrounded FP32 tanh outputs. Current noisy pages share no cross-event coefficients. Signed zero is canonicalized. Near-feature matches are descriptive only; no information-theoretic lower bound is assigned to them. Cross-tile, cross-step, cross-chunk and strict-past donor support are separated.

Page-scale analysis compares pages within the same event/Query set. Normalized 712D Euclidean nearest pages are contrasted with a fixed cyclic control. Feature-distance bins and scale p10/p90 are fixed using training data only, by layer/head. RMS normalization with epsilon is injective in exact arithmetic; numerical inversion diagnostics distinguish implicit ill-conditioned scale information from literal loss. Correlations do not establish that appending scales will improve action selection. The conditional learned probe is not automatically started.
