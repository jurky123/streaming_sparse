# E29 — Local response Reader

Analytic history/current teacher ablation, then a fixed-capacity local response Reader. Frozen E24 C summaries, E27 adaptive phi and B16; no new QKV or router capacity. Same 4608 updates / seed17,29, initial weights, Adam, training order and state sampling as E27-B.

Run from repository root with the existing project Python environment. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` recommended. Independent seeds can use separate GPUs.

```bash
python -m local_experiments.e29_local_reader.teacher --seed 17
python -m local_experiments.e29_local_reader.teacher --seed 29
python -m local_experiments.e29_local_reader.teacher_report
python -m local_experiments.e29_local_reader.stage0
python -m local_experiments.e29_local_reader.core --seed 17 --mode scale
python -m local_experiments.e29_local_reader.core --seed 29 --mode scale
python -m local_experiments.e29_local_reader.core --seed 17 --mode train
python -m local_experiments.e29_local_reader.core --seed 29 --mode train
python -m local_experiments.e29_local_reader.core --seed 17 --mode analyze
python -m local_experiments.e29_local_reader.core --seed 29 --mode analyze
python -m local_experiments.e29_local_reader.report
python -m local_experiments.e29_local_reader.audit
python -m local_experiments.e29_local_reader.finish
```

`test_once` refuses execution unless all preregistered development gates pass. Do not use Test for selecting or revising the model.

Local loss is exactly mass-weighted squared response error, divided by a fixed train-only median RMS squared for units. It does not normalize each page to cancel mass weighting. Beta and gain losses preserve E27-B rules. This target/loss change was requested; do not claim a loss-identical ablation. No LS coefficient distillation.

Inference takes summaries, Query, frozen bases and model parameters only. Teacher replacement rows and E29-A are offline diagnostics. Missing teacher correction sets both residual and denominator correction to zero. All actions retain exact count and own protection.

Artifacts: `artifacts/e29_local_reader_v1/`.
