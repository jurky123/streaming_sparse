# E49 frozen numeric audit

Compile (requires system MPFR runtime and g++):
```
g++ -O2 -std=c++17 -fno-fast-math -frounding-math -shared -fPIC local_experiments/e49_numeric/interval.cpp -Wl,-l:libmpfr.so.6 -o artifacts/e49_numeric_v1/interval.so
```
Run `run --seed 17/29 --shard 0/1/2/3 --shards 4`; then `reference --seed 17/29`, `reference --seed 17/29 --near`, `audit`, `report`, `finish` with module prefix `python -m local_experiments.e49_numeric.`

No training. Exact binary scoring inputs are held fixed. Acceptance certifies only the old highest proposed action, never falls through to a second action. MPFR directed exp and outward basic arithmetic bound the ideal gain on the recorded platform; unbounded/zero-containing intervals abstain. This is not model safety or a Dense-energy threshold. Independent certificates use a globally common positive rescaling, not the old energy magnitude.

Reference inputs and arbitrarily precise decimal output are in reference_review.tar.gz. Full512 scores and true gains for all models/events are split into eight archives, listed with SHA in packages.json. No-op is literal0; proposal_reference.csv records top-two gap. The original path's same-mask reduction error is separately recorded on reference events. Original cached data/models are needed for a full replay; the independent reference bundle includes the inputs needed to verify the precision examples without rerunning Reader.
