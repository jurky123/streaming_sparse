# E24 reproduction

Use the repository venv; set OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=1.
All commands are Python modules under `local_experiments.e24_joint_representatives`.

1. `prepare` registers the 8/2/4 prompt-disjoint trajectories. `stage0` checks the existing two snapshots only.
2. `capture --encode`, then `capture --index 0` through `13`, assigning GPUs with CUDA_VISIBLE_DEVICES. This produces 2016 events and 1008 shared physical KV blocks.
3. `repair_scale --worker 0 --workers 8` through worker7 constructs canonical page outputs/logZ and finite common-scale views. It preserves raw QKV and pages. Then run `dataset_audit` to freeze file hashes. Do not train on unrepaired linear contributions.
4. `train --fit` computes TRAIN-only equal-trace PCA16 and fixed median event-RMS scales.
5. Run `train --group A --seed 17` and seed29 for shared readout warm-up, and `train --group B --seed 17/29` for the Approx baseline.
6. From the identical corresponding A warm-up checkpoint, run `train --group C --seed 17/29` and `train --group A --seed 17/29 --matched`. Main Frozen-R4 is A_matched; warm-only A is not the main C comparator.
7. `finalize` independently audits selected decisions, applies the development gates, and evaluates trained models on Test only if both seeds pass. Precached test teacher data is not model-selection evidence.

Each main phase has 2304 updates, with original grouping/basis/theta frozen. A/C share an additional 2304-update readout warm-up. B does not need a reader. No claim of equal total end-to-end pretraining FLOPs is made.

Canonical exact page logZ must be retained: new layer29/head9 instances have logit ranges exceeding FP64's safe shared exponent span. Linear scaled N/Z may underflow for negligible pages, but beta labels use logZ directly, and no page or instance is discarded. Network weights/phi are FP32, page coefficients are actually BF16, and reference ratios/labels are FP64.

The independent audit intentionally replays the original cached summary for OldRisk, avoiding contamination from single-head BF16 reconstruction rounding. Existing output directories are resumable; for a new experiment use a new OUT version rather than overwriting frozen data.
