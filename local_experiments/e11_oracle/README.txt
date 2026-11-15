E11: full-page exact set optimization and D02 cancellation diagnosis

Frozen baseline: 178f95a8 / summary_pretrained. Run from the isolated baseline tree
/tmp/group_response_178f95a8, never from the modified optimization checkout.
Python: /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
Output root: /tmp/e11_results

1. Capture actual stored BF16 summaries on original G-KVR L3 trajectory:
   python -m local_experiments.e11_oracle.capture --case horse_field
   python -m local_experiments.e11_oracle.capture --case traffic_crossing
   The six-chunk prefix must be bitwise equal to archived baseline latents.
   Every selected Q/K/V must equal the prior 120-event archive.
2. Extract FP64 contributions and original M16 mass/risk masks:
   python -m local_experiments.e11_oracle.contributions /tmp/e11_results/snapshots/*.pt
3. Validate:
   python -m local_experiments.e11_oracle.check_invariants
   python -m local_experiments.e11_oracle.check_small
   python -m local_experiments.e11_oracle.check_real
   Enumeration proves only the reduced problem. Keep timed-out checks, too.
4. Full-P pilot, each .npz separately:
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m local_experiments.e11_oracle.run_pilot INPUT.npz --limit 900
5. Sampled closed loop, after cost gate in protocol.json/sample_protocol.json:
   python -m local_experiments.e11_oracle.rollout --case horse_field --method O-OPT --layer 15 --limit 60
   --layer 0 means sampled layers 1/8/15/22/29, NOT all 29 layers.
   Methods O-MASS/O-RISK use exactly the same intervention coverage.
   Each new state computes its own current contributions and masks.
6. python -m local_experiments.e11_oracle.report

Solver:
Gurobi 13.0.3 is installed, but its restricted license rejects large models.
SCIP 10.0.2 / PySCIPOpt 6.3.0 uses the exact convex epigraph equivalent of the
MIQP, keeping all candidate pages and the original variable denominator.
Every query has an epigraph for its sum of squared t variables; their sum is
minimized. These are convex quadratic constraints with a linear objective.
No Gram matrix, candidate-pool restriction, fixed denominator, or local NLP
solution replaces the problem. sum(w_qp)=(K-1)y_q is a valid strengthening.
NLP/subnlp/multistart are disabled after initial native-memory failures.
All failed development checks are retained under debug paths and logs.
The solve is serial SCIP optimize; maxnthreads=8 is a configuration ceiling,
not a claim of eight active threads. Resource accounting uses CPU seconds.

Numerics:
A single FP64 log scale per query is shared by all raw and representative
contributions. Representative mass 16 is included. Inputs and stored
summaries remain deployed BF16; basis is BF16-rounded as in SelectedAttention.
Positive sums, not cancellation-prone base+delta sums, construct denominator
extrema. Warm masks are validated and evaluated using all 64 queries.
Objective scaling changes neither minimizer nor reported NMSE; solver gaps
are converted back to original units. Reported certificate tolerances are
1e-10 absolute / 1e-6 relative, with independent recomputation and count/own,
integer, product and linear residual checks. Only GLOBAL_TOL passes all.
Checkpoints use an independent process because SCIP can hold Python's GIL
inside an LP. Each saved bound includes its actual most recent solver timestamp.
Bounds can be stale during a long root LP, but remain valid. Never interpolate
an improving bound or claim a checkpoint executed a new solver iteration.
Warm-start-only runs without a nontrivial search bound are conservatively
NO_FEASIBLE_CERTIFICATE; they still provide a verified baseline feasible UB.

L1 post-W_O replaces only the registered head and keeps the other 11 heads
Dense. It is a separately measured linear-projection error, not a globally
optimal post-W_O claim. SelectedAttention validation uses the original FP32
path and its deployed BF16 output; this differs from the FP64 oracle model.
D02 is computed per query and head before aggregation. Normalize A/C by the
actual denominator squared for additive NMSE attribution; raw A/C have a
query-specific common scale. kappa<1 alone does not imply lower output error.

Video metrics preserve the previous protocol: 9 chunks / 105 frames;
RGB[21,101), LPIPS and SSIM at the established 256x384 evaluation scale,
PSNR from full-resolution pixel MSE. Store per-frame curves and latent NMSE.
Matching G-KVR Dense reference has identical clean grouping/cache protocol.

Primary solver documentation:
https://pyscipopt.readthedocs.io/en/latest/tutorials/expressions.html
https://www.scipopt.org/doc/html/PARAMETERS.php
https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/objectives.html
https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html
