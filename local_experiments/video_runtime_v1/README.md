# Video runtime v1

Fixed Page64 / Q64 / R4 / full 128D V mean + rank8 detail / M16 / B16 /
10% exact pages / one 16×32 exchange or strict-positive no-op.

**Performance testing policy (latest user instruction): use one video only.**
Prefer isolated module timings when they answer the question. Do not expand
prompt/seed/video matrices for speed measurements. A tensor batch here combines
heads and query tiles of that one video, not multiple videos. Batch video jobs
remain a separate throughput/quality feature; their aggregate wall time is not
an acceleration benchmark.

Default weights deliberately broadcast one E24 C pooling checkpoint position
(l15h0, model seed17) across all sparse positions. E48 Reader, E27 Phi and frozen
train-only B16 are loaded separately. Outputs are engineering substitutes, not
full-coverage model quality results. `--checkpoint` accepts the native
`model` state_dict emitted by the full-coverage exporter. New weights require a
new run directory and worker, ensuring all cached summaries/graphs are rebuilt.

## Implementation

- `runtime.py`: resident native KV, current scratch, commit-time history slabs
  and page encodings, once-per-step current encoding, fixed-budget lifecycle.
- `page.py`: normalized features, BF16-rounded 85 Reader and 32 Phi coefficients,
  and cached output projections. Query-dependent responses are not cached.
- `scores.py`: positive five-group log denominators, shared background, small
  exclusion matrices, FP64 endpoint incremental gain, dominant-page fallback.
- `pipeline_graph.py`: preallocated Risk → candidate pool → scores → no-op/LUT
  graph and compensation graph; page inputs staged once per layer/forward.
- `attention.py`: repository Page64 exact kernel reads native KV directly;
  compensation uses full mean GEMM + rank8 detail GEMM. Exact base-2 LSE is
  converted to natural log; R4 representative token mass is 16.
- `queue.py`, `worker.py`, `batch.py`: SQLite claims, immutable identities,
  resident workers, clean-latent resume, separate generation/postprocess stages,
  one Dense VAE decode per reference job, cached reference pixels, comparisons.

## Single-video performance checks

Run from repository root with the existing environment:

```bash
PYTHON=/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
CUDA_VISIBLE_DEVICES=7 $PYTHON -m local_experiments.video_runtime_v1.benchmark --module scorer
CUDA_VISIBLE_DEVICES=7 $PYTHON -m local_experiments.video_runtime_v1.benchmark_video \
  --case city_walk --seed 701 --chunks 9 --reps 1 --warm-chunks 3
```

`benchmark --module scorer|readout|attention|kv` isolates one module and uses city_walk source data only. The default module is scorer, with 72 head–tile events from one video. `--module all` is available when a full module breakdown is required. Existing timing files are preserved; choose a fresh `--output` for subsequent measurements. Historical timing
artifacts from before the single-video instruction used an E49 forest_run input
for the scorer and city_walk for attention; they are retained as historical
component measurements, not a new multi-video speed benchmark.

The complete single-video comparison times old `StreamingRuntime` versus new
`VideoRuntime` with the same engine, weights and noise. Both receive a three
chunk warmup. Remaining optimized graph shapes can be captured in its timed
run; this makes the measurement conservative and is explicitly recorded.
Generation timing excludes model loading, VAE and file IO. There is one sample
per implementation, not a statistically established population speedup.

## Correctness checks

```bash
CUDA_VISIBLE_DEVICES=7 $PYTHON -m local_experiments.video_runtime_v1.validate
CUDA_VISIBLE_DEVICES=7 $PYTHON -m local_experiments.video_runtime_v1.check_resume
CUDA_VISIBLE_DEVICES=7 $PYTHON -m local_experiments.video_runtime_v1.check_cache
```

Checks cover frozen E49 action/sign cases, dominant-page removal, original
candidate rules, no-op, cached Reader, local attention error, exact clean replay
continuation, and all 338 surviving ring slots in layers 1/15/29 after 41 chunks.
FP32 QK/reduction changes and the exact attention implementation can cause
closed-loop trajectories to differ from the old reference; local gates do not
establish video quality equivalence.

## Batch generation / quality functionality

For quality or lifecycle work, separate from speed measurements:

```bash
$PYTHON -m local_experiments.video_runtime_v1.manifest \
  --cases city_walk --seeds 701 --chunks 21 --output /tmp/manifest.json
$PYTHON -m local_experiments.video_runtime_v1.batch \
  --root artifacts/my_video_run --manifest /tmp/manifest.json --gpus 7 --profile
```

Each GPU process generates one video at a time and reuses its engine, weights,
workspace and compiled graphs across tasks. Generation finishes before VAE,
video encoding and metric stages. Dense must exist in every prompt/seed/length
group; its decoded pixel cache is reused by the sparse methods. Comparison
videos have Dense on the left. `compare_videos` streams existing MP4s into a
three-column Dense | C+Risk | Full_Pset video without any new VAE decodes.

Re-run the same batch command to resume. Completed tasks are skipped, orphaned
same-host claims are recovered, and existing clean chunks rebuild raw KV and
all history encodings. Failed tasks require explicit
`queue --root ... --retry-failed` after fixing the cause. Code or weight identity
changes require a fresh output directory. A live worker is never reclaimed by
an arbitrary timeout.

Current measurements, limitations and artifact locations are recorded in
`docs/research/evidence/video_runtime_v1/REPORT.zh.md`.

## Hotspot profiling and optional fusion

`profile_route` profiles compiled routing and compensation on one city_walk
capture, records CUDA traces and the principal multiply-add counts.
`check_tail_fusion` compares the original and compiled full-mean/rank8 tail.
`VideoRuntime(..., compile_route=True, compile_tail=True)` opts into whole-route
and compensation compilation; warmup precedes CUDA Graph capture.
Compensation fusion remains opt-in: local Attention gates pass, but the measured
closed-loop latent NMSE versus the original tail is 0.1944. See
`docs/research/evidence/video_runtime_v1/HOTSPOT_ROUND3.zh.md` for numeric contracts
and the 40.01 → 37.82 second single-video generation measurement.

The next hotspot is handled by `scores_action.py`: a per-action FP64 Triton
reduction shares scalar coefficients across output coordinates and evaluates
the five-group fallback only when needed. Use `action_kernel=True` together
with compiled routing. `check_action_kernel` runs the numerical contracts.
The prior 9-chunk tail/action video outputs were bitwise equal.

Formal acceptance uses only one city_walk seed at 9 chunks/105 frames and
21 chunks/240 frames (15s), 360 history pages, 768×512 and 16 FPS.
`benchmark_hotspots --chunks` accepts 9 or 21 (default21). No further 30s runs.
`dense_native` is the official speed denominator; `dense` is a grouped-runtime
control. Completed 15s files remain in `formal_15_30/15s`; short files and
current protocol are in `formal_short_15/`.

This round is closed. See `docs/research/evidence/video_runtime_v1/ROUND_CLOSEOUT.zh.md`
for dominant-FLOP estimates, measured speed and optimization-only error.
Compensation compilation stays opt-in: small local errors did not preserve the
video trajectory. The action kernel can be enabled independently of tail fusion.
