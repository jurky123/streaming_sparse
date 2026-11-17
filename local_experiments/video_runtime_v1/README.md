# Video runtime v1

Current accepted baseline: full-coverage `C_s17/deployment.pt` and
`Full_Pset_s17/selected.pt`, E27 Phi and paired B16 basis. The batch and worker
entry points use `deployment.CURRENT`: history360, tile72, old fast exact/full
mean compensation, fused RoPE, split Reader, FP32 Reader output storage, BF16
CO/A storage, stable exclusion scan, projected Risk, cached log mass and explicit
endpoint denominator caching. E24 single-head broadcast and E48 Reader defaults
remain historical weight-loader fallbacks, not the production worker defaults.

Algorithm: Page64/Q64/R4, full 128D V mean + rank8 detail, M16/B16, 10% exact
pages, one 16×32 exchange or strict-positive no-op. Layer0, early chunks and clean
commits retain their existing dense semantics.

## Single-video performance and acceptance

Use one video or a captured module input for performance measurements. Routine
acceptance uses 9 chunks/105 frames, 768×512, 16 FPS, 5 history chunks/360 pages.
15-second quality runs require an explicit request; do not launch 30-second runs.
The benchmark entry has experimental defaults: use the complete accepted command
in [CURRENT_VIDEO_BASELINE.txt](../../local_setup/CURRENT_VIDEO_BASELINE.txt).
Do not equate multi-GPU batch throughput with single-video acceleration.

The current accepted report, source navigation, timing and precision contracts:
[storage_cached_video_v1/MODULE_DISTANCE.zh.md](../../artifacts/video_runtime_v1/storage_cached_video_v1/MODULE_DISTANCE.zh.md).
Previous experimental helpers retain their historical configurations and are not
substitutes for the current deployment command.

## Batch quality evaluation

Run from the repository root; choose a fresh root for a new configuration:

```bash
PYTHON=/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
$PYTHON -m local_experiments.video_runtime_v1.manifest \
  --cases city_walk kitchen_cook --seeds 701 702 --chunks 9 \
  --output /tmp/manifest.json
$PYTHON -m local_experiments.video_runtime_v1.batch \
  --root artifacts/my_video_run --manifest /tmp/manifest.json --gpus 0 1 \
  --local-assets /tmp/streaming_sparse-0/runtime
```

`batch.py` gets tile_batch=72 and history_pages=360 from `deployment.CURRENT`.
It forwards `--checkpoint`, `--reader-checkpoint`, `--history-pages`,
`--local-assets`, `--tile-batch`, `--profile` and `--validate` to workers. Explicit
CLI overrides are recorded in the existing execution/job identities. A non-default
tile size is an experimental configuration and cannot inherit tile72 timing claims.

The manifest/queue currently accept 9 or historical 41 chunks, not 21. The example
uses 9; 41 remains available for reproducing explicitly requested historical work.
Each case/seed/length includes Dense, C+Risk, and C+Risk+Full_Pset. Noise, generation
configuration and weights are fixed within the comparison.

Each GPU runs one resident process and one video at a time. Generation completes
before VAE/encoding/metrics. Dense pixels are decoded once and reused. Existing
per-frame metrics cover RGB [21,101) for the short video: native-resolution MSE/
PSNR, 256×384 SSIM and AlexNet LPIPS. Early/late summaries can be derived from the
first/last 40 saved per-frame measurements without another decode or metric run.
`compare_videos --root ...` creates Dense | C+Risk | Full_Pset comparison videos
from encoded videos, with no additional VAE decode.

Re-run the same batch command to resume. Completed tasks are skipped; clean
latents rebuild KV/summary state for interrupted jobs. Retry failed jobs with
`queue --root ... --retry-failed` only after fixing their cause. A live worker is
not reclaimed on a timeout. Code/weight/configuration changes require a new root.

## Implementation and review entry points

- `deployment.py`: accepted weights/options, shared initialization, metadata.
- `batch.py`, `worker.py`, `queue.py`: launch, persistent workers, immutable job queue.
- `runtime.py`, `page.py`: resident KV, clean-commit summaries/encodings, temporary
  noisy-page updates; page coefficients are cached, query responses are not.
- `pipeline_graph.py`: Risk → candidates → 512 scores → no-op/LUT CUDA Graph.
- `reader_reduce.py`, `reader_split.py`: scalar/vector Reader with FP32 output
  containers and FP64 consumers; full WORK is the accepted default.
- `endpoint_den.py`: positive remove/add endpoint mass combination and caching.
- `exclude_scan.py`: positive prefix/suffix exclusions and dominant-page rescaling.
- `legacy_components.py`, `review_v1/tuned_attention.py`: accepted fast exact
  Page64 LUT path, full-mean/rank8 compensation and base2-LSE merge.
- `profile_storage_route.py`, `check_storage_contract.py`,
  `check_endpoint_fixtures.py`: full-route profile and numerical contracts.

Weights/video arrays stay local; committed reports and compact evidence identify
configuration, source hashes, timing scope and optimization-only errors.

## 9d4d33d审查收尾

批量入口继承tile72/history360并透传部署参数。[84视频质量结果](../../artifacts/video_runtime_v1/quality_batch_9d4d33d_v1/REPORT.zh.md)已完成；[P2有界实验](../../artifacts/video_runtime_v1/p2_bounded_v1/REPORT.zh.md)无净收益，默认保持完整WORK；[FP32扫描消费验收](../../artifacts/video_runtime_v1/scan_fp32_video_v1/MODULE_DISTANCE.zh.md)已纳入部署默认。
