两轮优化收尾：`summary_moments=True`已通过三档短视频逐位验收；`risk_online=False`保持关闭。详细代码位置、profile、理论流量与误差见`artifacts/video_runtime_v1/post_rope_round{1,2}/MODULE_DISTANCE.zh.md`。

当前 `native_rope=True`，原生Dense测速共用逐元素/RMS/RoPE优化；详见 `artifacts/video_runtime_v1/fair_dense_v1/MODULE_DISTANCE.zh.md`。

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

## 两档和集合修正预算（本轮）

当前部署为M16路由入口，Full_Pset开关独立于`mode=stream`：

- 基础档：benchmark命令使用`--no-full-pset --route-m16`。不加载Reader/Phi/B，不分配页面编码缓存。
- 集合修正档：`--full-pset --pset-steps all --route-m16`，为完整方法默认。
- 策略实验：`--full-pset --pset-steps last --route-m16`，只在第4次noisy forward修正；clean commit持续维护Reader历史。它改变推理策略，不能称作同算法实现加速。
- G=2原型：`--reader-vector-g2`，暂不默认开启；满窗口route约2%收益，短视频仅约0.10%，不继续参数筛查。

基础档专用清单可用manifest的`--methods dense c_risk`；worker按清单自动跳过修正器加载。混合清单保留两套图，实际job档位写identity。batch与worker均支持`--pset-steps all/last`，两种策略使用不同结果根目录。

[同口径两档速度、模块增量、优化误差、预算质量及实现位置](../../artifacts/video_runtime_v1/tiers_m16_v1/MODULE_DISTANCE.zh.md)。仅一个固定短视频验收，未重跑84质量清单，不能把末次预算这一条视频的结果推广为稳定质量收益。

## Full_Pset 范围筛选实验

[逐层/head/step的完整排名和五种选择方案](../../artifacts/video_runtime_v1/pset_scope_v1/REPORT.zh.md)：city_walk、seed701/702、9chunks。固定形状重跑104份视频；按层前1/4的平均LPIPS最低，生成相对完整档1.252×。

当前stream命令增加`--full-pset --pset-policy artifacts/video_runtime_v1/pset_scope_v1/presets/top_layers.json`开启层筛选；其他文件为top_heads、top_steps、intersection、additive_quarter。batch/worker也支持该参数，文件SHA写identity。preset优先于`--pset-steps`，并强制static_route=True：实验已验证自动动态编译可能让后续策略依赖此前任务顺序。不开preset时保留既有默认；没有将单场景筛选直接设为全局默认。

## Frozen intersection-policy search and validation

[Search report](../../artifacts/video_runtime_v1/intersection_search_v1/REPORT.zh.md):
341 policies, city_walk seeds701/702, fixed 7-layer × 3-head × 1-step budget.
Both selected policies were evaluated alongside the original intersection and
layer/head/step selections on 5 other prompts × seeds703–706 (140 short videos).
[Validation report and videos](../../artifacts/video_runtime_v1/intersection_validation_v1/REPORT.zh.md).
The original intersection has better mean validation LPIPS than either searched
policy at essentially identical cost; defaults were not changed. Policy files
are accepted through `--pset-policy`; the frozen exact configurations and hashes
are in the validation `protocol.json`.

## Three retained deployment tiers

The user selected `base` (C+Risk, no Full_Pset), `intersection` (layers
3,4,5,8,14,17,27; heads3,4,9; noisy step0), and `step` (all applicable
layers/heads; noisy step0). These are inference budgets, not distinct kernels.
The searched variants are retained only as experiment evidence. Full execution
remains available for research; it is not one of the three retained tiers.

Run any retained tier using accepted defaults and fixed-shape compilation:

```sh
python -m local_experiments.video_runtime_v1.accepted_tiers \
  --tier intersection --output artifacts/video_runtime_v1/my_run \
  --local-assets /tmp/streaming_sparse-0/runtime
```

Change `--tier` to `base` or `step`. This entry uses a 9-chunk video, warmup,
three timed repetitions and a separate stage-profile run. For batch generation,
use `c_risk` for base; use `c_pset` and `--pset-policy` pointing to the tracked
`presets/intersection.json` or `presets/step.json` for the other two tiers.
Clean Reader cache maintenance remains enabled in both correction tiers.

### Clean grouping dataflow optimization

Accepted defaults now remove the unused final feature gathers and borrow
CUDA Graph grouping outputs until the next replay. Three retained tiers passed
bitwise latent comparison on the fixed 9-chunk acceptance video. The alternative
continuous-axis FP32 softmax (`--grouping-softmax`) remains experimental: it is
faster but changes the reduction tree and can change hard grouping and video
trajectories. Use `--no-grouping-spatial --no-grouping-dataflow` in `accepted_tiers`
to reproduce the original grouping baseline. See [module and video report](../../artifacts/video_runtime_v1/grouping_dataflow_v1/MODULE_DISTANCE.zh.md).

### Stable spatial softmax (accepted)

`grouping_spatial=True` is now shared by benchmark, retained tiers and workers.
The CUDA kernel preserves PyTorch 2.6's per-column FP32 accumulation and
32-thread reduction, distributing four columns per CTA. Softmax is 4.697×
faster and the complete grouping is 1.182× faster; all three short-video tiers
have bitwise-identical latents and correction counters.
[Acceptance, operation counts and source locations](../../artifacts/video_runtime_v1/group_softmax_v1/MODULE_DISTANCE.zh.md).

For the previous accepted grouping, use `--no-grouping-spatial` in
`accepted_tiers` or `benchmark_hotspots`. To reproduce the original grouping,
also pass `--no-grouping-dataflow`. The trajectory-changing layout prototype
requires `--no-grouping-spatial --grouping-softmax`. Both softmax variants
cannot be enabled together. The custom CUDA kernel targets A100/sm80 and is
compiled with NVCC during warmup into the fast `/tmp` cache; source SHA is
recorded in deployment identity and `.cu` is included in benchmark snapshots.

### Tail lifetime screening (not deployed)

[Bounded experiments and Nsight counters](../../artifacts/video_runtime_v1/tail_liveness_v1/MODULE_DISTANCE.zh.md)
cover sequential mean/detail consumption, split mean/detail kernels, BQ32, and
paired BF16 conversion. None improves the full-window tail consistently; the
accepted tail remains unchanged. Hardware counters point to shared-memory
layout conversions, not saturated HBM or Tensor Cores. Source-correlated
wavefront evidence is retained for the next structural implementation.

### Tail IR layout experiment (not deployed)

[Controlled TTGIR edits and hardware results](../../artifacts/video_runtime_v1/tail_layout_v1/MODULE_DISTANCE.zh.md)
confirm bitwise outputs on the captured P432 input. Pairing adjacent BF16
values removes most shared-store bank conflicts but adds shared-load conflicts;
steady tail speedup is only 1.010×, about 2ms for the tested full-window calls
in a short video. The accepted runtime does not depend on these IR rewrites.

### Complete-block capture (explicit experiment)

`accepted_tiers --block-graph` preserves the current summary and separates
correction/head branches. Fixed cross-attention K/V buffers support resets and
new prompts. [Three-tier acceptance](../../artifacts/video_runtime_v1/block_retained_v1/MODULE_DISTANCE.zh.md)
shows bitwise latent/counter equality and 1.010–1.014× warm generation speedup.
Extra capture cost needs roughly 457–1131 subsequent short videos to amortize,
so this is not a deployment default.

### Accepted block pointwise fusion

`block_pointwise=True` is shared by the three retained tiers and resident workers.
Two modulation and two gated residual expressions retain their intermediate
BF16 rounding while eliminating large temporary tensors. LayerNorm and GEMMs
are unchanged. [Module and video acceptance](../../artifacts/video_runtime_v1/block_pointwise_v1/MODULE_DISTANCE.zh.md)
shows bitwise latent/counter equality and 1.032–1.037× generation speedups.
Use `accepted_tiers --no-block-pointwise` for the previous implementation.

### Accepted RMSNorm dataflow

`rms_dataflow=True` preserves the original FP32 mean, epsilon addition and rsqrt,
while fusing the pointwise stages before and after them. Intermediate BF16
rounding before the learned weight multiplication remains explicit.
[Contracts, profile and three-tier acceptance](../../artifacts/video_runtime_v1/rms_dataflow_v1/MODULE_DISTANCE.zh.md)
show bitwise square/mean/output and video latent/counter equality; generation
speedup is 1.077–1.087×. Disable with `--no-rms-dataflow` in accepted_tiers or
benchmark_hotspots. The resident worker uses the shared deployment default.

### Accepted fixed-order square/mean fusion

`rms_mean=True` removes the FP32 square array while reproducing the current
PyTorch D1536 mean tree (four accumulators and increasing warp offsets).
[Provenance, contracts and three-tier acceptance](../../artifacts/video_runtime_v1/rms_mean_v1/MODULE_DISTANCE.zh.md)
show bitwise mean/output and video latent/counter equality; generation speedup
is 1.027–1.032× over the preceding RMS dataflow implementation.
Use `--no-rms-mean` for that preceding path, or `--no-rms-dataflow` to disable
all RMS replacements. The CUDA source hash is part of deployment identity.

### Accepted M16 QK tiling

`qk_tile64=True` uses a 64×64 output tile for M1152/K128, preserving all four
K32 partial-dot additions. Other shapes retain 32×64.
[Bounded candidates, complete-route and video results](../../artifacts/video_runtime_v1/qk_tiles_v1/MODULE_DISTANCE.zh.md)
show bitwise outputs and 1.0025–1.0054× generation speedup; the QK module alone
is 1.33–1.35× faster. Use `--no-qk-tile64` for the previous output tiling.

### Exact AV accumulator prototype (not deployed)

`--exact-fold` changes the FP32 accumulation order while retaining both AV
products. [Bounded trials and paired short-video differences](../../artifacts/video_runtime_v1/exact_pipeline_v1/MODULE_DISTANCE.zh.md)
show 1.09–1.12× exact-kernel speedup, but only 0.5–0.7% generation improvement
and noticeable trajectory differences (LPIPS 0.141–0.237 versus the same-tier
original implementation). It remains off. These are implementation differences,
not evidence of quality regression relative to Dense.
