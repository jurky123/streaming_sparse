"""Report the PyTorch-order fused square/mean implementation."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/rms_mean_v1'
def main():
 module=json.loads((OUT/'module.json').read_text());rows=[]
 for tier in ('base','intersection','step'):
  a,b=[json.loads((OUT/f'video/{tier}_{v}/result.json').read_text()) for v in ('old','new')]
  za,zb=[torch.load(OUT/f'video/{tier}_{v}/latents.pt',weights_only=True) for v in ('old','new')]
  rows.append(dict(tier=tier,before_s=a['median_seconds'],after_s=b['median_seconds'],speedup=a['median_seconds']/b['median_seconds'],saved_ms=1000*(a['median_seconds']-b['median_seconds']),latent_bitwise=torch.equal(za,zb),latent_nmse=((za.float()-zb.float()).square().sum()/za.float().square().sum()).item(),counters_bitwise=a['runtime_stats']['counters']==b['runtime_stats']['counters'],before_cold_s=a['cold_seconds'],after_cold_s=b['cold_seconds']))
 assert module['passed'] and all(r['latent_bitwise'] and r['counters_bitwise'] for r in rows)
 summary=dict(rows=rows,module=module,passed=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'local_experiments/video_runtime_v1').glob('*.py')},cuda_sha256=hashlib.sha256((ROOT/'local_experiments/video_runtime_v1/csrc/rms_mean.cu').read_bytes()).hexdigest())
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 table='\n'.join(f"| {r['tier']} | {r['before_s']:.6f} | {r['after_s']:.6f} | {r['speedup']:.4f}× | {r['saved_ms']:.2f} |" for r in rows)
 mt='\n'.join(f"| {r['name'].replace('blocks.5.','')} | {r['shape'][1]} | {r['before_ms']:.6f} | {r['after_ms']:.6f} | {r['speedup']:.3f}× |" for r in module['rows'])
 (OUT/'MODULE_DISTANCE.zh.md').write_text(f'''# RMSNorm平方与原顺序mean融合

## 验收结果

在上一轮 `rms_dataflow` 基线上，本轮省去FP32平方数组。自定义CUDA均值kernel复现当前PyTorch的每线程累加及warp合并顺序；epsilon、rsqrt和末级BF16舍入/权重乘法仍使用上一轮路径。56组均值/最终输出契约以及三档短视频均逐位一致。默认启用 `rms_mean=True`，仍可 `--no-rms-mean` 回到上一轮。

| 档位 | 上轮生成/s | 本轮生成/s | 加速比 | 节省/ms |
|---|---:|---:|---:|---:|
{table}

city_walk/seed701，9chunks，105帧，768×512，16FPS，history360，tile72；GPU6/7/5分别测试三档。每档同GPU先旧后新，各完整预热后5次中位数；仅生成，排除VAE、编码、IO。latent和逻辑计数全部逐位一致，没有改变Full_Pset策略、权重或算法配置。block_pointwise及rms_dataflow保持启用，整块Graph仍关闭。无15s/30s或额外质量测试。

## 为什么这次允许融合归约

先核对安装版本PyTorch2.6.0+cu124（commit `2236df1770800ffea5697b11b0bb0d910b2e59e1`）的 `ATen/native/cuda/Reduce.cuh` 和 `SharedReduceOps.h`。原header及SHA保存在本目录，来源记录在 `reduction_provenance.json`；相关代码遵循 `csrc/PYTORCH_LICENSE.txt`。

实际profile确认D1536、4608行使用block `[32,16,1]`、grid `[288,1,1]`。每warp处理一行；512行文本K同样使用16行/CTA。当前shape下规则如下：

1. 每lane读取4元素向量，向量下标从lane开始，步长32，共12轮。
2. 每lane保留4个独立FP32累加器。读入BF16后转FP32并平方，再加到对应累加器。
3. 以 `((s0+s1)+s2)+s3` 顺序合并4个累加器。
4. warp shuffle按 **1、2、4、8、16递增偏移** 合并，不采用常见的递减树。
5. lane0乘FP32的 `1/1536`，写出均值。

CUDA实现用 `__fmul_rn` 和 `__fadd_rn` 固定乘加边界，NVCC `--fmad=false`。未改变累加精度；不存在从总量减去大项的消减。均值输出继续进入原 `torch.rsqrt(mean+eps)`，最终BF16舍入继续由上一轮 `_finish` 保留。

实现明确限定连续BF16输入、D1536、行数至少16；小于16行时PyTorch可能使用不同block/归约结构，因此不把当前证明推广到该形状。

## 模块数据与实测

复用上轮四个真实输入：layer5的self Q、self K、cross Q为4608行，cross K为512行。56组契约覆盖四类×真实/随机/小/大/零/极小/极大七种数值×两种epsilon；比较FP32均值及最终BF16输出，全部逐位一致。原均值直接使用 `x.float().pow(2).mean(...)`。

主测模块为完整RMSNorm，包含epsilon、rsqrt及finish；CUDA Graph预热后交替30轮×20次，最终整理版测试在GPU0完成。

| 模块 | 行数 | 上轮/ms | 本轮/ms | 加速比 |
|---|---:|---:|---:|---:|
{mt}

新CUDA均值kernel为30寄存器、0shared、0stack、0local，证据 `cuda_resources.txt`。没有进行warps/register组合筛查。

## 优化后profile复核

新完整chunk中，450次均值kernel合计4.539ms，450次finish合计5.251ms。RMS GPU annotation总区间14.137ms，上轮相同口径35.844ms，相差21.707ms/chunk；乘9chunks约0.195s，与视频节省0.177–0.207s接近。区间包含保留的epsilon/rsqrt及调度，不与kernel事件重复累计。

finish本身没有改变，但从9.235降到5.251ms；这说明不再写读大平方数组也改善了后续数据的缓存状态。该归因是基于同kernel/不同数据流的推断，没有将这部分下降计成新的算术减少。原始事件保存在 `accepted_profile/detail.trace.json` 和 `kernel_totals.json`。

## 工作量、流量与距离

上一轮每个元素需平方阶段读2写4、mean读4、finish读2写2，共14字节；本轮均值kernel读2、finish读2写2，共6字节。小均值/epsilon/inv数组及可缓存权重另外存在，不能等同零成本。

| 每个4608×1536 RMSNorm | 上轮 | 本轮 |
|---|---:|---:|
| 大数组逻辑读写 | 99.090MB | 42.467MB |
| 大数组读写减少 | — | 57.14% |
| 按1266.396GB/s复制带宽换算 | 0.0782ms | 0.0335ms |
| 主路径kernel数 | 5 | 4 |

省掉的是FP32平方数组的写入和再次读取，约56.623MB/模块。乘方和求和的主算术量不变。流量参考不是硬件下限：本轮的输入和输出更容易命中L2，重复调用时可能快于该HBM参考。不能把这些逻辑字节直接当成Nsight测得的HBM字节，也不能用大GEMM峰值估算归约速度。

## 实现位置与配置

- `local_experiments/video_runtime_v1/csrc/rms_mean.cu`：固定树平方/均值kernel，含原PyTorch顺序说明。
- `rms_mean.py::mean_square/fused`：NVCC缓存、当前CUDA stream发射、均值后处理与原finish复用。源码SHA进入deployment identity，`.cu`进入benchmark快照。
- `rms_dataflow.py::RMSDataflow`：选择上轮或本轮实现；`deployment.py::make_runtime`统一benchmark/worker。
- `accepted_tiers.py --rms-mean / --no-rms-mean`：相对上轮开关；`--no-rms-dataflow`关闭整项RMS优化，回到原模型实现。
- `check_rms_mean.py`：中间均值/输出契约和模块计时。
- `profile_retained.py --rms-mean`：新完整chunk profile，保存于 `accepted_profile/`。
- `report_rms_mean.py`：报告与汇总。

编译目标sm80，NVCC12.2，快速缓存 `/tmp/streaming_sparse-0/rms_mean/<source-sha>/kernel.so`，并发worker通过文件锁复用编译产物。原始逐次视频时间及源码位于 `video/*/result.json` 和 `source/`。

```bash
CUDA_VISIBLE_DEVICES=6 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.video_runtime_v1.accepted_tiers --tier base --rms-mean --reps 5 --skip-profile --local-assets /tmp/streaming_sparse-0/runtime --output <fresh-dir>
```

对照使用 `--no-rms-mean`，其他配置相同。后续优化仍需根据新profile定位；本轮不把未经测试的其他维度、归约树或近似计算作为等价部署路径。
''')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
