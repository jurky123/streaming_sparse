"""Summarize RMSNorm dataflow work, exact contracts, and paired generation."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/rms_dataflow_v1'
def main():
 module=json.loads((OUT/'module.json').read_text());rows=[]
 for tier in ('base','intersection','step'):
  a,b=[json.loads((OUT/f'video/{tier}_{v}/result.json').read_text()) for v in ('old','new')]
  za,zb=[torch.load(OUT/f'video/{tier}_{v}/latents.pt',weights_only=True) for v in ('old','new')]
  rows.append(dict(tier=tier,before_s=a['median_seconds'],after_s=b['median_seconds'],speedup=a['median_seconds']/b['median_seconds'],saved_ms=1000*(a['median_seconds']-b['median_seconds']),latent_bitwise=torch.equal(za,zb),latent_nmse=((za.float()-zb.float()).square().sum()/za.float().square().sum()).item(),counters_bitwise=a['runtime_stats']['counters']==b['runtime_stats']['counters'],before_cold_s=a['cold_seconds'],after_cold_s=b['cold_seconds']))
 assert module['passed'] and all(r['latent_bitwise'] and r['counters_bitwise'] for r in rows)
 summary=dict(rows=rows,module=module,passed=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'local_experiments/video_runtime_v1').glob('*.py')})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 table='\n'.join(f"| {r['tier']} | {r['before_s']:.6f} | {r['after_s']:.6f} | {r['speedup']:.4f}× | {r['saved_ms']:.2f} |" for r in rows)
 mt='\n'.join(f"| {r['name'].replace('blocks.5.','')} | {r['shape'][1]} | {r['before_ms']:.6f} | {r['after_ms']:.6f} | {r['speedup']:.3f}× |" for r in module['rows'])
 (OUT/'MODULE_DISTANCE.zh.md').write_text(f'''# RMSNorm 前后数据流融合：保留原均值归约

## 结果

本轮只消除RMSNorm的转换、平方、缩放中间张量；FP32 mean、加epsilon、rsqrt继续调用原PyTorch算子。40组模块契约的平方、均值、最终输出全部逐位一致，三档9-chunk视频latent及逻辑计数也逐位一致。当前三档共有默认加入 `rms_dataflow=True`，保留 `--no-rms-dataflow` 对照。上一轮 `block_pointwise=True` 保持，整块CUDA Graph继续关闭。

| 档位 | 原生成/s | 融合后生成/s | 加速比 | 节省/ms |
|---|---:|---:|---:|---:|
{table}

city_walk/seed701，9chunks，105帧，768×512，16FPS，history360，tile72；每档同一GPU先原路径后新路径，完整视频预热后各5次中位数。仅生成，不含VAE/编码/IO。三档分别GPU6/7/5。无15s/30s测试，无算法相对Dense质量重测。

## 热点依据与真实形状

`capture_rms_dataflow.py` 对当前部署的原RMSNorm加标签。满窗口一次noisy step，30层self Q、self K、cross Q三类分别约6.607、6.553、6.606ms，合计19.766ms。各类每层一次，输入 `[1,4608,1536]`，BF16输入和权重。cross K输入 `[1,512,1536]`，每视频文本缓存初始化时计算，不属于每step热路径。

`profile.json` 含CPU/GPU同名annotation，不能重复相加；上述数值取CPU标签所归属的设备时间。原始trace与真实输入 `input.pt` 都保留。

## 算法与舍入保持

原表达式为：

```python
f = x.float()
inv = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
out = (f * inv).bfloat16() * weight
```

新执行顺序：

1. `_square` 从BF16读取x，提升FP32并平方，直接写FP32平方数组，省掉完整f数组。
2. 对完全相同的FP32平方数组，执行原来的PyTorch mean、epsilon相加、rsqrt。形状、布局和归约实现不变。
3. `_finish` 再读取BF16 x，提升FP32乘inv，**先舍入到BF16，再提升FP32乘BF16 weight，最后写回BF16**。

两个kernel均禁用FMA融合；不改epsilon、权重或算术精度。舍入位置与原路径相同。完整均值归约不换成另一棵树；这也是与直接采用融合RMSNorm实现的区别。

## 模块隔离结果

真实四类输入，完整预热后CUDA Graph交替30轮，每轮20次，报告模块中位数。不是CPU墙钟，也不是按模块加速比直接推算视频加速。

| 模块 | token数 | 原模块/ms | 融合/ms | 加速比 |
|---|---:|---:|---:|---:|
{mt}

40组契约：四类输入×真实/随机/小/大/零五种数值×原epsilon与1e-5两种参数。逐项验证平方、均值与最终输出；均逐位一致。代码生成的寄存器、spill、shared信息与PTX在 `module.json`、`square.ptx`、`finish.ptx`。

## 新profile与整段收益复核

完整chunk profile中，新 `_square` 450次合计12.035ms、`_finish` 450次合计9.235ms。RMS GPU annotation总区间35.844ms，包含保留的mean、epsilon、rsqrt及调度间隙；实际kernel计数单独保存在 `accepted_profile/kernel_totals.json`，不与annotation重复相加。

原单step19.766ms按同形状5次forward估算约98.832ms/chunk，与新值差约62.99ms；乘9chunks约0.567s，与视频实测节省0.575–0.582s接近。这是独立profile的近似归因，说明收益主要来自目标模块，不能当作严格的逐事件分解。

## 理论工作量与剩余空间

不计可缓存的小权重及逐行inv，原路径每元素的大数组读写为36字节：BF16→FP32转换6、平方8、mean读取4、FP32缩放8、转回BF16为6、权重乘法4。新路径为14字节：平方6、mean读取4、最终缩放与权重乘法4。

| 每个4608×1536 RMSNorm | 原路径 | 新路径 |
|---|---:|---:|
| 大数组逻辑读写 | 254.804MB | 99.090MB |
| 以1266.396GB/s复制带宽换算 | 0.2012ms | 0.0782ms |
| 大数组读写减少 | — | 61.11% |
| 主要kernel次数 | 8 | 5 |

新主路径实测约0.078–0.083ms，接近此流量参考。但它不是严格HBM下限：平方数组约28.3MB，可部分命中40MB L2，且mean归约、调度和小数组处理仍有成本。不能再把相对大GEMM峰值的差距当作尚未兑现的算术收益。

后续空间集中在剩余FP32平方数组的写入与读取。如果尝试进一步融合平方和mean，必须先复现当前PyTorch归约树并单独验证，不能沿用本轮“原mean完全不变”的数值结论。本轮不做block/warps大范围筛查。

## 代码位置、入口及证据

- `local_experiments/video_runtime_v1/rms_dataflow.py`：`_square`、`_finish`、`reference`、`fused`与`RMSDataflow`实例包装。只包装WanRMSNorm，其他归一化不变。
- `deployment.py::make_runtime`：统一初始化；`runtime.py::close`恢复原forward；worker代码identity包含该实现。
- `accepted_tiers.py --rms-dataflow / --no-rms-dataflow`：复现实验开关。
- `capture_rms_dataflow.py`：原路径profile及真实输入；`check_rms_dataflow.py`：中间值契约及隔离计时。
- `profile_retained.py --rms-dataflow`：新实现完整chunk profile，保存于 `accepted_profile/`。
- `report_rms_dataflow.py`：本报告及汇总生成器。

`video/*/result.json` 保存每次计时、配置、GPU逻辑计数及源码哈希；相邻 `source/` 保留实验源码，`latents.pt`为数值验收输入。报告比较同档当前实现，Full_Pset策略和摘要算法均未改。

```bash
CUDA_VISIBLE_DEVICES=6 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.video_runtime_v1.accepted_tiers --tier base --rms-dataflow --reps 5 --skip-profile --local-assets /tmp/streaming_sparse-0/runtime --output <fresh-dir>
```

原路径替换为 `--no-rms-dataflow`；其他两档使用 `--tier intersection` 或 `--tier step`。
''')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
