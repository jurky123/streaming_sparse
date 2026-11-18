"""Bounded QK tiling results with full-route and short-video acceptance."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/qk_tiles_v1'
def main():
 module=json.loads((OUT/'module.json').read_text());route=json.loads((OUT/'route.json').read_text());rows=[]
 for tier in ('base','intersection','step'):
  a,b=[json.loads((OUT/f'video/{tier}_{v}/result.json').read_text()) for v in ('old','new')]
  za,zb=[torch.load(OUT/f'video/{tier}_{v}/latents.pt',weights_only=True) for v in ('old','new')]
  rows.append(dict(tier=tier,before_s=a['median_seconds'],after_s=b['median_seconds'],speedup=a['median_seconds']/b['median_seconds'],saved_ms=1000*(a['median_seconds']-b['median_seconds']),latent_bitwise=torch.equal(za,zb),latent_nmse=((za.float()-zb.float()).square().sum()/za.float().square().sum()).item(),counters_bitwise=a['runtime_stats']['counters']==b['runtime_stats']['counters']))
 assert all(v['bitwise'] for r in module['rows'] for v in r['variants']) and all(all(r['bitwise']) for r in route['rows'])
 assert all(r['latent_bitwise'] and r['counters_bitwise'] for r in rows)
 summary=dict(rows=rows,module=module,route=route,passed=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'local_experiments/video_runtime_v1').glob('*.py')})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 vt='\n'.join(f"| {r['tier']} | {r['before_s']:.6f} | {r['after_s']:.6f} | {r['speedup']:.4f}× | {r['saved_ms']:.2f} |" for r in rows)
 mt='\n'.join(f"| {r['pages']} | {r['variants'][0]['ms']:.6f} | {r['variants'][1]['ms']:.6f} | {r['variants'][1]['speedup']:.3f}× | {r['variants'][2]['ms']:.6f} |" for r in module['rows'])
 rt='\n'.join(f"| {r['mode']} | {r['before_ms']:.6f} | {r['after_ms']:.6f} | {r['speedup']:.4f}× |" for r in route['rows'])
 (OUT/'MODULE_DISTANCE.zh.md').write_text(f'''# M16 QK输出分块：按流量与完整路由收益验收

## 结果

在前几轮共有主干优化之后，本轮将M16 QK的输出分块由32×64改为64×64。**Q64、M16、K32分段、BF16输入、FP32输出和累加顺序均不变。** 四种页面数、三种完整路由及三档短视频输出逐位一致。部署开关 `qk_tile64`，可用 `--no-qk-tile64` 复现旧路径。

| 档位 | 原生成/s | 新生成/s | 加速比 | 节省/ms |
|---|---:|---:|---:|---:|
{vt}

city_walk/seed701，9chunks/105帧，768×512，16FPS，history360/tile72。三档分别在GPU6/7/5，同GPU先旧后新，各完整预热后5次中位数；只计生成，排除VAE、编码、IO。latent及逻辑计数全部逐位一致。未运行15s/30s或重新评价算法相对Dense误差。

## 热点归因与理论口径

最新profile中通用名 `Kernel` 合计约38.2ms/chunk，实际有两类：grid `[36,27,12]` 是M16 QK，grid `[864,27,1]` 是Risk归约。二者不能都归入QK。QK原约0.155–0.162ms/次，输出 `[12,1152,1728]` FP32。

P432的QK主算术6.115GFLOPs；以约0.155ms估算约39TFLOP/s。但每次必须写95.551MB logits，输入A/B还分别7.078/10.617MB。因此：

- 用既有202.135TFLOP/s大BF16 GEMM参考，只得到0.0303ms算术参考。
- 一次读取A/B并写C的113.246MB，按1266.396GB/s复制带宽换算约0.0894ms。
- 这两项参考不能相加，也不是完整硬件下限。只引用39/202会夸大可回收差距。

本轮扩大M方向输出分块，让同一K页在一个CTA中服务更多probe行，CTA从11664减到5832。理论重复输入逻辑读取从约573.309MB减到382.206MB，减少三分之一；其中大部分可能命中缓存，不能称为减少191MB HBM流量。输出95.551MB不变。

## 有限筛查与算术保持

仅试两个候选：64×64及32×128，均4warps、BK32。没有扩展成多维参数搜索。原代码按K128分成四个K32 dot，分别形成partial再按原顺序相加；仅修改BM/BN，不把K32改为K64/K128，也不改变dot累加表达式。

真实输入来自此前保存的base/layer15/chunk8/step0；之后所有已部署更改均通过latent逐位一致验收。P216/288/360使用同一输入前缀，P432为完整真实输入。每种完整输出逐位检查，CUDA Graph预热后30轮交替顺序，每轮20次。

| 页面数 | 32×64/ms | 64×64/ms | 加速比 | 32×128/ms |
|---|---:|---:|---:|---:|
{mt}

64×64更优，保留；32×128不部署。寄存器/shared分别为80/30720B、96/40960B、139/51200B；均0spill。较大分块并未提升理论occupancy，收益来自数据复用和CTA组织，不能归因于“占用率更高”。

启用条件限定M=1152、K=128（当前tile72×M16）；其他形状继续32×64。原始输入已可由BF16精确表示，不引入新的输入量化。

## 完整路由

复用真实P432 Q/summary，使用当前C_s17与Full_Pset_s17配套权重重新编码页面。配置与部署一致：M16入口、projected Risk、cached logmass、Reader split/压缩、endpoint denominator等。

| 路由分支 | 原/ms | 新/ms | 加速比 |
|---|---:|---:|---:|
{rt}

三种分别为关闭修正、head [4,3,9]修正、全head修正；对应三档在修正层/step执行的分支，不把分支微基准当成整视频比例。输出ids、logits、mask、risk、stats五项全部逐位一致。完整路由每次约节省0.041–0.045ms，与单QK节省一致。

## 新profile复核

按kernel的grid区分QK与Risk后，当前完整chunk的116次QK从17.967ms降到13.114ms，节省4.854ms；Risk从20.280到19.951ms，基本维持同量级。不能把二者共有的通用 `Kernel` 名称整项记成QK加速。完整路由与视频结果均显示净收益，但端到端仅约0.25–0.54%，远小于模块1.35×。

默认启用64×64，保留32×64关闭开关。后续不继续铺开QK参数扫描；输出95.6MB与Risk消费仍存在，下一轮根据新的profile及精算kernel硬件执行情况重新排序。

## 实现位置与复现

- `local_experiments/video_runtime_v1/tensor_bmm.py::bf16_inputs_bmm`：根据 `QK_TILE64` 选择输出BM；同一个原kernel，K32与累加表达式未改。
- `deployment.py::configure`：统一配置；worker identity包含tensor_bmm.py。
- `accepted_tiers.py --qk-tile64 / --no-qk-tile64`：视频对照。
- `check_qk_tiles.py`：四页面数、三个固定配置的模块测试。
- `check_qk_route.py`：配套权重下三个路由分支的完整输出和速度测试。
- `profile_retained.py --qk-tile64`：新的完整chunk profile，位于 `accepted_profile/`。
- `report_qk_tiles.py`：本报告及汇总。

`module.json`、`route.json`保留逐次计时和契约；`video/*/result.json`、`source/`、`latents.pt`保留视频结果、配置和源码证据。

```bash
CUDA_VISIBLE_DEVICES=6 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.video_runtime_v1.accepted_tiers --tier base --qk-tile64 --reps 5 --skip-profile --local-assets /tmp/streaming_sparse-0/runtime --output <fresh-dir>
```

其他两档替换tier。这里没有将“模块1.35×”宣传为整视频加速，最终收益以上表端到端生成计时为准。
''')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
