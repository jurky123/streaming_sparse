"""Summarize completed matched short/15-second measurements, without mixing controls."""
import json
from pathlib import Path
from .weights import ROOT
from ..learned_cache.io import atomic

def run():
    base=ROOT/'artifacts/video_runtime_v1/formal_short_15';rows=[];quality=[]
    for root in (base/'short',ROOT/'artifacts/video_runtime_v1/formal_15_30/15s'):
        r=json.loads((root/'comparison.json').read_text());seconds=r['seconds']
        for method,p in r['performance'].items():rows.append(dict(seconds=seconds,method=method,**{k:v for k,v in p.items() if k!="seconds"}))
        for pair,windows in r['quality'].items():
            for window,q in windows.items():quality.append(dict(seconds=seconds,pair=pair,window=window,**{k:q[k] for k in ('lpips','ssim','psnr')}))
    atomic(base/'summary.json',dict(performance=rows,quality=quality))
    lines=['# 短视频／15秒正式验收','','city_walk/seed701，768×512，16FPS；9/21个生成chunks，解码105/240帧；双方360页历史，即5个历史chunks。全覆盖C和新Full_Pset seed17，AdaptivePhi/B16配套冻结；Dense不使用Reader。只测试同一个视频的两种长度。','','## 耗时与相对Dense速度','','每个配置完整预热一次后计时两次。VAE另行完整预热并计时两次；总计为两阶段中位数之和，不含模型加载、首次编译、视频编码或质量计算。Dense是原生Engine SDPA，无稀疏分组／摘要提交；旧dense目录仅保留为同运行时控制，不作加速比分母。','','|时长|方法|生成秒|VAE秒|合计秒|生成加速比|生成+VAE加速比|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['seconds']}s|{r['method']}|{r['generation_seconds']:.3f}|{r['median_seconds']:.3f}|{r['generation_plus_decode_seconds']:.3f}|{r['generation_speedup_vs_dense']:.3f}×|{r['generation_vae_speedup_vs_dense']:.3f}×|")
    lines+=['','original=整路由编译+原补偿；optimized=同权重+补偿编译+按需回退的FP64动作kernel。小于1表示慢于Dense。','','## RGB一致性','','指标计算使用未压缩RGB；PSNR全分辨率，LPIPS/SSIM在256×384。短视频窗口21:101；15s窗口21:101、80:160、160:240。','','|时长|比较|段|LPIPS↓|SSIM↑|PSNR↑|','|---|---|---|---:|---:|---:|']
    for r in quality:lines.append(f"|{r['seconds']}s|{r['pair']}|{r['window']}|{r['lpips']:.5f}|{r['ssim']:.5f}|{r['psnr']:.3f}|")
    lines+=['','## 配置和训练解释','','此前338页、9chunks的消融不与本表混算。原训练数据的历史上限是338页，此处按统一窗口协议扩展为360页，存在轻微部署分布变化。新Reader的开发集收益和此前受控短视频训练对照见 QUALITY_AND_TRAINING.zh.md；本表不是训练前后的因果对照。原/融合补偿浮点差异及离散路由可能导致轨迹分岔，相对Dense指标包含构图/运动偏移，不能直接解释为纯画质损坏。','','原始结果、权重和代码SHA、latent、视频、帧缓存与并排视频均位于 artifacts/video_runtime_v1/formal_short_15（短视频）和 formal_15_30/15s（已完成15秒数据保留原目录）。每组仅两次计时，支持本机同视频描述性比较。']
    dest=ROOT/'docs/research/evidence/video_runtime_v1/FORMAL_SHORT_15.zh.md';dest.write_text('\n'.join(lines)+'\n');print(dest)
if __name__=='__main__':run()
