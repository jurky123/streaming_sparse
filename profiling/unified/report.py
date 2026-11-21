"""Join same-input A100 SXM4 generator timings with full decoded-frame metrics."""
import csv,json,math
from statistics import mean
from pathlib import Path

folder=Path(__file__).resolve().parent
clean=folder.parent/'clean_forward'
def load(path):return json.loads(path.read_text())
rows=[]
for n,seconds in [(7,5),(41,30)]:
    key=str(n)
    frames=80 if n==7 else 480
    baseline=mean(load(clean/'dense_before_times.json')[key][1:])
    quality=load(clean/('quality_5s.json' if n==7 else 'quality.json'))
    for method,label in [('dense','Dense'),('base','Base'),('intersection_fast','Intersection fast'),('lightforcing','LightForcing'),('forcingkv','Forcing-KV'),('svgear','SVG-EAR'),('solattn','Sol-Attn'),('pisa','PISA（流式适配，Q 补齐）'),('dasa','DASA')]:
        if method=='dense':
            times=load(clean/'dense_before_times.json')[key]
            q={'psnr':math.inf,'lpips':0.,'ssim':1.}
        elif method in ('base','intersection_fast'):
            times=load(clean/f'{method}_after_times.json')[key]
            q=quality[method]['after_vs_dense'][str(frames)]
        else:
            if not (folder/f'{method}_{n}.quality.json').exists():continue
            times=load(folder/f'{method}_{n}.json')['times']
            q=load(folder/f'{method}_{n}.quality.json')
        assert len(times)==3
        t=mean(times[1:])
        rows.append(dict(seconds=seconds,method=label,wall=t,speedup=baseline/t,**q,hot1=times[1],hot2=times[2]))
with (folder/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
lines=['# 前两块 Dense、首帧精算和时间 RoPE 重定位对比','',
       'A100 SXM4 80GB，768×512，16 fps，short_721 咖啡场景，seed 701，同一 Self Forcing 1.3B EMA、BF16、缓存文本嵌入及分块噪声。每项预热一次，取两次热运行的均值。生成计时含 clean KV 提交、滚窗及重定位，排除模型加载、真实 VAE 解码、视频编码和评分。全片 80/480 帧对同输入 Dense 评分，PSNR 用原分辨率，LPIPS-Alex 和 SSIM 在 256×384 上逐帧计算。','',
       '| 时长 | 方法 | 生成耗时（s） | 相对 Dense | PSNR（dB） | LPIPS ↓ | SSIM ↑ |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['seconds']} s | {r['method']} | {r['wall']:.2f} | {r['speedup']:.2f}× | {r['psnr']:.2f} | {r['lpips']:.4f} | {r['ssim']:.4f} |")
lines+=['','## 执行口径','',
        '- Base、intersection_fast 使用原权重、稀疏 clean 实验配置。外部方法保留各自后续的 clean、层覆盖和压缩规则。所有方法强制前两块全量 attention，实际执行的查询均纳入完整首帧。以 18 帧名义窗口确定首帧的时间坐标，分头或按区域压缩的方法保留自己的较短历史。',
        '- LightForcing 走官方 pipeline，按长短选择官方预算；统一 18 帧窗，用官方长窗路径计算相对 RoPE，SM80 top-k 内优先保留首帧所有块。Forcing-KV、DASA 走上游 pipeline，接入兼容的共用 FFN 和 RMSNorm。Forcing-KV 保留分头压缩；DASA 保留区域复用，按时间移除超窗历史。',
        '- PISA、Sol-Attn、SVG-EAR 使用共同优化主干、第 0 层及 clean 沿用历史 dense 策略。PISA 精算块数取 24 与名义 10% 预算向下取整后的较大值，优先保留完整首帧；Sol-Attn 保留原有当前块 sink，同时将首帧加入 exact；SVG-EAR 保留 1,000 个 KV 簇，把首帧独立成一个强制精算簇。PISA、Sol-Attn 沿用历史 Q 补齐和裁剪；SVG-EAR 用矩形 Q/KV。满窗时 PISA 将 3 帧 Q 补至 18 帧，输出裁回 3 帧。尚无原生短 Q 的配对计时。',
        '- 官方 pipeline 的转移噪声预制，共同主干逐块按相同公式取噪声；以各实现实际生成时间比较。单一场景和 seed。']
(folder/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
