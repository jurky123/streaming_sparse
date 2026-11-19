from pathlib import Path
import json, statistics as st, math
root=Path(__file__).resolve().parent
def median(name):
    d=json.loads((root/f'{name}.json').read_text())
    if 'wall' in d: return st.median(d[k] for k in ('wall','wall2','wall3'))
    assert len(d['times'])==4, name
    return st.median(d['times'][1:])
def pair(name,cards=None):
    return [median(f'{name}_{s}'+(f'_gpu{cards[s]}' if cards else '')) for s in range(2)]
def ratio(a,b):return math.sqrt((a[0]/b[0])*(a[1]/b[1]))
def fmt(x):return ' / '.join(f'{i:.3f}' for i in x)
rows=[]
for name,label in [('base','base'),('intersection','intersection'),('step','step'),('table_10','条件 intersection 10%'),('table_15','条件 intersection 15%')]:
    tiers=name in ('base','intersection','step')
    cards=(4,5) if tiers else (0,1)
    dense=pair('native',cards if tiers else None)
    official=pair('light12',cards)
    matched=pair('light',cards if tiers else None)
    t=pair(name)
    rows.append(f'| {label} | {fmt(t)} | {ratio(dense,t):.3f}× | {ratio(official,t):.3f}× | {ratio(matched,t):.3f}× |')
d=pair('native'); o=pair('light12',(0,1)); m=pair('light');old=pair('legacy')
report=f"""# 30 秒视频生成速度复测

两条输入为东京街景 long_0/701、雪地猛犸 long_1/702。每个方法独占一张 A100 40 GB，以相同 Self Forcing EMA、BF16、缓存文本嵌入及对应的分块噪声生成，768×512，41 个 chunk，输出对应 480 帧、16 fps。时间从生成开始到 latent 完成并同步 GPU，排除模型加载、文本编码、VAE 和 MP4。LightForcing 的噪声在计时前备好，本项目在生成中取噪声，与各自 VBench 入口一致。

## 配对结果

时间顺序为东京 / 猛犸，单位秒。每个输入取三轮热运行中位数。三档部署预设和两档实验预算各用对应 GPU 上测得的 Dense 与 LightForcing 作分子，表内加速比为两个输入配对比值的几何平均。大于 1 表示本项目更快。Dense 是优化主干上的原生 18 帧滑窗。

| 实现 | 生成时间（秒） | 对 Dense | 对 LightForcing 官方长窗 12 | 对 LightForcing 同窗 18 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

部署三档直接使用 accepted_tiers 预设。两档实验方案使用首帧重定位、window/step/layer/head 条件配额和原始权重，10% / 15% 的分母均为首帧以外的页。实验行单独预热 7 个 chunk，其余方法排除首轮完整生成。三档部署参照在 GPU 4/5 上的 Dense、官方长窗、同窗中位数依次是 {fmt(pair('native',(4,5)))}、{fmt(pair('light12',(4,5)))}、{fmt(pair('light',(4,5)))} 秒；实验两档在 GPU 0/1 的三个参照分别为 {fmt(d)}、{fmt(o)}、{fmt(m)} 秒。

## 旧 LightForcing 为什么慢

归档旧入口固定 18 帧并清空 efficient_deployment。两条长视频均实际调用 SM80 稀疏内核 6,150 次，Dense 回退为零。计时为 {fmt(old)} 秒。恢复官方 RoPE、scale-shift 内核，在相同 18 帧下用时 {fmt(m)} 秒（相对旧入口 {ratio(old,m):.3f}×）。继续使用官方长视频 12 帧窗口，用时 {fmt(o)} 秒。旧数据不能代表修正后的官方配置。额外对照在修复后的 18 帧 LightForcing 主干上强制改用 Dense attention，用时 {fmt(pair('lfdense'))} 秒。

在 GPU 0/1 上，官方长窗版相对本项目原生 Dense 为 {ratio(d,o):.3f}×，18 帧同窗为 {ratio(d,m):.3f}×。官方短视频配置另在 7 个 chunk 上验证，短窗 21，LightForcing 用时 {fmt(pair('short'))} 秒，原生 Dense {fmt(pair('native_short'))} 秒，对应 {ratio(pair('native_short'),pair('short')):.3f}×。短视频每轮实际稀疏调用 1,050 次。

## 实现与范围

VBench 的 light.py 按视频长短加载对应官方 YAML，计算对应窗口下的稀疏率，保留可用的上游优化；worker.py 按任务长度构建管线并在 SM80 检查底层稀疏核调用。A100 测试使用 BF16，环境没有 sgl_kernel，RMSNorm 使用常规实现。修复后的长视频每轮 6,150 次底层稀疏调用、零 Dense 回退。此次只测速度，不涉及预算、权重切换或画质结论。历史 VBench 作业和视频仍对应原入口。另用隔离队列测试实际 worker，同一个进程依次完成短、长任务，成功切换配置、解码并写入 MP4（worker_smoke/mixed.log）。

同目录内保留旧 light.py 副本 legacy_light.py、测量脚本、逐轮时间 JSON 以及配置日志。表中各行来自 densebench.py、tablebench.py、lfbench.py；报告由 report.py 汇总。
"""
(root/'RESULTS.zh.md').write_text(report)
print(report)
