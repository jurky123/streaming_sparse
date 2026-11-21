"""Report paired video changes and local output error by step, layer, and head."""
import csv
import json
from pathlib import Path
import numpy as np

folder=Path(__file__).resolve().parent
names=dict(full='完整 intersection_fast',base='Base（独立配置对照）',mean='均值 K/V 摘要',mean_tail='仅补偿读出换均值',no_tail='关闭未精算页补偿',mass='估计概率质量选页',set_off='intersection_fast 保留条件配额，仅关闭集合纠偏',ungrouped='自然顺序分页',uniform='统一总预算和分配',whole_window='每次对全窗 token 重排、分组并重算摘要')
def arrays(case):
    rows=json.loads((folder/f'sparse_{case}.json').read_text())
    assert len(rows)==4
    return ([(r['prompt'],r['seed']) for r in rows],
            np.array([[r[k] for k in ('seconds','mse','lpips')] for r in rows]),
            np.load(folder/f'sparse_{case}_local.npy'))
cases,base,ref=arrays('full')
lines=['# intersection_fast 主消融表','',
'5 秒，80 帧，512×768，A100 SXM4，city_walk 和 short_721，各 seed 721、722。所有变体使用原权重、首帧锚点、RoPE 重定位、前两块 dense 和 sparse clean。生成耗时取四个已预热的配对生成的均值，排除 VAE 与评分。视频 LPIPS 对照同输入 dense 全部 80 帧，latent MSE 对照第 4–7 个 chunk。局部采样每 64 个 Q 取一行，与本次调用相同 Q/K/V 的 dense 输出比较，回填 dense 输出以隔离后续稀疏误差。NMSE 为四条轨迹 SSE 总和除以 dense 输出能量总和。自然分页在滚窗处可能改变保留的 token；其局部参照使用本变体的 KV。','',
'| 变体 | 局部 NMSE | 相对变化 | latent MSE | LPIPS | LPIPS 配对变化 | LPIPS 改善条数 | 秒 | 耗时变化 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
def nmse(a):return a[...,0].sum()/a[...,1].sum()
with (folder/'cells.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['variant','step','layer','head','nmse','baseline_nmse','delta_nmse'])
    for case,name in names.items():
        keys,a,local=arrays(case)
        assert keys==cases
        t,m,p=a.mean(0)
        ratio=nmse(local)
        lines.append(f'| {name} | {ratio:.6f} | {100*(ratio/nmse(ref)-1):+.2f}% | {m:.5f} | {p:.5f} | {p-base[:,2].mean():+.5f} | {(a[:,2]<base[:,2]).sum()}/4 | {t:.3f} | {100*(t/base[:,0].mean()-1):+.1f}% |')
        for s,l,h in np.argwhere(ref[...,1]>0):
            cell=local[s,l,h];reference=ref[s,l,h];x=cell[0]/cell[1];y=reference[0]/reference[1]
            w.writerow([case,s,l,h,x,y,x-y])
# old and base are the paired correction-on and correction-off scopes here.
paired_path=folder.parent/'conditional_scope/neighbors/paired.csv'
with paired_path.open() as source:
    paired=list(csv.DictReader(source))
def correction_rows(method):
    rows=sorted((r for r in paired if r['method']==method),key=lambda r:(r['prompt'],r['seed']))
    keys=[(r['prompt'],r['seed']) for r in rows]
    assert len(set(keys))==len(rows)==20
    return keys,np.array([[float(r[k]) for k in ('LPIPS','SSIM','PSNR','seconds')] for r in rows])
keys,on=correction_rows('old')
other,off=correction_rows('base')
assert keys==other
lines+=['','## 主消融表 · 集合纠偏的跨场景配对结果','',
        '5 个留出场景，各用 seed 703–706，每条生成 9 个 chunk。以同卡同输入 dense 为参照，对第 21–100 帧评分。表中差值按开启减关闭计算。','',
        '| 条件配额下的集合纠偏 | LPIPS ↓ | SSIM ↑ | PSNR（dB）↑ | 生成耗时（秒）↓ |',
        '|---|---:|---:|---:|---:|']
for label,values in [('关闭',off),('开启旧交集',on),('开启减关闭',on-off)]:
    lp,ss,ps,t=values.mean(0)
    lines.append(f'| {label} | {lp:.5f} | {ss:.5f} | {ps:.3f} | {t:.3f} |')
lines+=['',f'开启后 LPIPS 在 {sum(on[:,0]<off[:,0])}/20 条视频上降低。[逐样本记录](../conditional_scope/neighbors/paired.csv)']
lines+=['','## 自然分页与未精算页补偿 · 20 条独立配对','',
        '相同的 5 个场景和种子，RGB [21,101)。两个实验分别在同一卡上配对重跑完整模型。','',
        '| 实验 | 配置 | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ |','|---|---|---:|---:|---:|']
for directory,title,alt,label in [('natural_validation','分页','ungrouped','自然分页'),('compensation_validation','补偿','no_tail','关闭补偿')]:
    for case,description in [('full','完整模型'),(alt,label)]:
        records=[json.loads(f.read_text()) for f in sorted((folder/directory/case).glob('*.json'))]
        assert len(records)==20
        lp,ss,ps=np.array([[r[k] for k in ('lpips','ssim','psnr')] for r in records]).mean(0)
        lines.append(f'| {title} | {description} | {lp:.5f} | {ss:.5f} | {ps:.3f} |')
lines+=['','[分页保留集合、真实注意力质量及逐场景分析](PAGING.zh.md)']
lines+=['','## 干预口径','',
'- 均值 K/V：一个物理页四个代表 K 全部设为原始 64 个 K 的均值，读出用 64 个 V 的均值，rank-8 细节置零。保留统计特征和原权重。另测只改未选页读出的对照，局部实验的精算页与原路由一致。均值组沿用原摘要生成并覆盖字段，只读出对照也包含诊断用张量重建，耗时不可用来估计均值专用 kernel 的速度。',
'- 概率质量：对 4 个代表 K 的 logits 做 logsumexp，跨页 softmax，平均 16 个查询探针的页概率后排名，保留原配额、锚点和合法交换。',
'- 集合决策关闭：清空获准执行换页的层、头、步集合。7 层 × 3 头只在首个去噪步开放，完整配置的 clean 已经关闭换页。',
'- 自然分页：clean KV 按原 token 顺序每 64 个成页。页数与淘汰槽位规则维持一致，但部分 chunk 淘汰时物理槽位保留的 token 可随分组而改变。',
'- 预算：非锚点页数 P 的统一预算为 floor(P/10)，当前取 round(0.42 × 预算)，历史取余下部分。42% 按短片各窗口的原表平均分配取近似值。满窗统一精算 24+17+23=64 页。所有层、头及五步共用此规则。',
'- 未精算块补偿关闭：保留原 risk、配额和换页，用原始精算页 LUT 调用独立双项 BF16 AV 核。省略未精算页的归一化质量与 V 贡献，所选页内重新归一化。耗时含核实现差异。',
'- 全窗重排：每次稀疏调用在已施加 RoPE 的全部可见 K/V 上计算分组特征与全窗排序，再按锚点／其他历史／当前分开成页。当前 Q 按当前 K 的同头置换重排，输出逆置换回原序。所有页重算摘要，纠偏范围内重算 Reader 编码。完整首帧精算、当前／历史配额、使用原 clean 提交及淘汰规则确定本次调用的输入 token。时间包括排序、摘要、复制和重打包。',
'- 原先的历史重建仅作缓存等价性对照：按各个原始 clean chunk 重做相同分组，局部与视频指标持平，耗时 8.428 秒。']
(folder/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
