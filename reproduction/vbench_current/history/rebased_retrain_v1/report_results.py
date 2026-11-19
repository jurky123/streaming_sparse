import hashlib,json,re,statistics
from pathlib import Path
from collections import defaultdict
from statistics import mean,median

root=Path(__file__).parent
repo=Path('/home/shuchenweng/jzk/streaming_sparse')
def load(f):return json.loads(f.read_text())
policy=load(repo/'local_experiments/video_runtime_v1/presets/intersection.json')
quality={k:v for scene in (0,1) for k,v in load(root/f'quality_{scene}.json')['rows'].items()}
doc=['# 两档预算的分区纠偏与重训练结果','','## 30 秒视频闭环','',
     '统一采用首帧时间重定位与分区动作池。每格按原正式权重 → 同预算重训练权重排列，接缝、颜色、空间误差与 latent NMSE 越低越好，CLIP 越高越好。空间误差取后段 32×32 池化 RGB，相对 Dense；低运动区取教师后段帧间 RGB 变化最小的三分之一空间格，作为背景稳定性的代理；CLIP 按整帧计算。','',
     '| 场景 / total | 生成秒 | 对锚点 Dense 加速 | 首稀疏接缝 | 首滚窗接缝 | 后段 RGB | 后段空间 | 后段低运动区 | 后段 CLIP | latent NMSE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for scene,tag in enumerate(('东京/701','猛犸/702')):
    dense=load(root/f'{scene}_11_dense.json')['wall']
    for i,total in ((11,10),(21,15)):
        orig,trained=[quality[f'{scene}_{i}_{v}'] for v in ('original','trained')]
        tm=[load(root/f'{scene}_{i}_{v}.json')['wall'] for v in ('original','trained')]
        cells=[f'{tag} / {total}%',f'{tm[0]:.2f} → {tm[1]:.2f}',f'{dense/tm[0]:.3f}× → {dense/tm[1]:.3f}×']
        for k in ('seam_transition_error','first_roll','color_late','late_grid_error','late_low_motion_grid_error','late_clip_dense','latent_nmse'):
            cells.append(f'{orig[k]:.4f} → {trained[k]:.4f}')
        doc.append('| '+' | '.join(cells)+' |')
doc+=['','## 东京同卡配对计时','','每档 3 对，按 AB、BA、AB 排序。生成计时排除编译、VAE、编码。','',
      '| total | 原权重中位秒 | 重训练中位秒 | 配对节省毫秒 |','|---|---:|---:|---|']
for idx in (11,21):
    pairs=[[load(root/f'0_{idx}_{v}_repeat{r}.json')['wall'] for v in ('original','trained')] for r in (1,2,3)]
    doc.append(f'| {10 if idx==11 else 15}% | {median(x[0] for x in pairs):.3f} | {median(x[1] for x in pairs):.3f} | '+', '.join(f'{(a-b)*1000:+.1f}' for a,b in pairs)+' |')
doc+=['','## 独立开发集模块消融','','每档 2,784 个 Q64 块。按块等权；intersection 实际启用 reader 的层、头、去噪步由正式 preset 指定。','',
      '| total | 权重 | 所有头只做风险选页 | 所有头假设执行 reader | 按 intersection 启用 | 启用样本数 |','|---|---|---:|---:|---:|---:|']
for idx in (11,21):
    for label in ('original','C_only','reader_only','trained'):
        rows=load(root/f'budget_{idx}/dev_{label}.json')['rows']
        used=0; deployed=[]
        for path,risk,corrected in rows:
            l,h,s=map(int,re.search(r'_l(\d+)_h(\d+)_s(\d+)',path).groups())
            on=l in policy['layers'] and h in policy['heads'] and s in policy['steps']
            used+=int(on);deployed.append(corrected if on else risk)
        doc.append(f'| {10 if idx==11 else 15}% | {label} | {mean(x[1] for x in rows):.6f} | {mean(x[2] for x in rows):.6f} | {mean(deployed):.6f} | {used} |')
doc+=['','## 拆分重训练模块的视频消融','','每行使用相同分区策略、预算、提示词与 seed。','',
      '| 场景 / total | 权重 | 首稀疏接缝 | 后段颜色 | 后段低运动区 | 后段 CLIP |','|---|---|---:|---:|---:|---:|']
for scene,tag in enumerate(('东京/701','猛犸/702')):
    for idx,total in ((11,10),(21,15)):
        for stage in ('original','C_only','reader_only','trained'):
            key=f'{scene}_{idx}_{stage}'
            if stage in ('original','trained'): row=quality[key]
            else: row=load(root/f'{stage}_quality_{scene}_{idx}.json')['rows'][key]
            doc.append(f'| {tag} / {total}% | {stage} | '+ ' | '.join(f'{row[k]:.4f}' for k in ('seam_transition_error','color_late','late_low_motion_grid_error','late_clip_dense'))+' |')
doc+=['','## 阶段耗时','','每档训练 8,352 条，两轮共 16,704 次更新/阶段，覆盖 348 个层头。','',
      '| total | C 秒 | Full_Pset 秒 | 原权重开发评估秒 | 重训练开发评估秒 |','|---|---:|---:|---:|---:|']
for idx in (11,21):
    d=root/f'budget_{idx}'
    doc.append('| '+str(10 if idx==11 else 15)+'% | '+' | '.join(f"{load(d/f)['seconds']:.1f}" for f in ('C.json','Full_Pset.json','dev_original.json','dev_trained.json'))+' |')
doc+=['','采样耗时与源文件身份见 identity.json。分区动作、真实 teacher/random 状态、模块主项 FLOP、参考吞吐和实测差额分别见 verify.json、module_states.json、pool_benchmark.json、training_module_profile.json 和 ledger.json。','',
      '正式对照采用相同分区动作池。旧的全局截取动作池与当前分区池会选出不同的页。东京 Dense 重新生成的 MP4 与此前的同锚点 Dense SHA256 一致。']
(root/'DIAGNOSTICS.zh.md').write_text('\n'.join(doc)+'\n')

main=['# 重定位首帧与预算重训练验收','',
      '质量按成片中可见的接缝、主体身份、背景结构与色调连续性审阅。Dense 作为独立生成样本供参考。相对 Dense 的逐像素 RGB 和 latent 误差仅保留在 DIAGNOSTICS.zh.md 供排查。','',
      '东京与猛犸各有一份同步 30 秒四格视频 side_by_side_0.mp4 与 side_by_side_1.mp4。第一行为 10%，第二行为 15%；左为原权重，右为联合重训练。','']
if (root/'VISUAL.zh.md').exists():main.append((root/'VISUAL.zh.md').read_text())
main+=['','## 整条视频速度','','| 场景 | 同锚点 Dense | 10% 原 → 联合 | 15% 原 → 联合 |','|---|---:|---:|---:|']
for scene,tag in enumerate(('东京/701','猛犸/702')):
    dense=load(root/f'{scene}_11_dense.json')['wall']
    cells=[tag,f'{dense:.3f} s']
    for idx in (11,21):
        vals=[load(root/f'{scene}_{idx}_{v}.json')['wall'] for v in ('original','trained')]
        cells.append(' → '.join(f'{x:.3f} s ({dense/x:.3f}×)' for x in vals))
    main.append('| '+' | '.join(cells)+' |')
main+=doc[doc.index('## 东京同卡配对计时'):doc.index('## 拆分重训练模块的视频消融')]
main+=doc[doc.index('## 阶段耗时'):]
(root/'RESULTS.zh.md').write_text('\n'.join(main)+'\n')
