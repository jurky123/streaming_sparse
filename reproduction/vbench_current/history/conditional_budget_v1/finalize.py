from report import D,weights,times,read
from statistics import median
import numpy as np
doc=(D/'RESULTS.zh.md').read_text()
extra=['','## 同卡交替计时','','使用东京/701、原权重，同卡 AB、BA、AB 三对。差值为条件表耗时减去固定配额耗时。']
for t in (10,15):
 v=np.array([[read(D/f'time_{t}_{m}_{r}.json')['wall'] for m in ('fixed','table')] for r in (1,2,3)])
 dt=np.diff(v,axis=1).ravel()
 extra.append(f'- {t}% 每对差值为 '+', '.join(f'{x*1000:+.1f} ms' for x in dt)+f'，中位差 {median(dt)*1000:+.1f} ms。')
extra+=['','## 固定条件表后微调','','沿用上轮六条训练和两条独立开发轨迹，在每个训练块查询实际 head、layer、step、chunk 对应配额，原权重热启动，两个阶段各 16,704 次更新。']
for t in (10,15):
 d=D/f'budget_{t}'
 base,refit=[read(d/f'dev_{v}.json')['mean'] for v in ('original','trained')]
 extra.append(f"- {t}% 的 C、Full_Pset 各用时 {read(d/'C.json')['seconds']:.1f}、{read(d/'Full_Pset.json')['seconds']:.1f} 秒。2,784 个开发块，假设每头都执行 reader 的局部 NMSE 从 {base[1]:.5f} 变为 {refit[1]:.5f}。")
 for sc in (0,1):
  x=read(D/f'{sc}_{t}_refit.json'); ev=x['fusion_device_ms'];assert sum(row['calls'] for row in ev)==4524
  ms=sum(row['ms'] for row in ev)
  model=read(D/'ragged_cost.json');work=next(row['ideal_tflop'] for row in model if row['budget']==t and row['schedule']=='window_step_layer_head')
  dense=read(weights/f'{sc}_11_dense.json')['wall']
  extra.append(f"  - 场景 {sc}：30 秒生成 {x['wall']:.3f} s，对同锚点 Dense {dense/x['wall']:.3f}×；稀疏融合设备区间 {ms:.1f} ms，有效吞吐 {work*1000/ms:.1f} TFLOP/s。")
extra+=['','逐格预算见 active_10.csv.gz 与 active_15.csv.gz。frontier.csv 对六档平均预算分别给出四条留出轨迹的质量成本曲线，age_comparison.csv 按满窗时间段拆分。head_curves.png 展示六个层步头位置的局部曲线，calibration_curves.npz 保存全部交叉位置。','']
if (D/'VISUAL.zh.md').exists():extra.append((D/'VISUAL.zh.md').read_text())
(D/'RESULTS.zh.md').write_text(doc+'\n'.join(extra))
