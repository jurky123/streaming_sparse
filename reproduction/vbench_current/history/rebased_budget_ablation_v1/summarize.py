from pathlib import Path
import json,csv,hashlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=Path(__file__).parent
read=lambda n:json.loads((d/(n+'.json')).read_text())
scenes=['forest_run_113','snow_ski_661','kitchen_cook_557']
scans=[read(n) for n in scenes];curves=np.array([x['nmse'] for x in scans]);mean=curves.mean(0)
mods=read('module_rows');grid=scans[0]['grid']
nm=lambda x:str(x[0])+' / '+str(x[1])
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
for j,share in enumerate([0,25,50,75,100]):
    ax.plot([5,8,10,12,15,20],mean[j:30:5],marker='o',label=f'{share}%')
for i,name in enumerate(scenes):bx.plot([5,8,10,12,15,20],curves[i,1:30:5],marker='o',label=name)
for a in (ax,bx):a.set_xlabel('Exact budget outside anchor (%)');a.set_ylabel('Attention NMSE');a.grid(alpha=.2);a.set_ylim(bottom=0)
ax.set_title('Share assigned to current chunk');bx.set_title('25% of budget to current chunk')
ax.legend(ncol=2);bx.legend();fig.savefig(d/'budget_curve.png',dpi=180);plt.close(fig)
# Keep data resolved by algorithm stage as well as the original per-cell NPZ.
with (d/'curve_by_stage.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['scene','group_type','index','total','current_share_or_history_cap','sse','dense_energy','nmse'])
 for scene in scenes:
    a=np.load(d/(scene+'.npz'));s=a['sse'][2:].astype(float);en=a['dense_energy'][2:].astype(float)
    for ix,group in enumerate(['chunk','step','layer','head']):
        axes=tuple(i for i in range(4) if i!=ix);err=s.sum(axis=axes);den=en.sum(axis=axes)
        for j,(row,wt) in enumerate(zip(err,den)):
            for spec,x in zip(grid,row):w.writerow((scene,group,j+([2,0,1,0][ix]),*spec,x,wt,x/wt))
with (d/'curve.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['total_or_full','current_budget_share_or_history_percent','full_current_pages','full_history_pages',*scenes,'mean'])
 for i,(spec,ct) in enumerate(zip(grid,scans[0]['counts']['432'])):w.writerow([*spec,*ct,*curves[:,i],mean[i]])
roof=read('roof')
ref=read('dense_no_route')['wall'];quality=read('video_metrics')
runs=['originalB','closed_11','closed_21','closed_30','closed_31']
labels=['原重定位 B','total 10%，25% 份额','total 15%，25% 份额','当前全算，历史 5%','当前全算，历史 10%']
lines=['# 预算消融结果与进度','', '完整协议见 DESIGN.zh.md。三条教师轨迹等权汇总，每条内部按 Dense 输出能量加权。每个预算有 162,864 个 query-head 观测；32 组共 5,211,648 个同状态误差观测。扫的是每个稀疏层头的一个轮转查询行，另有四窗口完整 Q 的融合核核查。','',
'## 预算误差曲线','','| 非锚点 total | 0% 份额给当前 | 25% | 50% | 75% | 100% |','|---|---:|---:|---:|---:|---:|']
for i in range(0,30,5):lines.append('| '+str(grid[i][0])+'% | '+' | '.join(f'{x:.5f}' for x in mean[i:i+5])+' |')
lines+=['',f'当前全算＋历史 5% / 10% 的 NMSE 分别为 {mean[30]:.5f}、{mean[31]:.5f}。每层/头/步/chunk 的拆分见 curve_by_stage.csv，每个交叉位置保留在三份 NPZ 中。','',
'## 东京街景闭环','','共同参照为相同重定位和分组/逐出顺序的 Dense。无路由 Dense 与开启路由但强制 Dense 的教师导出 MP4 的 SHA256 相同。无路由 Dense 单次生成 '+f'{ref:.3f} s。','',
'| 方案 | 满窗总精算页 | 生成 s | 加速比 | 首次稀疏接缝 ↓ | 首次滚窗接缝 ↓ | 后段颜色对 Dense ↓ | 后段 CLIP 对 Dense ↑ |','|---|---:|---:|---:|---:|---:|---:|---:|']
for n,label,ix in zip(runs,labels,[None,11,21,30,31]):
 a=read(n);v=quality[n];tm=a['wall'];pages=64 if ix is None else 24+sum(scans[0]['counts']['432'][ix])
 lines.append(f"| {label} | {pages} | {tm:.3f} | {ref/tm:.3f}× | {v['seam_transition_error']:.3f} | {v['first_roll']:.3f} | {v['color_late']:.3f} | {v['late_clip_dense']:.4f} |")
lines+=['','接缝是 32×32 图像块平均 RGB 的帧间变化相对教师的 MAE，量纲为 0–255；首次切换使用帧 20→21，首次滚窗使用 68→69。后段取帧 320–479。CLIP 为整帧场景代理。计时仅一次预热后的生成，非配对卡；小于 1% 的差异不作提速判断。','',
'## 分模块计时和理论主项','','| 方案 | 注意力设备区间 s | 路由和纠偏 s | 摘要/编码 s | clean 提交区间 s | 重定位 s | 编译预热墙钟 s |','|---|---:|---:|---:|---:|---:|']
for n,label in zip(runs,labels):
 a=read(n);vals=[a['stages'][k]['milliseconds']/1000 for k in ('attention','route_and_correction','summary_and_encoding','clean_commit','anchor_rebase')]
 lines.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in vals+[a['warmup']])+' |')
lines+=['','以上区间包含 29 个托管层，第 0 层和骨干不在这些设备区间内。clean 有异步流，提交区间只表示入口观测时间。attention 含托管层 Dense 调用和融合核；下表另列隔离融合核。','',
'| 满窗方案 | 精算 GFLOP | 补偿 GFLOP | 分区选页 ms | 融合核 ms | 按同形状 Dense 吞吐折算 ms | 差额 ms |','|---|---:|---:|---:|---:|---:|---:|']
for ix in (11,21,30,31):
 m=next(r for r in mods if r['pages']==432 and r['spec']==grid[ix])
 lines.append(f"| {nm(grid[ix])} | {m['exact_flops']/1e9:.1f} | {m['comp_flops']/1e9:.1f} | {m['select_ms']:.3f} | {m['fused_ms']:.3f} | {m['at_dense_rate_ms']:.3f} | {m['gap_ms']:.3f} |")
lines+=['','全 128 个模块测点见 module_rows.json。该形状 Dense 主项为 782.76 GFLOP；指数运算、掩码、内存、流水与低秩投影尾项不在主矩阵主项模型。不同页数/形状不能共用一个参考吞吐。','']
# Conservative partial ledgers for stages containing multiple dtypes/non-matmul work.
ledger=[]
for n in runs:
 a=read(n)
 for stage,calls,flops,band,source in [
   ('route_and_correction',4524,sum(116*2*12*72*16*min(72*(c+1),432)*4*128 for c in range(2,41)),'torch.bfloat16','M16 probe QK only; risk, top-k, correction excluded'),
   ('summary_and_encoding',4524,4524*2*12*4608*128*32,'torch.bfloat16','current K feature projection only; omits pooling, stats and encoding'),
   ('clean_commit',1189,1189*2*12*4608*128*32*2,'torch.bfloat16','K and V feature projection lower work count; async timing is submission scope'),
   ('anchor_rebase',35,35*30*1536*12*22*6,'torch.float64','22 complex time pairs; excludes summary refresh, casts, canonical copies and encoding')]:
    obs=a['stages'][stage]['milliseconds'];partial=flops/(roof[band]['tflops']*1e9)
    ledger.append(dict(variant=n,module=stage,calls=calls,measured_stage_ms=obs,known_flops=flops,empirical_rate_tflops=roof[band]['tflops'],partial_reference_ms=partial,measured_minus_partial_ms=None if stage=='clean_commit' else obs-partial,scope=source))
(d/'partial_module_ledger.json').write_text(json.dumps(ledger,indent=2))
lines+=['partial_module_ledger.json 给出路由、摘要、重定位、clean 已能核实的分项 FLOP 和部分工作参考。参考使用本机实测 roof.json；没有把排序、编码、拷贝和异步成本说成全部可追回。完整异构模块的理论差额仍待逐核归因。','','## 扫描与重训练进度','']
for name,s in zip(scenes,scans):
 lines.append(f"- {name}：扫描墙钟 {s['wall_seconds']:.2f} s，页统计 {s['device_ms']['page_statistics']/1e3:.2f} s，32 组预测 {s['device_ms']['all_budget_predictions']/1e3:.2f} s。")
lines+=['', '第一次用差分累计分子/质量进行扫描未通过有限性或覆盖联合断言，废弃该次结果。报告使用 log-mass + 正项 softmax 重跑的数据，三条均通过覆盖和有限值检查。复用统计将 32 次完整教师 rollout 减至每 prompt 一次。正式训练样本体积明显大于扫描误差归档，不应存下全部窗口的全模型原始 KV。','',
'已重采 train_pilot 下训练集森林视频的 4 个 FP64 教师块，共同验证 Dense 重构和原始数据加载器。对 10% 预算、25% 份额做了两阶段优化器实跑，梯度覆盖 pool_k、pool_v、pool_feature 与 reader。原 candidate-pool 先全局截取移除/加入再限制同区，在一层一头四次采样中三次无合法交换。仅在训练接线实验中增加分区候选回退后，4 个样本两阶段都可更新。两套日志分别为 updates.json 和 updates_stratified.json。闭环部署视频仍是原候选池过滤规则。','',
'未完成项：全训练集/开发集采集、覆盖 348 个位置的完整重训练、冻结前后配对视频与性能验收、非融合异构模块的完整硬件计数。小训练样本不构成全量重训练成果。现阶段候选 10%/25%（优先速度）与 15%/25%（优先局部误差及该样本对 Dense 颜色）；分区候选回退需训练/运行时一致并重测。']
(d/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
