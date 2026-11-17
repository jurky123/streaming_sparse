"""Report the fixed-shape independent-axis experiment and measured combined policies."""
import csv,json,statistics
from pathlib import Path
import numpy as np
from .scope_experiment import OUT,SEEDS,write
from .weights import ROOT

def main():
 load=lambda p:json.loads(p.read_text())
 selection=load(OUT/'selection.json');names=['base','full','top_layers','top_heads','top_steps','intersection','additive_quarter']
 labels=dict(base='C＋Risk 基础档',full='全部 Full_Pset',top_layers='分别筛选：层前1/4',top_heads='分别筛选：head前1/4',top_steps='分别筛选：step前1/4',intersection='三个前1/4的交集',additive_quarter='独立收益相加：组合前1/4')
 table=[];per_seed=[]
 for name in names:
  phase='individual' if name in ('base','full') else 'selected';records=[load(OUT/phase/name/f's{s}/result.json') for s in SEEDS];timings=[load(OUT/'timing'/f's{s}'/(name+'.json')) for s in SEEDS]
  cells=records[0]['runtime']['counters'][1]//504
  row=dict(name=name,enabled_cells=cells,budget=cells/1392,generation_seconds=float(np.mean([v['median_seconds'] for v in timings])))
  for metric in ['lpips','ssim','psnr']:row[metric]=float(np.mean([v['quality'][metric] for v in records]))
  row['gain_vs_base']=float(np.mean([selection['base_lpips'][str(s)]-v['quality']['lpips'] for s,v in zip(SEEDS,records)]))
  table.append(row)
  for seed,v,t in zip(SEEDS,records,timings):per_seed.append(dict(name=name,seed=seed,generation_seconds=t['median_seconds'],**{m:v['quality'][m] for m in ['lpips','ssim','psnr']},gain_vs_base=selection['base_lpips'][str(seed)]-v['quality']['lpips']))
 full=next(v for v in table if v['name']=='full');base=table[0]
 for v in table:v['speedup_vs_full']=full['generation_seconds']/v['generation_seconds'];v['extra_vs_base_s']=v['generation_seconds']-base['generation_seconds']
 for filename,rows in [('comparison.csv',table),('per_seed.csv',per_seed)]:
  with (OUT/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 flat=[]
 for axis,rows in selection['rankings'].items():
  for rank,r in enumerate(rows,1):flat.append(dict(axis=axis,rank=rank,index=r['index'],mean_gain=r['gain'],gain_s701=r['seed_gains']['701'],gain_s702=r['seed_gains']['702'],mean_lpips=r['mean_lpips'],selected=r['index'] in selection['selected'][axis]))
 with (OUT/'individual_rankings.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 specs={j['name']:{k:v for k,v in j.items() if k!='seed'} for j in load(OUT/'selected_manifest.json')}
 for name,spec in specs.items():write(OUT/'presets'/(name+'.json'),dict(spec,static_route=True))
 write(OUT/'summary.json',dict(table=table,per_seed=per_seed,selection=selection,ranking_samples='city_walk, seeds701/702; same samples used for selection and combined-policy evaluation; not held out',generation_timing='one GPU per seed, all 7 policies on that GPU, one warmup then 3 repetitions; mean of the 2 seed medians, excludes VAE/IO'))
 lines=['# Full_Pset：逐层、逐head、逐step收益与前1/4选择','','本实验完成三个独立维度的单独开启测试，再实测三种选择规则（共5个组合方案）。所有结论限定在city_walk、seed701/702的短视频；两个seed既参与选择也参与组合验收，没有独立测试集。',
 '', '## 1. 固定实验定义', '', '- 9chunks、105帧、16FPS、768×512，history360（5个历史chunk）、tile72；Page64/Q64/R4、完整V均值+rank8、M16/B16、10%精算、16×32交换+no-op；当前全覆盖C_s17/Full_Pset_s17，M16入口开、G2关。', '- 层测试：每次仅一个层的所有head、所有4个noisy step开启Full_Pset。head测试：每次仅一个head跨所有可修正层、所有4步开启。step测试：每次仅一个noisy step跨所有可修正层/head开启。其余位置均走C＋Risk。', '- 可修正层索引1–29，共29层；模型层0原本固定Dense，因此Full_Pset无可开位置，其收益按结构为0，未改写为稀疏层。head索引0–11。noisy step索引0–3；clean step4只维护历史，不参加交换选择。', '- 45个单维度设置×2seeds=90视频，加基础/完整档对照4视频=94；五个组合方案×2seeds=10视频。正式固定形状清单共104份视频，另有预热、计时重复及顺序检查。Dense复用已有city_walk对应两seed参考，只生成/解码一次。', '- 排名收益 `gain = LPIPS(基础档, Dense) - LPIPS(单独开启, Dense)`，对两个seed等权平均；越大越好，相同收益按索引升序。SSIM/PSNR同步记录，未混成新的加权指标。', '- 指标窗口[21,101)，LPIPS AlexNet和SSIM为256×384，PSNR使用原分辨率MSE。所有新视频采用相同TF32/cudnn设置。',
 '', '## 2. 先解决常驻任务顺序的数值影响', '', '首批自动动态编译结果中，先运行head-only任务再运行step3，与新进程直接运行step3得到不同轨迹，latent NMSE约0.20686。基础/完整控制仍与旧视频一致，所以只检查这两端不足以发现此问题。', '', '直接四路对照：自动动态编译的两种顺序不一致；固定编译形状（static_route=True）后两种顺序逐位一致，并与此前step3结果一致。[顺序检查](order_check/summary.json)记录实际结果。首次结果、日志、排名和组合视频全部保留在`dynamic_order_dependent/`，不用于以下排名。', '', '随后以固定编译形状重新执行全部94个排名视频、10个组合视频和正式计时。固定形状基础档与旧视频逐位一致，完整档与旧自动动态编译版本latent NMSE为8.6654e-5；因此本表完整档也重新生成，不混用旧质量结果。不是把旧排名套到新路径；这是本实验的统一数值控制。初次还触发默认编译变体上限，实验进程提升到64（累计512），没有修改数学、精度或学习权重。部署选定preset也强制固定编译形状。',
 '', '## 3. 单维度排名与选择', '', f"选中层（收益降序）：**{selection['selected']['layer']}**；head：**{selection['selected']['head']}**；step：**{selection['selected']['step']}**。", '', '取整规则：29层取floor(29/4)=7层，即24.14%；12heads取3个；4个noisy step取1个。严格选收益前列，即使名额中存在负收益也不另行减少预算。图中绿色为被选中项，两个点分别表示两个seed。', '', '![独立收益](independent_gains.png)']
 for axis,rows in selection['rankings'].items():
  lines+=['',f'### {axis} 全部结果（按平均收益排序）','','|排名|索引|seed701收益|seed702收益|平均收益↑|平均LPIPS↓|选中|','|---|---:|---:|---:|---:|---:|---|']
  for i,v in enumerate(rows,1):lines.append(f"|{i}|{v['index']}|{v['seed_gains']['701']:+.6f}|{v['seed_gains']['702']:+.6f}|{v['gain']:+.6f}|{v['mean_lpips']:.6f}|{'是' if v['index'] in selection['selected'][axis] else ''}|")
 lines+=['','## 4. 五种组合方案的实际结果','','三种规则如下：分别按维度选前1/4得到层/head/step三档；三个选中集合取交集得到一档；将层/head/step独立收益相加，枚举29×12×4=1392个组合并取最高348个，得到第五档。相加使用同一LPIPS收益单位的原始边际收益，不额外按覆盖数归一化；三个轴原始覆盖面不同，这个分数不是经过校准的单个cell因果收益。所有组合都实际生成视频，不用相加分数代替实际质量。', '', '|方案|启用预算|生成/s↓|相对完整档加速↑|LPIPS↓|SSIM↑|PSNR/dB↑|对基础档LPIPS收益↑|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v in table:lines.append(f"|{labels[v['name']]}|{v['budget']:.2%}|{v['generation_seconds']:.4f}|{v['speedup_vs_full']:.3f}×|{v['lpips']:.6f}|{v['ssim']:.6f}|{v['psnr']:.3f}|{v['gain_vs_base']:+.6f}|")
 lines+=['','生成时间：A100-SXM4-40GB，PyTorch CPU线程数2；seed701所有方案固定GPU6，seed702固定GPU7；各方案完整预热一次，3次计时取中位数，再平均两个seed中位数。排除IO、模型加载、编译和VAE，速度比为完整档平均生成时间/该方案平均生成时间。批量质量worker的单次时间不用于这张正式速度表。', '', 'head筛选仅让选中head进入在线Reader/交换，当前页面编码仍覆盖该层全部head；按层筛选同时减少需要维护编码的层数。因此相同评分预算不代表完全相同的额外时间。预算分母为1392个layer×head×step组合；实际短视频每个cell产生7个稀疏chunk×72tiles=504次评分。所有文件核对计数：完整701568，层前1/4为169344，head/step/相加组合为175392，交集10584。交集只有21/1392=1.51%预算，并不是25%。', '', '|方案|seed|LPIPS↓|对基础收益↑|SSIM↑|PSNR/dB↑|','|---|---:|---:|---:|---:|---:|']
 for v in per_seed:lines.append(f"|{labels[v['name']]}|{v['seed']}|{v['lpips']:.6f}|{v['gain_vs_base']:+.6f}|{v['ssim']:.6f}|{v['psnr']:.3f}|")
 best=min(table[2:],key=lambda v:v['lpips']);lines+=['',f"本清单中，五个筛选方案的最低平均LPIPS来自 **{labels[best['name']]}**（{best['lpips']:.6f}）；它比基础档变化{best['gain_vs_base']:+.6f}，相对完整档生成速度{best['speedup_vs_full']:.3f}×。这不是未见prompt/seed上的泛化结论，也不因排名假定独立就认为组合一定叠加获益。", '', '![相加收益选中的cell](additive_mask.png)', '', '## 5. 如何开启及实现位置', '', '五个可复现配置保存在[presets/](presets/)。在当前stream基线命令中添加`--full-pset --pset-policy artifacts/video_runtime_v1/pset_scope_v1/presets/top_layers.json`即可；将文件名替换为top_heads、top_steps、intersection、additive_quarter可切换。benchmark、batch、worker均透传此参数；policy文件内容SHA进入identity，实际强制static_route=True。未修改全局默认完整档，不把这一个场景的筛选自动推广为部署默认。', '', '- [runtime.py](../../../local_experiments/video_runtime_v1/runtime.py)：`set_pset_scope`和`set_pset_schedule`；逐layer/step选择当前修正；有修正需求的层在所有clean commit持续维护编码。','- [pipeline_graph.py](../../../local_experiments/video_runtime_v1/pipeline_graph.py)：Risk和基础mask仍覆盖全部head，仅选中head进入Reader/512交换，再把修改的mask写回；未选head保持基础LUT。图key包含选中head数量，索引存入GPU缓冲，任务间更新索引并复用同形状图。','- [deployment.py](../../../local_experiments/video_runtime_v1/deployment.py)：加载preset、固定编译形状、policy SHA；batch/worker/benchmark入口透传。','- [check_scope.py](../../../local_experiments/video_runtime_v1/check_scope.py)：head0/7/11及3-head组合的GPU输出契约；未选head和基础一致、所选与完整修正一致、事件数正确。','- [check_scope_order.py](../../../local_experiments/video_runtime_v1/check_scope_order.py)：任务先后顺序及静态形状对照。','- [scope_experiment.py](../../../local_experiments/video_runtime_v1/scope_experiment.py)：8卡常驻模型、45项单维度清单、两seed指标、三种规则的5项配置与组合验证。','- [scope_timing.py](../../../local_experiments/video_runtime_v1/scope_timing.py)：每seed固定一张GPU测量全部7方案。','- [report_scope.py](../../../local_experiments/video_runtime_v1/report_scope.py)：完整排名、对比表、preset和可视化。', '', '四列对照视频依次为Dense、基础档、完整Full_Pset、按层前1/4：[seed701](comparison_s701.mp4)、[seed702](comparison_s702.mp4)。公共preset入口已与实验脚本逐位对照，step筛选的组合复验也与单项测试逐位一致，见[入口检查](entry_checks.json)。\n\n原始证据：[单项排名CSV](individual_rankings.csv)、[组合对比CSV](comparison.csv)、[分seed结果](per_seed.csv)、[完整选择规则和348个cell](selection.json)、[输入输出契约](contracts.json)、[数值控制](numerical_protocol.json)。`individual/`与`selected/`保存每个视频的latent、视频和逐帧指标，`timing/`保存3次正式计时。']
 (OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained')
 for ax,axis in zip(axes,['layer','head','step']):
  rows=sorted(selection['rankings'][axis],key=lambda x:x['index']);indices=[r['index'] for r in rows]
  ax.bar(indices,[r['gain'] for r in rows],color=['#26966b' if i in selection['selected'][axis] else '#a8b6c3' for i in indices]);ax.scatter(indices,[r['seed_gains']['701'] for r in rows],s=18,label='seed701',color='#265d9c');ax.scatter(indices,[r['seed_gains']['702'] for r in rows],s=18,label='seed702',color='#d98039');ax.axhline(0,color='black',lw=.7);ax.set_xticks(indices);ax.set_title(axis);ax.set_ylabel('LPIPS gain vs base');ax.legend()
 fig.savefig(OUT/'independent_gains.png',dpi=160);plt.close(fig)
 mask=np.zeros((4,29,12))
 for _,l,h,s in selection['additive_selected_cells']:mask[s,l-1,h]=1
 fig,axes=plt.subplots(1,4,figsize=(12,9),layout='constrained')
 for s,ax in enumerate(axes):ax.imshow(mask[s],vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_title(f'noisy step {s}');ax.set_xticks(range(12));ax.set_yticks(range(29),range(1,30));ax.set_xlabel('head');ax.set_ylabel('layer')
 fig.savefig(OUT/'additive_mask.png',dpi=160);plt.close(fig)
 print(json.dumps(table,indent=2),flush=True)
if __name__=='__main__':main()
