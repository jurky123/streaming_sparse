from .run import *
import shutil,tarfile

def table(name,rows):
 if not rows:return
 with (E25/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def aggregate(rows):
 oracle=sum(x['oracle_gain'] for x in rows);initial=sum(x['initial'] for x in rows);gain=sum(x['executed_gain'] for x in rows);rel=[100*x['executed_gain']/x['initial'] for x in rows if x['initial']>1e-12]
 return dict(n=len(rows),initial_sum=initial,oracle_gain_sum=oracle,executed_gain_sum=gain,hindsight_gain_sum=sum(x['hindsight_gain'] for x in rows),R_executed=gain/oracle if oracle else None,R_proposal=sum(x['hindsight_gain'] for x in rows)/oracle if oracle else None,pooled_improvement_pct=100*gain/initial,mean_improvement_pct=float(np.mean(rel)),relative_valid_n=len(rel),p10_pct=float(np.quantile(rel,.1)),worst_pct=float(min(rel)),positive=sum(x['executed_gain']>1e-12 for x in rows),degraded=sum(x['executed_gain']< -1e-12 for x in rows),noop=sum(not x['execute'] for x in rows),near_zero_executed=sum(x['execute'] and abs(x['executed_gain'])<=1e-12 for x in rows))

def report():
 rows=[];summary=[];conditions=[];audits=[];gates=[]
 groups=["A","C"]+(["D"] if all((E25/f"D_s{x}"/"complete.json").exists() for x in [17,29]) else [])
 for group in groups:
  for seed in [17,29]:
   dest=E25/f'{group}_s{seed}';assert (dest/'complete.json').exists();audits.append(json.loads((dest/'complete.json').read_text()));rr=json.loads((dest/'ladder.json').read_text());rows+=rr
   for method in ['P0','P1','P2','P3']:
    ss=aggregate([r for r in rr if r['method']==method]);summary.append(dict(group=group,seed=seed,method=method,**ss))
   cr=json.loads((dest/'conditional.json').read_text());alltc=[];allpc=[];normtc=[];normpc=[];dt=[];dp=[]
   for idx,r in enumerate(cr):
    p=np.load(dest/f'panel_{idx:04d}.npz');t=p['true_gain'];v=p['predicted_gain'];tc=t-t.mean(0);pc=v-v.mean(0);alltc.extend(tc.ravel());allpc.extend(pc.ravel());scale=max(r['initial'],1e-12);normtc.extend((tc/scale).ravel());normpc.extend((pc/scale).ravel());dt.extend((tc-tc.mean(1,keepdims=True)).ravel());dp.extend((pc-pc.mean(1,keepdims=True)).ravel())
   cond=dict(group=group,seed=seed,n=len(cr),R_conditional_raw=corr(alltc,allpc),R_conditional_event_normalized=corr(normtc,normpc),R_ranking_doublecentered=corr(dt,dp),median_event_conditional_corr=float(np.median([r['conditional_corr'] for r in cr if r['conditional_corr'] is not None])),mean_common_actions=float(np.mean([r['common_actions'] for r in cr])),mean_unique_states=float(np.mean([r['unique_states'] for r in cr])))
   for kind in ['sign_flip','rank_flip']:
    n=sum(r[kind+'_n'] for r in cr);tp=sum(r[kind+'_tp'] for r in cr);tr=sum(r[kind+'_true'] for r in cr);pr=sum(r[kind+'_pred'] for r in cr);correct=sum(r[kind+'_correct'] for r in cr);cond.update({kind+'_true_fraction':tr/n if n else None,kind+'_precision':tp/pr if pr else None,kind+'_recall':tp/tr if tr else None,kind+'_accuracy':correct/n if n else None})
   conditions.append(cond)
 for seed in [17,29]:
  s={x['method']:x for x in summary if x['group']=='C' and x['seed']==seed};gates.append(dict(seed=seed,teacher16=s['P1']['R_executed'],proposal=s['P0']['R_proposal'],passed=s['P1']['R_executed']>=.5 and s['P1']['R_executed']-s['P0']['R_proposal']>=.2))
 passed=all(x['passed'] for x in gates);dump(E25/'D_gate.json',dict(passed=passed,seeds=gates,rule='both C teacher16 >=50%, exceeds P0 hindsight >=20 percentage points'))
 table('ladder_per_event.csv',rows);table('ladder_summary.csv',summary);table('conditional_summary.csv',conditions);table('conditional_per_event.csv',[r for g in groups for s in [17,29] for r in json.loads((E25/f'{g}_s{s}'/'conditional.json').read_text())]);dump(E25/'audit.json',dict(runs=audits,test_evaluated=False,teacher_only=['P1','P2','P3','hindsight'],P1_stable_algebra='project Dense-minus-mixed output onto frozen B; same as true online-centred t/u objective',conditional_panel='common legal subset of original Medium across risk,teacher1/2,random1/2; state averages remove action means; separate double-centering removes state main effects',raw_data_unchanged=True))
 general=[]
 for group in groups:
  for seed in [17,29]:
   for method in ['P0','P1','P2','P3']:
    rr=[r for r in rows if r['group']==group and r['seed']==seed and r['method']==method]
    for dimension in ['trajectory','layer','chunk','step','head']:
     for val in sorted({x[dimension] for x in rr}):general.append(dict(group=group,seed=seed,method=method,dimension=dimension,value=val,**aggregate([r for r in rr if r[dimension]==val])))
 table('generalization.csv',general)
 lines=['# E25：冻结模型集合决策诊断','', '仅使用E24开发集288事件、两个种子的匹配预算Frozen-R4 A与Joint-R4 C。Test保持封存；不采集新QKV、不更新代表。所有动作在各自新摘要、新Risk初选、同一Medium池下，用真实64 Query NMSE评价。','', '| 组 | seed | P0执行恢复 | P0提议+事后真门控 | P1真16维/M16 | P2真128维/M16 | P3真128维/Q64 |','|---|---:|---:|---:|---:|---:|---:|']
 for group in groups:
  for seed in [17,29]:
   s={x['method']:x for x in summary if x['group']==group and x['seed']==seed};lines.append(f"|{group}|{seed}|{s['P0']['R_executed']:.2%}|{s['P0']['R_proposal']:.2%}|{s['P1']['R_executed']:.2%}|{s['P2']['R_executed']:.2%}|{s['P3']['R_executed']:.2%}|")
 lines+=['','恢复率均为Σ实际收益/Σ同池真实最佳正收益，分母一致；P1/P2不是完整目标的严格上界。事后真门控使用真实标签，只定位排序与接受瓶颈，不是可部署结果。完整均值、汇总、P10、最差和执行比例见ladder_summary.csv。','', '|组|seed|条件相关（原尺度）|按事件初始NMSE归一化|双重中心化排序交互相关|','|---|---:|---:|---:|---:|']
 for x in conditions:lines.append(f"|{x['group']}|{x['seed']}|{x['R_conditional_raw']:.4f}|{x['R_conditional_event_normalized']:.4f}|{x['R_ranking_doublecentered']:.4f}|")
 lines+=['','C诊断使用Risk、重新生成的真实1/2步局部交换、两个固定随机合法交换状态。E24没有持久保存所有训练状态，因此按其规则在冻结新摘要上重建；不使用旧摘要标签。共同面板取初始Medium动作在五状态中的合法交集，不将不同page编号当作相同动作。每动作减去跨状态平均值，再计算相关；另报双重中心化以剔除状态整体主效应。重复状态数量、共同动作数、符号及排序翻转precision/recall均保留。近零真实收益阈值1e-12；不能用高翻转accuracy替代稀少翻转的recall。','', 'D预登记门槛：两个C种子的P1恢复均至少50%，且均超出P0事后门控20个百分点。当前判定：'+('通过。'+('已完成冻结重训，D为相同容量新读出器，原C摘要逐tensor保持不变。' if 'D' in groups else '允许冻结重训相同读出器。') if passed else '未通过，不启动读出器重训。'),'','P0必须逐事件重现E24已保存动作与执行判断，P3与独立原目标计算一致；误差见audit.json。P1通过投影真实混合输出差进行稳定计算，等价于投影真实在线中心化t并保留真实质量修正，避免巨大校正量相减。所有真实教师量仅用于诊断。','', '数据来自两条开发轨迹，不是288条独立视频。此诊断无法单独证明摘要信息不存在；它只能区分当前固定子空间、M16覆盖、既有预测与门控造成的损失。']
 lines += ['', '条件分析中的预测分数除以该事件16 probes的真实Dense平均能量以对齐NMSE量纲；该正常数仅用于离线指标、不进入读出器、不会改变动作或no-op。跨事件汇总相关性依赖这种尺度，逐事件相关中位数不受影响。事件归一化相关性对近零初选误差敏感（归一化分母下限1e-12），因此不作为唯一结论；逐事件相关中位数、原尺度相关和双重中心化相关一并保留。A和C的真实候选池及摘要不同，不能将两组Oracle恢复率差直接当作同一目标上的能力差。']
 if 'D' in groups:
  lines += ['', 'E25-D：两个种子各从头训练4608次，匹配E24读出总更新量；两组各用其对应冻结C摘要。同结构、同训练集与原训练尺度、同60/20/20状态采样，不训练pooling/basis/Risk。每epoch评价开发集，以汇总最终NMSE最小选择，Test未使用。D未复用C读出初始化；它与联合训练的区别同时包括冻结目标与从头学习，因此即使D改善，也只能支持存在优化耦合问题，不能单独证明因果来源。']
 (E25/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(gate=passed,conditions=conditions,summary=summary),indent=2))
if __name__=='__main__':report()
