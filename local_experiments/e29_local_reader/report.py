from .teacher_report import *
from ..e26_oracle_coefficients.report import stablecorr

def main():
 rows=[];summary=[];conditions=[];info=[];curves=[];checkpoints=[];general=[];gates=[]
 for seed in [17,29]:
  dest=E29/f's{seed}';done=json.loads((dest/'training_complete.json').read_text());assert (dest/'analysis_complete.json').exists();rr=json.loads((dest/f"development_epoch{done['best']['epoch']}.json").read_text());rows+=rr;info+=json.loads((dest/'information.json').read_text());curves.extend(dict(seed=seed,**r) for r in json.loads((dest/'curves.json').read_text()));checkpoints.extend(dict(seed=seed,selected=r['epoch']==done['best']['epoch'],**r) for r in done['checkpoints'])
  for method in ['Local-Online','Local-TrueBeta','Local-TrueResponse']:
   for policy in ['Set','Independent']:summary.append(dict(seed=seed,method=method,policy=policy,**aggregate([r for r in rr if r['method']==method and r['policy']==policy])))
  olddone=json.loads((E27/f'B_s{seed}'/'training_complete.json').read_text());oldrows=json.loads((E27/f'B_s{seed}'/f"development_epoch{olddone['best']['epoch']}.json").read_text())
  for policy in ['Set','Independent']:summary.append(dict(seed=seed,method='E27-B',policy=policy,**aggregate([r for r in oldrows if r['method']=='PredictedBeta' and r['policy']==policy])))
  teacher=json.loads((ROOT/'artifacts/e28_local_response_v1'/f's{seed}'/'decisions.json').read_text())
  for policy in ['Set','Independent']:summary.append(dict(seed=seed,method='E28-LocalLOTO',policy=policy,**aggregate([r for r in teacher if r['method']=='Local-LOTO' and r['policy']==policy])))
  cr=json.loads((dest/'conditional.json').read_text());panels=[np.load(dest/f'panel_{i:04d}.npz') for i in range(288)]
  for method in ['Local-Online','Local-TrueBeta','Local-TrueResponse']:
   tt=[];pp=[];med=[]
   for p in panels:
    t=p['true_gain'];v=p[method];t=t-t.mean(0);v=v-v.mean(0);t=t-t.mean(1,keepdims=True);v=v-v.mean(1,keepdims=True);tt.extend(t.ravel());pp.extend(v.ravel());cc=stablecorr(t,v)
    if cc is not None:med.append(cc)
   sub=[r for r in cr if r['method']==method];tr=sum(r['sign_flip_true'] for r in sub);pr=sum(r['sign_flip_pred'] for r in sub);tp=sum(r['sign_flip_tp'] for r in sub);conditions.append(dict(seed=seed,method=method,doublecentered=stablecorr(tt,pp),median_event_interaction=float(np.median(med)),sign_flip_recall=tp/tr if tr else 0.,sign_flip_precision=tp/pr if pr else 0.))
  for method in ['Local-Online','Local-TrueBeta','Local-TrueResponse']:
   sub=[r for r in rr if r['method']==method and r['policy']=='Set']
   for dimension in ['trajectory','layer','chunk','step','head']:
    for val in sorted({r[dimension] for r in sub}):general.append(dict(seed=seed,method=method,dimension=dimension,value=val,**aggregate([r for r in sub if r[dimension]==val])))
  S=next(r for r in summary if r['seed']==seed and r['method']=='Local-Online' and r['policy']=='Set');I=next(r for r in summary if r['seed']==seed and r['method']=='Local-Online' and r['policy']=='Independent');old=next(r for r in summary if r['seed']==seed and r['method']=='E27-B' and r['policy']=='Set');cond=next(r for r in conditions if r['seed']==seed and r['method']=='Local-Online');oldcond=next(r for r in csv.DictReader((E27/'B_conditional_summary.csv').open()) if int(r['seed'])==seed and r['method']=='PredictedBeta');tra=[r for r in general if r['seed']==seed and r['method']=='Local-Online' and r['dimension']=='trajectory']
  checks=dict(positive_gain_each_trajectory=all(x['executed_gain_sum']>0 for x in tra),set_better_independent=S['executed_gain_sum']>I['executed_gain_sum'],oracle_recovery_ge20pct=S['R_executed']>=.2,nonnegative_mean=S['mean_improvement_pct']>=0,degradation_le15pct=S['degraded']/S['n']<=.15,p10_better_than_E27=S['p10_pct']>old['p10_pct'],worst_better_than_E27=S['worst_pct']>old['worst_pct'],conditional_corr_better=cond['doublecentered']>float(oldcond['doublecentered_raw']),signflip_recall_better=cond['sign_flip_recall']>float(oldcond['sign_flip_recall']))
  gates.append(dict(seed=seed,passed=all(checks.values()),checks=checks,trajectory_gains={r['value']:r['executed_gain_sum'] for r in tra}))
 tails=[]
 for seed in [17,29]:
  sub=[r for r in rows if r['seed']==seed and r['method']=='Local-Online' and r['policy']=='Set' and r['initial']>1e-12]
  tails.extend(dict(**r,final_nmse=r['initial']-r['executed_gain'],relative_improvement_pct=100*r['executed_gain']/r['initial']) for r in sorted(sub,key=lambda r:r['executed_gain']/r['initial'])[:10])
 table('tail_cases.csv',tails)
 table('decision_per_event.csv',rows);table('decision_summary.csv',summary);table('conditional_summary.csv',conditions);table('information_quality.csv',info);table('training_curves.csv',curves);table('checkpoints.csv',checkpoints);table('generalization.csv',general);dump(E29/'test_gate.json',dict(passed=all(g['passed'] for g in gates),seeds=gates,test_evaluated=False,rule='All preregistered criteria, both seeds. No post-result tuning.'))
 lines=['# E29：页面局部响应在线Reader验证','', '**结论：局部响应目标改善了部分在线决策能力，但两个种子未同时通过可靠性门槛，Test继续封存。真实beta替换显著提高收益与条件相关，质量修正及其与响应误差的耦合是更明确的诊断线索；尚不能归因为唯一原因。**','', '固定E24 C摘要、E27 Adaptive phi、B16、Risk、Page64/R4、10%/own、M16与Medium一次交换。没有采集QKV或扩大网络。两种子分别4608次更新，开发集两轨迹288事件；按预登记门槛决定是否解封Test。','', '## A：历史/当前教师依赖','', '|教师配置（Set）|seed17 Oracle恢复|seed29 Oracle恢复|','|---|---:|---:|']
 teacher=list(csv.DictReader((E29/'teacher_summary.csv').open()))
 for method in ['FullTeacher','HistoryOnly','CurrentOnly']:
  vals=[next(r for r in teacher if int(r['seed'])==s and r['method']==method and r['policy']=='Set') for s in [17,29]];lines.append(f"|{method}|{float(vals[0]['R_executed']):.2%}|{float(vals[1]['R_executed']):.2%}|")
 lines+=['','FullTeacher完整重现E28 Local-LOTO。缺失修正的页面同时令z=0、beta=0（u=0），其分母仍为原摘要质量，未保留真实分母。历史教师为跨事件LOTO、当前教师为本事件Oracle，全部不可部署。两类信息对集合目标有非线性共同作用，不能将两单支收益相加解释。','', '## B/C：实际收益与教师替换','', '|Set信息|seed17执行恢复|seed29执行恢复|seed17提议+真实门控|seed29提议+真实门控|','|---|---:|---:|---:|---:|']
 for method in ['E27-B','Local-Online','Local-TrueBeta','Local-TrueResponse','E28-LocalLOTO']:
  vals=[next(r for r in summary if r['seed']==s and r['method']==method and r['policy']=='Set') for s in [17,29]];lines.append(f"|{method}|{vals[0]['R_executed']:.2%}|{vals[1]['R_executed']:.2%}|{vals[0]['R_proposal']:.2%}|{vals[1]['R_proposal']:.2%}|")
 lines+=['','恢复率为Σ真实执行收益/Σ同池Q64最佳正收益，所有方法摘要/初选和分母一致。Local-Online只有摘要输入，是本轮主结果；TrueBeta/TrueResponse及LOTO使用教师信息，仅诊断。事后真实门控也不可部署。','', '|seed|决策|Risk NMSE总和|决策后NMSE总和|Oracle后NMSE总和|汇总改善|逐例平均改善|退化/no-op|P10|最差|','|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
 for seed in [17,29]:
  for policy in ['Set','Independent']:
   r=next(r for r in summary if r['seed']==seed and r['method']=='Local-Online' and r['policy']==policy);lines.append(f"|{seed}|{policy}|{r['initial_sum']:.6f}|{r['initial_sum']-r['executed_gain_sum']:.6f}|{r['initial_sum']-r['oracle_gain_sum']:.6f}|{r['pooled_improvement_pct']:+.3f}%|{r['mean_improvement_pct']:+.3f}%|{r['degraded']}/{r['noop']}|{r['p10_pct']:+.3f}%|{r['worst_pct']:+.3f}%|")
 lines+=['','正收益、近零已执行、逐例相对指标有效样本数见CSV；近零初始NMSE沿用1e−12阈值，仅影响相对指标，不删除汇总样本。No-op就是Risk，修正收益0。','', '## 集合条件变化','', '|seed|信息|双重中心化相关|逐事件交互相关中位数|收益符号翻转召回|','|---|---|---:|---:|---:|']
 for r in conditions:lines.append(f"|{r['seed']}|{r['method']}|{r['doublecentered']:.4f}|{r['median_event_interaction']:.4f}|{r['sign_flip_recall']:.2%}|")
 lines+=['','沿用E25五个状态、共同合法动作面板，固定页面预测，只更新集合合成；没有重拟合教师来迎合状态。原尺度汇总相关对极端幅度敏感，逐事件中位数与翻转召回同时保留。相对退化的尾部可能由近乎精确的初始状态放大；tail_cases.csv同时记录初始/最终绝对NMSE和收益，不裁剪样本，也不以极端百分比替代绝对误差。','', '## 训练公平性与变量','', '网络仍为712→64 GELU→85，共51157参数；BF16页面系数、输出16维局部响应与质量系数。C、phi逐tensor冻结；初始权重与E27-B相同，训练轨迹/顺序/4608更新/Adam设置/60-20-20集合状态规则相同。每epoch评价，只有Local-Online Set汇总真实收益用于选检查点；教师替换不参与选择。','', '按本轮指定，将原中心化Huber换为mass加权局部平方误差，除以训练集固定median RMS²作单位归一化；没有使用逐页响应尺度消掉mass权重。beta Huber及集合gain/listwise/no-op沿用E27-B归一化。此比较同时包含目标、解析重构和指定局部损失形式的改变，不能仅归因于目标命名；没有开展损失权重搜索。','', '页面编码器只读现有摘要；phi冻结。全局中心由摘要计算，真实页面输出/质量只生成训练标签或离线指标。预测路径不读取原始KV、Dense或教师残差。数学组合使用稳定logmass与expm1，零修正恢复原摘要，不裁剪beta或交换分母。','', '## 开发门槛','']
 for g in gates:lines.append(f"- seed{g['seed']}：通过={g['passed']}；{json.dumps(g['checks'],ensure_ascii=False)}；轨迹收益={g['trajectory_gains']}")
 lines+=['','可靠性中的“显著降低”在运行前量化为退化≤15%、P10和最差均优于E27-B；要求相关和翻转召回也均改善。未根据结果放宽。门槛总判定：'+('通过，允许一次冻结Test评价（完成状态另列）。' if all(g['passed'] for g in gates) else '未通过，Test继续封存；不追加训练、扩大模型或调参。'),'','两个种子的Set汇总收益均优于同信息Independent，且两条开发轨迹汇总收益均为正；但Seed17恢复不足20%、逐例均值和尾部仍失败，两种子的退化比例均超过门槛。事后真实门控仅恢复25.67%/30.97%，仍低于真实beta干预，不能只靠加接受门控解释全部缺口。','', '局部响应监督和解析中心路径确有进展，但真beta是教师信息，不能把46.37%/55.48%当作在线结果。两种教师替换改变了误差抵消，不能据此证明beta是唯一瓶颈或R4缺乏信息。按计划停止追加训练，不更改学习率、层数、特征数或损失权重。','', '尚未进行Kernel、端到端视频或实际延迟验证。']
 (E29/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
