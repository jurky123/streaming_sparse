from .run import *
from ..e25_decision_diagnostics.report import aggregate
import csv

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 allrows=[];summary=[];infos=[];cond=[];sens=[];pages=[];gates=[];general=[];fits=[];replay=[];audits=[]
 for seed in [17,29]:
  dest=DEST/f's{seed}';done=json.loads((dest/'complete.json').read_text());assert done['events']==288 and not done['training'] and not done['test_accessed']
  rows=json.loads((dest/'decisions.json').read_text());allrows+=rows;infos+=json.loads((dest/'information.json').read_text());sens+=json.loads((dest/'sensitivity.json').read_text());pages+=json.loads((dest/'sensitivity_pages.json').read_text());fits+=json.loads((dest/'fits.json').read_text());audits+=json.loads((dest/'audit.json').read_text())
  cr=json.loads((dest/'conditional.json').read_text());panels=[np.load(dest/f'panel_{i:04d}.npz') for i in range(288)]
  olddone=json.loads((E29/f's{seed}'/'training_complete.json').read_text());old=json.loads((E29/f's{seed}'/f"development_epoch{olddone['best']['epoch']}.json").read_text());old={(r['event'],r['method'],r['policy']):r for r in old}
  for r in rows:
   if r['method'] not in ['Predicted','True']:continue
   before=old[(r['event'],{'Predicted':'Local-Online','True':'Local-TrueBeta'}[r['method']],r['policy'])]
   equal=all(r[k]==before[k] for k in ['remove','add','execute']);err=abs(r['executed_gain']-before['executed_gain']);# Record precision-induced differences; independently audit physical mixtures.
   replay.append(dict(seed=seed,event=r['event'],method=r['method'],policy=r['policy'],actions_match=equal,true_gain_absolute_error=err))
  for method in METHODS:
   for policy in ['Set','Independent']:
    sub=[r for r in rows if r['method']==method and r['policy']==policy];summary.append(dict(seed=seed,method=method,policy=policy,**aggregate(sub)))
   tt=[];pp=[];med=[]
   for p in panels:
    t=p['true_gain'];v=p[method];t=t-t.mean(0);v=v-v.mean(0);t=t-t.mean(1,keepdims=True);v=v-v.mean(1,keepdims=True);tt.extend(t.ravel());pp.extend(v.ravel());cc=stablecorr(t,v)
    if cc is not None:med.append(cc)
   sub=[r for r in cr if r['method']==method];tr=sum(r['sign_flip_true'] for r in sub);pr=sum(r['sign_flip_pred'] for r in sub);tp=sum(r['sign_flip_tp'] for r in sub)
   cond.append(dict(seed=seed,method=method,doublecentered=stablecorr(tt,pp),median_event_interaction=float(np.median(med)),sign_flip_recall=tp/tr if tr else 0.,sign_flip_precision=tp/pr if pr else 0.))
   sub=[r for r in rows if r['method']==method and r['policy']=='Set']
   for dimension in ['trajectory','layer','chunk','step','head']:
    for val in sorted({r[dimension] for r in sub}):general.append(dict(seed=seed,method=method,dimension=dimension,value=val,**aggregate([r for r in sub if r[dimension]==val])))
  ss={r['method']:r for r in summary if r['seed']==seed and r['policy']=='Set'};gap=ss['True']['R_executed']-ss['Predicted']['R_executed'];ratio=(ss['OracleCoeff64']['R_executed']-ss['Predicted']['R_executed'])/gap if gap>0 else None
  gates.append(dict(seed=seed,baseline=ss['Predicted']['R_executed'],true_beta=ss['True']['R_executed'],oracle_coeff=ss['OracleCoeff64']['R_executed'],gap_recovery=ratio,target=ss['Predicted']['R_executed']+.7*gap,passed=ratio is not None and ratio>=.7,true_beta_degradation_fraction=ss['True']['degraded']/288,true_beta_mean_improvement=ss['True']['mean_improvement_pct'],true_beta_p10=ss['True']['p10_pct']))
 info_summary=[]
 for seed in [17,29]:
  for method in METHODS:
   for queries,region in sorted({(r['queries'],r['region']) for r in infos}):
    sub=[r for r in infos if r['seed']==seed and r['method']==method and r['queries']==queries and r['region']==region]
    out=dict(seed=seed,method=method,queries=queries,region=region,events=len(sub))
    keys=[k for k in sub[0] if k not in ['seed','method','queries','region','event','trajectory','layer','head','chunk','step','tile']]
    for k in keys:
     vs=[r[k] for r in sub if r.get(k) is not None and np.isfinite(r[k])]
     if vs:out[k+'_event_mean']=float(np.mean(vs));out[k+'_event_median']=float(np.median(vs))
    info_summary.append(out)
 sensitivity_summary=[]
 for seed in [17,29]:
  for kind in ['all','executed_wrong','executed_positive','noop']:
   sub=[r for r in sens if r['seed']==seed and (kind=='all' or kind=='executed_wrong' and r['executed'] and r['true_gain']<-1e-12 or kind=='executed_positive' and r['executed'] and r['true_gain']>1e-12 or kind=='noop' and not r['executed'])]
   row=dict(seed=seed,subset=kind,events=len(sub))
   for k in ['top1_linear_absolute_share','top4_linear_absolute_share','top8_linear_absolute_share','gradient_abs_max']:
    if sub:row[k+'_median']=float(np.median([r[k] for r in sub]))
   if sub:
    row['top8_same_action_positive_fraction']=float(np.mean([r['top8_same_action_score']>0 for r in sub]));row['true_beta_same_action_positive_fraction']=float(np.mean([r['full_true_beta_same_action_score']>0 for r in sub]))
   sensitivity_summary.append(row)
 tails=[]
 for seed in [17,29]:
  for method in METHODS:
   sub=[r for r in allrows if r['seed']==seed and r['method']==method and r['policy']=='Set' and r['initial']>1e-12]
   tails.extend(dict(**r,final_nmse=r['initial']-r['executed_gain'],relative_improvement_pct=100*r['executed_gain']/r['initial']) for r in sorted(sub,key=lambda r:r['executed_gain']/r['initial'])[:10])
 for name,data in [('decision_per_event',allrows),('decision_summary',summary),('information_per_event',infos),('information_summary',info_summary),('conditional_summary',cond),('generalization',general),('sensitivity_per_event',sens),('sensitivity_pages',pages),('sensitivity_summary',sensitivity_summary),('tail_cases',tails),('fit_audit',fits),('baseline_replay',replay)]:table(name+'.csv',data)
 gate=dict(passed=all(g['passed'] for g in gates),seeds=gates,training_executed=False,test_accessed=False,scope='A1-A3 completed; B not executed',rule='Both seeds recover >=70% of the E29-to-TrueBeta gap')
 dump(DEST/'capacity_gate.json',gate);dump(DEST/'audit.json',dict(events=576,decisions=len(allrows),baseline_replayed=len(replay),max_replay_true_gain_error=max(r['true_gain_absolute_error'] for r in replay),max_counterfactual_relative_error=max(r['replay_relative_error'] for r in audits),max_ridge_kkt=max(r['kkt_max'] for r in fits),max_gradient_fd_relative_error=max(r['gradient_fd_relative_error'] for r in sens),summary_identity_checks=sum(r['history_summary_hashes_checked'] for r in audits),test_accessed=False,training=False))
 lines=['# E30-A：页面质量修正的表达、复用与决策敏感性','','仅完成A1–A3：两个冻结C/E29种子，各288例开发事件。没有训练β分支，没有改变响应Reader、摘要、Adaptive φ、B16、Risk或候选池；Test保持封存。','', '## A1：同一预测响应，只替换β','', '|β来源（Set）|seed17 Oracle恢复|seed29 Oracle恢复|','|---|---:|---:|']
 for method in METHODS[:5]:
  ss=[next(r for r in summary if r['seed']==s and r['method']==method and r['policy']=='Set') for s in [17,29]];lines.append(f"|{method}|{ss[0]['R_executed']:.2%}|{ss[1]['R_executed']:.2%}|")
 lines+=['','OracleCoeff64逐事件使用全部64 Query真实β，Shared使用同页全部出现，LOTO排除当前事件全部64 Query；Shared/LOTO当前页仍使用逐事件Oracle拟合。三者都是教师诊断，不是页面摘要在线预测。历史身份直接复用E28，逐出现核验当前C摘要SHA，每事件另核验原始page0的token/K/V SHA。','', '拟合目标为Σ m²(Fb−β)²＋相对ridge；采用E26固定1e−6×trace(XᵀX)/5和既有训练bscale，不按收益选择。跨事件统一logmass最大值作数值缩放，与原权重相对ridge等价。五项F保持不变。','', '## 70%差距门槛','']
 for g in gates:lines.append(f"- seed{g['seed']}：OracleCoeff恢复{g['oracle_coeff']:.2%}；门槛{g['target']:.2%}；填补TrueBeta差距{g['gap_recovery']:.2%}；通过={g['passed']}。")
 lines+=['','两种子联合容量门槛：'+('通过。仍不是在线成功；本阶段未训练B。' if gate['passed'] else '未通过。不直接启动原五项β Head训练。') ,'', '## A2：候选页与背景页','', '|β来源（Set）|seed17恢复|seed29恢复|','|---|---:|---:|']
 for method in ['Predicted','Candidate-True','Background-True','True']:
  ss=[next(r for r in summary if r['seed']==s and r['method']==method and r['policy']=='Set') for s in [17,29]];lines.append(f"|{method}|{ss[0]['R_executed']:.2%}|{ss[1]['R_executed']:.2%}|")
 lines+=['','候选页为初始Medium所有移出/加入页并集；背景包括其余页及own。替换后重新构造全页残差与质量、集合分子/分母，不只改候选的局部分数。反事实面板保持该初始划分，以隔离同一信息干预。两种局部替换作用不相加，差额可能来自交互及动作离散切换。','', '## 真实质量、同信息Independent与尾部','', '|seed|β|Set汇总改善|Independent汇总改善|逐例均值|退化/no-op|P10|最差|','|---|---|---:|---:|---:|---|---:|---:|']
 for seed in [17,29]:
  for method in METHODS:
   s=next(r for r in summary if r['seed']==seed and r['method']==method and r['policy']=='Set');i=next(r for r in summary if r['seed']==seed and r['method']==method and r['policy']=='Independent');lines.append(f"|{seed}|{method}|{s['pooled_improvement_pct']:+.3f}%|{i['pooled_improvement_pct']:+.3f}%|{s['mean_improvement_pct']:+.3f}%|{s['degraded']}/{s['noop']}|{s['p10_pct']:+.3f}%|{s['worst_pct']:+.3f}%|")
 lines+=['','所有结果使用Q64真实NMSE和同一Medium动作。Oracle恢复是汇总真实收益/汇总同池最佳正收益，不是逐例比值均值。小初始NMSE导致的极端相对退化同时保留绝对初始、最终误差和增益于tail_cases.csv；不裁剪失败实例。Teacher、hindsight都不可部署。','', '## 条件变化','', '|seed|β|双重中心化相关|逐事件中位数|符号翻转召回|','|---|---|---:|---:|---:|']
 for c in cond:lines.append(f"|{c['seed']}|{c['method']}|{c['doublecentered']:.4f}|{c['median_event_interaction']:.4f}|{c['sign_flip_recall']:.2%}|")
 lines+=['','## A3：回归与决策敏感性','', 'information_summary.csv按M16/Q64、全部/候选/背景记录β MAE/RMSE，以及summary/true/predicted页面质量加权误差；权重在每事件所选区域内归一化。quality_mass_l1/l2_relative直接比较exp(logm+β)，避免单独exp(β)溢出。汇总为事件等权均值/中位数，不能混作全页统一质量权重。','', 'action_panel行记录真实Q64交换gain的Pearson/Spearman、I符号准确率（另列匹配M16真值）、B误差、正收益判断与最佳动作regret。I/B采用净残差分子分解，未将稳定输出差公式里的二次项误当成E17的分子能量项。','', '对每例E29最高分提议计算gain对全部M16 β的梯度。页面敏感贡献为Σq |∂gain/∂β·(βtrue−βpred)|，报告top1/4/8集中度；对top8页面另做逐页及累计真实β修正，重新评价同一个提议动作。该排序读取真标签，是诊断而非部署候选机制。线性贡献仅是局部一阶近似，完整真实β变化可高度非线性，不能当作严格因果贡献分解。','', '|seed|事件子集|数量|top1集中度中位|top4|top8|','|---|---|---:|---:|---:|---:|']
 for s in sensitivity_summary:
  if s['events']:lines.append(f"|{s['seed']}|{s['subset']}|{s['events']}|{s['top1_linear_absolute_share_median']:.2%}|{s['top4_linear_absolute_share_median']:.2%}|{s['top8_linear_absolute_share_median']:.2%}|")
 lines+=['','敏感性导数使用FP64重算的同一冻结响应与系数数值；相对原E29混合精度评分的底噪差另存sensitivity_fp64_base_score。每事件对最大梯度坐标执行中心有限差分，误差见audit.json。原始评分/β/设计矩阵/响应/中心和梯度保留在s17、s29目录，便于复核，不在精简包中重复放入全部派生张量。','', '## 结论边界与下一步','', '容量门槛与在线成功是两件事。即使真实β提高平均收益，仍需检查True自身的退化比例和P10；若真实β尚不能满足可靠性，单独β Head也不能被预期为必然解决全部问题。Shared包含评价标签，LOTO仍读取其他上下文标签且当前页有教师，不证明仅R4能预测固定页面系数。','', '本轮不进行训练、特征维度扩展、新QKV或Test评价。若原五项容量未通过，可按方案另做同五项容量的Key方差特征解析对照；不因回归误差下降而直接启动部署。若通过，再依据这些诊断决定是否执行B1/B2，不能将本报告的教师收益写成在线结果。','', '复现：`python -m local_experiments.e30_beta_diagnostics.run --seed 17` / `--seed 29`，然后分别运行`components --seed 17`、`components --seed 29`补充signed-log诊断，再运行`python -m local_experiments.e30_beta_diagnostics.report`、`audit`和`finish`。使用现有项目环境，CUDA；两种子可分卡，BLAS/OMP线程1。完整完成输出受保护，不覆盖；重跑应先选择新的结果版本。']
 interpretation=['## 本轮判断','','1. **事件内五项β表达通过容量门槛，但跨上下文复用没有通过。** OracleCoeff64填补TrueBeta差距97.38%/79.75%，Shared与LOTO却大幅下降。当前Adaptive φ适合本事件拟合，不等于能用一套固定历史页系数覆盖多个Query上下文。这个差异也可能受共享mass加权拟合和Query覆盖影响，不能归因为β目标本身依赖全局中心，更不能据此证明R4没有信息。','','2. **候选页质量准确性在本次干预中更有价值。** Candidate-True恢复38.80%/53.16%，Background-True为26.01%/27.23%；但seed17的背景仍有明显贡献，不能把背景误差设为零。该结果没有证明只预测48页即可部署：仍保留其余页的现有预测，且真实β来自教师。','','3. **较小的平均回归误差并不能保证更好的动作。** Shared/LOTO的普通β MAE低于原预测器，但集合收益明显更差；候选页原始MAE低于背景页，修正候选却更有决策价值。应关注Query/页面敏感性和联合质量效应，而非只优化全页平均MAE。','','4. **错误动作有一定敏感性集中，但不是独有特征。** 错误执行的top8页面一阶影响集中度中位约54.89%/56.05%，正确执行约52.61%/52.91%。当前结果不支持把集中度直接变成可靠门控器。','','5. **即使真实β也未满足完整可靠性标准。** True仍退化50/288、51/288例（17.36%/17.71%），P10约−2.11%/−1.99%。β是有价值的改进方向，但单独精确β不保证当前响应和搜索的全部门槛通过。OracleCoeff64在seed17的条件汇总相关也没有超过原预测器，不能只凭初始单步恢复率声称条件变化已解决。','','本阶段按请求停在A1–A3。没有训练B1/B2，也没有自动增加Key方差特征。若下一步进行β-only训练，应同时面对页面系数跨上下文不稳定和TrueBeta自身可靠性不足这两项限制，不把容量通过当作预计必然成功。','']
 lines[4:4]=interpretation
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
