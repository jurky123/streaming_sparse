from .run import *
import csv

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def sp(x,y):
 good=[(a,b) for a,b in zip(x,y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
 if len(good)<3:return None
 xx,yy=map(np.array,zip(*good))
 if np.ptp(xx)==0 or np.ptp(yy)==0:return None
 return float(spearmanr(xx,yy).statistic)

def main():
 rows=[];summary=[];cat_summary=[];coverage=[];quality=[];losses=[];conditions=[];weights=[];fits=[];teachers=[];replay=[];general=[];condition_rows=[];panels={};gates=[]
 fit=torch.load(OUT/'fit.pt',weights_only=True)
 for seed in [17,29]:
  dest=DEST/f's{seed}';done=json.loads((dest/'complete.json').read_text());assert done['events']==288 and not done['training'] and not done['test_accessed']
  rr=json.loads((dest/'decisions.json').read_text());rows+=rr;qq=json.loads((dest/'quality.json').read_text());quality+=qq;ll=json.loads((dest/'losses.json').read_text());cc=json.loads((dest/'conditional.json').read_text());condition_rows+=cc;coverage+=json.loads((dest/'coverage.json').read_text());weights+=json.loads((dest/'weight_dominance.json').read_text());fits+=json.loads((dest/'fit_audit.json').read_text());teachers+=json.loads((dest/'teacher_gap.json').read_text())
  pp=[np.load(dest/f'panel_{i:04d}.npz') for i in range(288)];panels[seed]=pp;pmap={str(p['event']):p for p in pp}
  # Use symmetric KL temperature: a perfect TrueBeta student must have zero loss.
  for r in ll:
   p=pmap[r['event']];r.update(decision_loss(p[r['method']],p['True'],fit['gscale']))
  dump(dest/'losses.json',ll);losses+=ll
  methods=list(dict.fromkeys(r['method'] for r in rr))
  for method in methods:
   for policy in ['Set','Independent']:
    sub=[r for r in rr if r['method']==method and r['policy']==policy];summary.append(dict(seed=seed,method=method,policy=policy,**aggregate(sub)))
    if method not in PRIMARY and method!='HistoryPred-CurrentOracle':
     sub=[r for r in sub if r['covered_history_pages']>0]
     if sub:
      ids={r['event'] for r in sub};base=[r for r in rr if r['event'] in ids and r['method']=='HistoryPred-CurrentOracle' and r['policy']==policy]
      cat_summary.append(dict(seed=seed,method=method,policy=policy,scope='covered_events',history_pages=sum(r['history_pages'] for r in sub),covered_pages=sum(r['covered_history_pages'] for r in sub),baseline_gain=sum(r['executed_gain'] for r in base),gain_over_same_events_baseline=sum(r['executed_gain'] for r in sub)-sum(r['executed_gain'] for r in base),baseline_recovery=sum(r['executed_gain'] for r in base)/sum(r['oracle_gain'] for r in base),**aggregate(sub)))
   if method not in PRIMARY:continue
   tt=[];vv=[];med=[]
   for p in pp:
    t=p['true_gain'];v=p[method];t=t-t.mean(0);v=v-v.mean(0);t=t-t.mean(1,keepdims=True);v=v-v.mean(1,keepdims=True);tt.extend(t.ravel());vv.extend(v.ravel());value=stablecorr(t,v)
    if value is not None:med.append(value)
   sub=[r for r in cc if r['method']==method];tr=sum(r['sign_flip_true'] for r in sub);tp=sum(r['sign_flip_tp'] for r in sub);pr=sum(r['sign_flip_pred'] for r in sub)
   conditions.append(dict(seed=seed,method=method,doublecentered=stablecorr(tt,vv),median_event_interaction=float(np.median(med)),sign_flip_recall=tp/tr if tr else None,sign_flip_precision=tp/pr if pr else None))
   for dim in ['trajectory','layer','chunk','step','head']:
    ss=[r for r in rr if r['method']==method and r['policy']=='Set']
    for val in sorted({r[dim] for r in ss}):general.append(dict(seed=seed,method=method,dimension=dim,value=val,**aggregate([r for r in ss if r[dim]==val])))
  old=json.loads((E30/f's{seed}'/'decisions.json').read_text());old={(r['event'],r['method'],r['policy']):r for r in old}
  for r in rr:
   mapping={'Predicted':'Predicted','True':'True','EventOracle':'OracleCoeff64','A0-Shared':'Shared','A0-LOTO':'LOTO'}
   if r['method'] not in mapping:continue
   o=old[(r['event'],mapping[r['method']],r['policy'])];same=all(r[k]==o[k] for k in ['remove','add','execute']);err=abs(r['executed_gain']-o['executed_gain']);assert same and err<1e-12,(seed,r['event'],r['method'],same,err)
   replay.append(dict(seed=seed,event=r['event'],method=r['method'],policy=r['policy'],actions_equal=same,gain_error=err))
  ss={r['method']:r for r in summary if r['seed']==seed and r['policy']=='Set'};co={r['method']:r for r in conditions if r['seed']==seed}
  for scheme in ['A1','A2']:
   x=ss[scheme+'-LOTO'];b=ss['Predicted'];a0=ss['A0-LOTO'];ind=next(r for r in summary if r['seed']==seed and r['method']==scheme+'-LOTO' and r['policy']=='Independent')
   checks=dict(recovery_better_E29=x['R_executed']>b['R_executed'],recovery_better_A0=x['R_executed']>a0['R_executed'],set_better_independent=x['executed_gain_sum']>ind['executed_gain_sum'],degradation_less_E29=x['degraded']<b['degraded'],p10_better_E29=x['p10_pct']>b['p10_pct'],worst_better_E29=x['worst_pct']>b['worst_pct'],conditional_better_E29=co[scheme+'-LOTO']['doublecentered']>co['Predicted']['doublecentered'])
   gates.append(dict(seed=seed,scheme=scheme,checks=checks,passed=all(checks.values()),training_started=False))
 # Losses vs actual executed regret, not just teacher fit.
 qmap={(r['seed'],r['event'],r['method'],r['queries'],r['region']):r for r in quality};rmap={(r['seed'],r['event'],r['method']):r for r in rows if r['policy']=='Set'};joined=[]
 for r in losses:
  d=rmap[(r['seed'],r['event'],r['method'])];x=dict(**r,initial=d['initial'],oracle_gain=d['oracle_gain'],executed_gain=d['executed_gain'],regret_relative_initial=r['true_regret']/d['initial'] if d['initial']>1e-12 else None)
  for region in ['all','candidate','background','history']:
   q=qmap[(r['seed'],r['event'],r['method'],'M16',region)]
   for k in ['beta_mae','quality_sum','quality_weight_normalized']:x[region+'_'+k]=q[k]
  joined.append(x)
 metricnames=['all_beta_mae','candidate_beta_mae','background_beta_mae','all_quality_sum','all_quality_weight_normalized','candidate_quality_weight_normalized','background_quality_weight_normalized','decision_loss','gain_huber','rank_noop_kl'];correlations=[];within=[]
 for seed in [17,29]:
  for method in PRIMARY:
   sub=[r for r in joined if r['seed']==seed and r['method']==method]
   for metric in metricnames:
    correlations.append(dict(seed=seed,method=method,scope='across_events',metric=metric,n=len(sub),spearman_regret_absolute=sp([r[metric] for r in sub],[r['true_regret'] for r in sub]),spearman_regret_relative_initial=sp([r[metric] for r in sub],[r['regret_relative_initial'] for r in sub])))
  methods=['Predicted']+[s+'-'+m for s in SCHEMES for m in ['Shared','LOTO']]
  for event in sorted({r['event'] for r in joined if r['seed']==seed}):
   sub=[r for r in joined if r['seed']==seed and r['event']==event and r['method'] in methods]
   for metric in metricnames:within.append(dict(seed=seed,event=event,metric=metric,methods=len(sub),spearman_loss_regret=sp([r[metric] for r in sub],[r['true_regret'] for r in sub])))
 within_summary=[]
 for seed in [17,29]:
  for metric in metricnames:
   vv=[r['spearman_loss_regret'] for r in within if r['seed']==seed and r['metric']==metric and r['spearman_loss_regret'] is not None]
   within_summary.append(dict(seed=seed,metric=metric,defined_events=len(vv),median_spearman=float(np.median(vv)) if vv else None,positive_fraction=float(np.mean(np.array(vv)>0)) if vv else None))
 qsummary=[];lsummary=[];wsummary=[];covsummary=[]
 for seed in [17,29]:
  for method in PRIMARY:
   for region in ['all','candidate','background','history']:
    sub=[r for r in quality if r['seed']==seed and r['method']==method and r['queries']=='M16' and r['region']==region];qsummary.append(dict(seed=seed,method=method,region=region,**{k:float(np.mean([r[k] for r in sub])) for k in ['beta_mae','beta_rmse','quality_sum','quality_weight_normalized']}))
   sub=[r for r in joined if r['seed']==seed and r['method']==method];lsummary.append(dict(seed=seed,method=method,**{k:float(np.mean([r[k] for r in sub if r[k] is not None])) for k in ['decision_loss','gain_huber','rank_noop_kl','true_regret','gain_spearman','teacher_gain_spearman','sign_accuracy','positive_recall']}))
  sub=[r for r in weights if r['seed']==seed];n=sum(r['pages'] for r in sub);wsummary.append(dict(seed=seed,unique_history_pages=n,mean_max_context_weight=sum(r['max_event_weight_mean']*r['pages'] for r in sub)/n,mean_effective_contexts=sum(r['effective_contexts_mean']*r['pages'] for r in sub)/n,dominated_gt90_fraction=sum(r['dominated_gt90_pages'] for r in sub)/n,dominated_gt99_fraction=sum(r['dominated_gt99_pages'] for r in sub)/n))
  for kind in CATEGORIES+['matched']:
   sub=[r for r in coverage if r['seed']==seed and r['kind']==kind];covsummary.append(dict(seed=seed,kind=kind,covered_events=sum(r['pages']>0 for r in sub),history_page_occurrences=sum(r['history_pages'] for r in sub),covered_page_occurrences=sum(r['pages'] for r in sub),mean_history_mass_fraction=float(np.mean([r['history_mass_fraction'] for r in sub])),candidate_pages=sum(r['candidate_pages'] for r in sub),candidate_history_pages=sum(r['candidate_history_pages'] for r in sub)))
 tails=[]
 for seed in [17,29]:
  for method in PRIMARY:
   sub=[r for r in rows if r['seed']==seed and r['method']==method and r['policy']=='Set' and r['initial']>1e-12];tails.extend(dict(**r,final_nmse=r['initial']-r['executed_gain'],relative_pct=100*r['executed_gain']/r['initial']) for r in sorted(sub,key=lambda r:r['executed_gain']/r['initial'])[:10])
 for name,data in [('decision_per_event',rows),('decision_summary',summary),('context_decision_summary',cat_summary),('coverage_per_event',coverage),('coverage_summary',covsummary),('quality_per_event',quality),('quality_summary',qsummary),('loss_regret_per_event',joined),('loss_summary',lsummary),('loss_regret_correlations',correlations),('within_event_correlations',within),('within_event_summary',within_summary),('conditional_per_event',condition_rows),('conditional_summary',conditions),('weight_dominance',weights),('weight_dominance_summary',wsummary),('fit_audit',fits),('baseline_replay',replay),('generalization',general),('teacher_gap',teachers),('tail_cases',tails)]:table(name+'.csv',data)
 dump(DEST/'continuation_checks.json',dict(seeds=gates,training=False,test_accessed=False,rule='Diagnostic conjunction, not online success. No automatic training.'))
 lines=['# E31：β跨Query复用与决策目标诊断','','两个冻结C种子，各288个开发事件；只做解析拟合和离线损失分析。未训练网络、未采集QKV、未修改摘要/Reader/候选池，Test保持封存。','', '## A：原权重、事件均衡与Key方差特征','', '|方法（Set）|seed17 Oracle恢复|seed29 Oracle恢复|','|---|---:|---:|']
 for method in PRIMARY:
  vv=[next(r for r in summary if r['seed']==s and r['method']==method and r['policy']=='Set') for s in [17,29]];lines.append(f"|{method}|{vv[0]['R_executed']:.2%}|{vv[1]['R_executed']:.2%}|")
 lines+=['','A0完全复现E30原mass²共享拟合。A1每个物理页的每次出现先将64 Query的mass²总权重归一化为1，再合并上下文；事件内仍保持mass²相对权重。A2采用同样均衡权重，仅将F的最后一项phi4替换为未经拟合缩放的Key方差特征，常数和phi1–3不变。固定相对ridge=1e−6×trace(XᵀX)/5，不搜索替换位置/尺度。A0→A1隔离权重，A1→A2隔离特征。','', '为了只检查历史复用，所有Shared/LOTO及分类方法的当前页都使用相同的E30原EventOracle β；A2没有同时改变当前页特征。Predicted是原E29实际方法，其他包含教师。Set/Independent完全共享同一信息。','', '## 原mass²权重是否由少数出现主导','', '|seed|不同历史页|平均最大上下文权重|平均有效上下文数|单上下文>90%页占比|>99%|','|---|---:|---:|---:|---:|---:|']
 for w in wsummary:lines.append(f"|{w['seed']}|{w['unique_history_pages']}|{w['mean_max_context_weight']:.2%}|{w['mean_effective_contexts']:.3f}|{w['dominated_gt90_fraction']:.2%}|{w['dominated_gt99_fraction']:.2%}|")
 lines+=['','有效上下文数=1/Σcontext(weight_fraction²)。按物理页等权，非按出现次数或候选重要性加权。权重不均衡存在不自动说明它是决策失败主因，需要同时看A1的真实收益。','', '## LOTO上下文类别与覆盖','', '|seed|类别|有覆盖事件|覆盖历史页次数|历史页总次数|平均历史mass覆盖|','|---|---|---:|---:|---:|---:|']
 for c in covsummary:lines.append(f"|{c['seed']}|{c['kind']}|{c['covered_events']}|{c['covered_page_occurrences']}|{c['history_page_occurrences']}|{c['mean_history_mass_fraction']:.2%}|")
 lines+=['','tile：同chunk、同step、另一个tile；step：同chunk、其他step；forward：严格更早chunk的同页，允许其他已保存step/tile，不使用未来chunk。后者只诊断长期缓存复用，不能把教师标签变成在线信息。没有可用donor的历史页保持E29预测β；当前页仍是共同事件教师。所有类别参考相同HistoryPred-CurrentOracle，而非把无覆盖页偷偷改为真实β。','', 'matched只保留三类donor都存在的历史页，三种类别在相同目标事件、相同页面支持集比较。普通分类覆盖不同，不能直接把其收益差解释为分布变化。','', '|seed|方法|覆盖事件数|恢复率|同事件基线恢复|相对基线真实增益|','|---|---|---:|---:|---:|---:|']
 for r in cat_summary:
  if r['policy']=='Set':lines.append(f"|{r['seed']}|{r['method']}|{r['n']}|{r['R_executed']:.2%}|{r['baseline_recovery']:.2%}|{r['gain_over_same_events_baseline']:+.6g}|")
 lines+=['','## 真实决策与可靠性','', '|seed|方法|汇总改善|逐例均值|退化/no-op|P10|最差|Independent汇总改善|','|---|---|---:|---:|---|---:|---:|---:|']
 for seed in [17,29]:
  for method in PRIMARY:
   r=next(r for r in summary if r['seed']==seed and r['method']==method and r['policy']=='Set');i=next(r for r in summary if r['seed']==seed and r['method']==method and r['policy']=='Independent');lines.append(f"|{seed}|{method}|{r['pooled_improvement_pct']:+.3f}%|{r['mean_improvement_pct']:+.3f}%|{r['degraded']}/{r['noop']}|{r['p10_pct']:+.3f}%|{r['worst_pct']:+.3f}%|{i['pooled_improvement_pct']:+.3f}%|")
 lines+=['','Oracle恢复为Σ实际Q64收益/Σ同池最佳正收益。相对指标保留原1e−12有效阈值，绝对/汇总指标包含所有事件；tail_cases.csv同时保留初始、最终NMSE和相对退化。','', '## B：质量回归与条件收益蒸馏的对齐程度','', '质量损失保存Σm²误差²及除以Σm²后的版本，按全部/候选/背景/历史分列。决策教师固定E29预测局部响应，仅换真实β；使用既有五状态共同合法动作面板。蒸馏损失为gain Huber＋0.1×包含no-op的KL，teacher/student同温度0.3；尺度为每状态教师RMS与原训练gscale的较大者。这里只计算损失，没有更新任何参数。','', '下面在每个事件内比较7种非True教师基准方案（Predicted及三方案Shared/LOTO）的损失与真实regret，随后汇总事件相关性的中位数。正相关意味着该损失较低时通常regret较低；它仍只是相关，不证明用此损失训练会改善。','', '|seed|损失/误差|有定义事件数|事件内Spearman中位|正相关比例|','|---|---|---:|---:|---:|']
 for r in within_summary:lines.append(f"|{r['seed']}|{r['metric']}|{r['defined_events']}|{r['median_spearman']:.4f}|{r['positive_fraction']:.2%}|")
 lines+=['','loss_regret_correlations.csv另按固定方法跨事件报告与绝对regret和初始NMSE归一化regret的Spearman，避免仅报告跨大误差事件的共同放大。候选/背景损失分别保留，不相加解释因果贡献。TrueBeta是有误差响应下的教师，可能与实际Q64 Oracle不一致；其每状态regret和执行收益保存在teacher_gap.csv。','', '## 集合条件变化','', '|seed|方法|双重中心化相关|事件中位数|符号翻转召回|','|---|---|---:|---:|---:|']
 for r in conditions:lines.append(f"|{r['seed']}|{r['method']}|{r['doublecentered']:.4f}|{r['median_event_interaction']:.4f}|{r['sign_flip_recall']:.2%}|")
 lines+=['','## 是否继续训练','']
 for g in gates:lines.append(f"- seed{g['seed']} {g['scheme']}：诊断继续条件全部满足={g['passed']}；{json.dumps(g['checks'],ensure_ascii=False)}")
 lines+=['','本轮没有训练，也不自动启动训练。事件内Oracle系数不是固定页面预测的上界，Shared/LOTO最小二乘同样不是决策最优上界；负结果不能证明R4无信息。A2只检验预先固定的一个Key方差替换，不否定所有Key统计特征。','', '范围仍是已有两个开发轨迹，不是288条独立视频。forward覆盖只包含保存且SHA一致的跨chunk页面，不能外推所有历史页或未见视频。','', '输出：decision_summary.csv、context_decision_summary.csv、coverage_summary.csv、quality_summary.csv、loss_regret_correlations.csv、within_event_summary.csv、conditional_summary.csv，以及逐事件数据、拟合值和面板。复现见源码README。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
