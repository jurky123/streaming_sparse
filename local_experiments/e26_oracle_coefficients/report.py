from .core import *
from ..e25_decision_diagnostics.report import aggregate
import shutil,tarfile

def table(name,rows):
 if not rows:return
 with (E26/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def stablecorr(x,y):
 x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
 if not len(x):return None
 x=x/max(np.max(abs(x)),1e-300);y=y/max(np.max(abs(y)),1e-300);x=x-x.mean();y=y-y.mean();norm=np.linalg.norm(x)*np.linalg.norm(y)
 return float(x@y/norm) if norm else None

def report():
 rows=[];fits=[];conditions=[];summary=[];condsummary=[];failures=[];done=[];gate=[]
 for seed in [17,29]:
  dest=E26/f's{seed}';done.append(json.loads((dest/'complete.json').read_text()));rr=json.loads((dest/'decisions.json').read_text());rows+=rr;ff=json.loads((dest/'fit.json').read_text());fits+=ff;cc=json.loads((dest/'conditional.json').read_text());conditions+=cc;failures+=json.loads((dest/'numerical_failures.json').read_text())
  for method,policy in sorted({(r['method'],r['policy']) for r in rr}):summary.append(dict(seed=seed,method=method,policy=policy,**aggregate([r for r in rr if r['method']==method and r['policy']==policy])))
  models=sorted({r['method'] for r in cc});stored={i:np.load(dest/f'panel_{i:04d}.npz') for i in range(288)}
  for method in models:
   tc=[];pc=[];trank=[];prank=[];mc=[];md=[]
   for i,p in stored.items():
    if method not in p:continue
    t=p['true_gain'];pred=p[method];t=t-t.mean(0);pred=pred-pred.mean(0);tc.extend(t.ravel());pc.extend(pred.ravel());dt=t-t.mean(1,keepdims=True);dp=pred-pred.mean(1,keepdims=True);trank.extend(dt.ravel());prank.extend(dp.ravel());co=stablecorr(t,pred);di=stablecorr(dt,dp)
    if co is not None:mc.append(co)
    if di is not None:md.append(di)
   sub=[r for r in cc if r['method']==method];row=dict(seed=seed,method=method,n=len(sub),conditional_raw=stablecorr(tc,pc),interaction_doublecentered=stablecorr(trank,prank),median_event_conditional=float(np.median(mc)),median_event_interaction=float(np.median(md)))
   for kind in ['sign_flip','rank_flip']:
    n=sum(r[kind+'_n'] for r in sub);t=sum(r[kind+'_true'] for r in sub);p=sum(r[kind+'_pred'] for r in sub);tp=sum(r[kind+'_tp'] for r in sub);row.update({kind+'_true_fraction':t/n if n else None,kind+'_precision':tp/p if p else None,kind+'_recall':tp/t if t else None})
   condsummary.append(row)
  ss={x['method']:x for x in summary if x['seed']==seed and x['policy']=='Set'};ratio=ss['OracleCoeff_Cphi']['executed_gain_sum']/ss['P1']['executed_gain_sum'];gate.append(dict(seed=seed,OracleCoeff_recovery=ss['OracleCoeff_Cphi']['R_executed'],P1_recovery=ss['P1']['R_executed'],fraction_of_P1_gain=ratio,passed=ss['OracleCoeff_Cphi']['n']==288 and ratio>=.8 and ss['OracleCoeff_Cphi']['executed_gain_sum']>0))
 table('decision_per_event.csv',rows);table('decision_summary.csv',summary);table('conditional_per_event.csv',conditions);table('conditional_summary.csv',condsummary);table('fit_per_event.csv',fits);dump(E26/'numerical_failures.json',failures);dump(E26/'B_gate.json',dict(passed=all(r['passed'] for r in gate),seeds=gate,training_executed=False,reason='Only E26-A authorized in this round; gate is diagnostic'))
 fitrows=[]
 for seed in [17,29]:
  for method in sorted({x['method'] for x in fits}):
   sub=[r for r in fits if r['seed']==seed and r['method']==method];fitrows.append(dict(seed=seed,method=method,n=len(sub),z_nmse_median=float(np.median([r['z_nmse'] for r in sub])),z_nmse_p90=float(np.quantile([r['z_nmse'] for r in sub],.9)),unselected_z_nmse_median=float(np.median([r['z_unselected_nmse'] for r in sub])),beta_nmse_median=float(np.median([r['beta_nmse'] for r in sub])),max_abs_beta=max(r['max_abs_beta'] for r in sub),kkt_max=max((r.get('z_kkt_max',0) for r in sub)),regularized_condition_max=max((r.get('z_regularized_condition_max',0) for r in sub))))
 table('fit_summary.csv',fitrows);general=[]
 for seed in [17,29]:
  for method,policy in sorted({(r['method'],r['policy']) for r in rows}):
   sub=[r for r in rows if r['seed']==seed and r['method']==method and r['policy']==policy]
   for dim in ['trajectory','layer','chunk','step','head']:
    for v in sorted({r[dim] for r in sub}):general.append(dict(seed=seed,method=method,policy=policy,dimension=dim,value=v,**aggregate([r for r in sub if r[dim]==v])))
 table('generalization.csv',general)
 lines=['# E26-A：Oracle页面系数与五项Query因子化','', '仅完成解析能力诊断：两个C种子、每种子288个开发head–tile；没有训练、没有采集QKV、没有访问Test。残差投影基、R4摘要、Risk与Medium候选池均冻结。','', '| 方法（Set） | seed17 Oracle恢复 | seed29 Oracle恢复 |','|---|---:|---:|']
 methods=['E25_D','E24_C','OracleCoeff_Cphi','OracleCoeff_Dphi','Cphi_trueBeta','SVD5_trueBeta','P1','P3']
 for name in methods:
  vals=[next(r for r in summary if r['seed']==seed and r['method']==name and r['policy']=='Set') for seed in [17,29]];lines.append(f"|{name}|{vals[0]['R_executed']:.2%}|{vals[1]['R_executed']:.2%}|")
 lines+=['','Oracle恢复 = 汇总真实64Query NMSE收益 / 汇总同池真实最佳一次正收益。主OracleCoeff使用原C的φ；D重训更新过Query投影，因此另列匹配Dφ的解析拟合，避免把不同φ的比较归因于Page Encoder。所有系数一次性用该事件全部64Query真标签拟合，在初始及全部反事实集合中固定使用，不按状态重新拟合。','', '## 同信息独立选择与执行风险','', '| seed | 方法 | Set汇总改善 | Independent汇总改善 | Set逐例平均改善 | 正收益/退化/no-op | P10 | 最差 |','|---|---|---:|---:|---:|---|---:|---:|']
 for seed in [17,29]:
  for name in ['E25_D','OracleCoeff_Cphi','OracleCoeff_Dphi','Cphi_trueBeta','SVD5_trueBeta']:
   s=next(r for r in summary if r['seed']==seed and r['method']==name and r['policy']=='Set');i=next(r for r in summary if r['seed']==seed and r['method']==name and r['policy']=='Independent');lines.append(f"|{seed}|{name}|{s['pooled_improvement_pct']:+.3f}%|{i['pooled_improvement_pct']:+.3f}%|{s['mean_improvement_pct']:+.3f}%|{s['positive']}/{s['degraded']}/{s['noop']}|{s['p10_pct']:+.3f}%|{s['worst_pct']:+.3f}%|")
 lines+=['','Independent完全共享同一预测z与beta，沿用E24/E25净残差能量排序，一进一出；Set使用共同分母的集合目标。两者允许no-op，实际增益均用真实64Query评价。近零执行动作另列CSV，正负阈值1e-12。逐例百分比仅在初选NMSE>1e-12时计算；汇总绝对指标不排除任何事件。','', '## 跨集合条件变化','', '| seed | 方法 | 动作去均值相关 | 双重中心化交互相关 | 逐例交互相关中位数 | 正负翻转召回 |','|---|---|---:|---:|---:|---:|']
 for r in condsummary:lines.append(f"|{r['seed']}|{r['method']}|{r['conditional_raw']:.4f}|{r['interaction_doublecentered']:.4f}|{r['median_event_interaction']:.4f}|{r['sign_flip_recall']:.2%}|")
 lines+=['','直接复用E25的五个集合与共同合法动作，不新选面板。相关性量纲与E25一致：预测分数除以M16真实Dense平均能量，仅作离线统计，不影响动作/no-op；另保留逐例中位数避免仅看大尺度事件。Oracle系数与SVD全部使用教师信息，不是可部署算法；全64Query拟合包含决策Query，不代表跨Query泛化。','', '## 拟合与数值定义','', '残差设计矩阵为X=diag(m)[1,φ]，质量log-ratio为F=[1,φ]。固定使用E24训练z/beta尺度；正则固定λ=1e-6·trace(XᵀX)/5（质量拟合使用F），包括截距，不通过交换结果选择。通过每页最大logmass缩放X，再在系数中保留反缩放；相对Gram正则使该变换严格等价。目标max缩放仅用于避免数值溢出，解后恢复。Oracle系数保留FP64，未宣称可用现有BF16缓存精度部署。','', '预测质量使用logmass+beta并直接正数求和；不裁剪beta、不固定分母。Independent使用相同预测质量修正。SVD5是真实64×16残差矩阵的Frobenius最佳rank5，搭配真实beta；Cphi_trueBeta只替换质量修正，用于隔离质量拟合损失。SVD5与OracleCoeff都是重建优化，均不是决策目标的严格上界。','', '拟合误差、条件数、KKT残差见fit_summary.csv；独立增强最小二乘与log-sum-exp审计见independent_audit.json。页面系数以缩放值和logmass偏移持久保存，避免反缩放后巨大数值破坏可复现性。','', '## E26-B参考门槛','']
 for g in gate:lines.append(f"- seed{g['seed']}：OracleCoeff收益/P1收益={g['fraction_of_P1_gain']:.2%}；达到80%且真实增益为正：{g['passed']}。")
 lines+=['','本轮按用户最后要求只执行E26-A；无论门槛结果如何，均未启动E26-B。若LS系数效果差，只能说明这套固定φ＋重建拟合未恢复决策收益，不能据此证明任意五项系数都无法优化动作。若SVD5较好而固定φ较差，才支持优先研究Query特征方向；需结合真实beta控制避免混淆。',f'数值未解记录：{len(failures)}，详见numerical_failures.json。主结果若覆盖不足288，不视为通过门槛。']
 (E26/'REPORT.md').write_text('\n'.join(lines)+'\n');dump(E26/'audit.json',dict(runs=done,failures=len(failures),all_events=len({(r['seed'],r['event']) for r in rows}),test_accessed=False,training=False,coefficient_fit_queries=64,decision_queries=16,evaluation_queries=64,source_E24_manifest=sha(OUT/'dataset_manifest.json'),source_E25_protocol=sha(E25/'protocol.json')));print(json.dumps(dict(gate=gate,summary=summary),indent=2))
if __name__=='__main__':report()
