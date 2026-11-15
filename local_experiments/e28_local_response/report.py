from .run import *
from ..e25_decision_diagnostics.report import aggregate
from ..e26_oracle_coefficients.report import stablecorr
import csv

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 rows=[];summary=[];fitrows=[];idsummary=[];conditions=[];general=[]
 for seed in [17,29]:
  dest=DEST/f's{seed}';assert (dest/'complete.json').exists();rr=json.loads((dest/'decisions.json').read_text());rows+=rr;fitrows+=json.loads((dest/'fits.json').read_text());ident=json.loads((dest/'identities.json').read_text());methods=sorted({x['method'] for x in rr})
  for method in methods:
   for policy in ['Set','Independent']:summary.append(dict(seed=seed,method=method,policy=policy,**aggregate([r for r in rr if r['method']==method and r['policy']==policy])))
  old=json.loads((E27/f's{seed}'/'decisions.json').read_text());summary.append(dict(seed=seed,method='E27-Adaptive64',policy='Set',**aggregate([r for r in old if r['method']=='Adaptive64' and r['policy']=='Set'])))
  old=json.loads((E25/f'C_s{seed}'/'ladder.json').read_text());summary.append(dict(seed=seed,method='P1',policy='Set',**aggregate([r for r in old if r['method']=='P1'])))
  eventmeta={r['event']:r for r in rr}
  def field(o,n):
   # event metadata lookup avoids parsing token identity from filenames.
   return eventmeta[o['event']][n]
  count=collections.Counter(len(x['occurrences']) for x in ident);crosschunk=sum(len({field(o,'chunk') for o in x['occurrences']})>1 for x in ident);crossstep=sum(len({field(o,'step') for o in x['occurrences']})>1 for x in ident)
  idsummary.append(dict(seed=seed,unique_history_pages=len(ident),history_occurrences=sum(len(x['occurrences']) for x in ident),repeated_unique=sum(len(x['occurrences'])>1 for x in ident),repeat_occurrence_coverage=sum(len(x['occurrences']) for x in ident if len(x['occurrences'])>1)/sum(len(x['occurrences']) for x in ident),cross_chunk_pages=crosschunk,cross_step_pages=crossstep,occurrence_histogram=dict(count)))
  cr=json.loads((dest/'conditional.json').read_text());panels=[np.load(dest/f'panel_{i:04d}.npz') for i in range(288)]
  for method in methods:
   tt=[];pp=[];med=[]
   for p in panels:
    t=p['true_gain'];v=p[method];t=t-t.mean(0);v=v-v.mean(0);t=t-t.mean(1,keepdims=True);v=v-v.mean(1,keepdims=True);tt.extend(t.ravel());pp.extend(v.ravel());med.append(stablecorr(t,v))
   sub=[x for x in cr if x['method']==method];tr=sum(x['sign_flip_true'] for x in sub);tp=sum(x['sign_flip_tp'] for x in sub)
   conditions.append(dict(seed=seed,method=method,doublecentered=stablecorr(tt,pp),median_event=float(np.median([x for x in med if x is not None])),sign_flip_recall=tp/tr if tr else None))
  for method in methods:
   sub=[r for r in rr if r['method']==method and r['policy']=='Set']
   for dimension in ['trajectory','layer','chunk','step','head']:
    for v in sorted({r[dimension] for r in sub}):general.append(dict(seed=seed,method=method,dimension=dimension,value=v,**aggregate([r for r in sub if r[dimension]==v])))
 fs=[]
 for seed in [17,29]:
  for target in ['Centered','Local']:
   for mode in ['Shared','LOTO']:
    sub=[x for x in fitrows if x['seed']==seed and x['target']==target and x['mode']==mode]
    fs.append(dict(seed=seed,target=target,mode=mode,blocks=len(sub),page_occurrences=sum(x['pages'] for x in sub),target_nmse_median=float(np.median([x['target_nmse'] for x in sub])),residual_nmse_median=float(np.median([x['residual_nmse'] for x in sub])),coefficient_shift_median=float(np.median([x['coefficient_shift'] for x in sub]))))
 table('decision_per_event.csv',rows);table('decision_summary.csv',summary);table('fit_per_block.csv',fitrows);table('fit_summary.csv',fs);table('conditional_summary.csv',conditions);table('generalization.csv',general);dump(DEST/'identity_summary.json',idsummary)
 lines=['# E28：页面局部响应与跨 Query 共享诊断','', '只执行解析诊断，没有训练新网络、没有更新C摘要/Adaptive phi/B16，没有新增QKV，Test保持封存。两个C种子分别评价已有两条开发轨迹288个head–tile。不是288条独立视频。所有方法真实beta、M16决策、真实Q64评价、相同新Risk与Medium合法候选池。','', '|方法（Set）|seed17 Oracle恢复|seed29 Oracle恢复|','|---|---:|---:|']
 for method in ['E27-Adaptive64','Centered-Oracle64','Local-Oracle64','Centered-Shared','Local-Shared','Centered-LOTO','Local-LOTO','P1']:
  vals=[next(x for x in summary if x['seed']==s and x['method']==method and x['policy']=='Set') for s in [17,29]];lines.append(f"|{method}|{vals[0]['R_executed']:.2%}|{vals[1]['R_executed']:.2%}|")
 lines+=['','Oracle恢复=Σ真实执行收益/Σ同池最佳正收益，所有行分母一致。P1和解析LS不是严格决策上界。Shared共同拟合使用评价Query的真实标签；LOTO拟合排除该事件全部64个Query，但使用其他匹配事件标签，仍是教师诊断，不是在线Page Encoder。当前页各方法均逐事件拟合，仅替换历史页，因而LOTO不是全页无标签测试。','', '## 页面身份和共享范围','']
 for x in idsummary:lines.append(f"- seed{x['seed']}：{x['unique_history_pages']}个不同历史页，{x['history_occurrences']}次出现；可共享覆盖{x['repeat_occurrence_coverage']:.2%}；跨step {x['cross_step_pages']}页，跨chunk {x['cross_chunk_pages']}页。出现次数分布{x['occurrence_histogram']}。")
 lines+=['','身份在同轨迹/layer/head内，联合核对有序token ID、实际BF16 K/V、当前C摘要全部字段的SHA256。不同page编号不会被误当作相同页面。没有凭相同token ID忽略K/V或摘要变化。每页的全部出现与哈希保存在identities.json。','', '## 拟合定义','', '原目标Y=t/m；局部目标Y=Bᵀ(O_page−Ohat_page)。主比较共同使用E26的mass加权LS：X=m[1,phi]，拟合mY，相对ridge=1e−6×trace(XᵀX)/5。不因换成局部目标而偷偷改成无权LS。固定尺度只作数值缩放，不改变线性解；跨事件共享时最大mass缩放在所有拟合Query上统一计算。','', 'Local按精确恒等式z=exp(logm+beta)×r+(exp(logm+beta)−exp(logm))×Bᵀ(Ohat_page−Ohat_all)重构，低beta用expm1避免抵消。使用canonical page_output和page_logz。系数与beta不依赖当前精算集合；五个反事实集合共用同一预测。','', '|seed|目标|拟合|目标NMSE中位数|净残差NMSE中位数|归一化系数形状差中位数|','|---|---|---|---:|---:|---:|']
 for x in fs:lines.append(f"|{x['seed']}|{x['target']}|{x['mode']}|{x['target_nmse_median']:.4g}|{x['residual_nmse_median']:.4g}|{x['coefficient_shift_median']:.4g}|")
 lines+=['','拟合误差统计单位是同一出现模式的页块，每块大小和重复次数在CSV中记录。两种目标NMSE分母不同，不能仅横比target NMSE判定局部目标更优；相同净残差指标和真实动作收益才是共同评价。系数先逐页最大绝对值归一化后比较形状，避免不同缩放基准干扰；该指标不测幅度稳定性，也受病态设计影响，不能单独当作信息缺失证据。原目标误差在t空间计算，避免t/m溢出；两种目标误差不直接等价。','', '|seed|方法|Set汇总NMSE改善|Independent汇总改善|Set逐例平均|退化/no-op|','|---|---|---:|---:|---:|---:|']
 for seed in [17,29]:
  for method in ['Local-Oracle64','Centered-Shared','Local-Shared','Centered-LOTO','Local-LOTO']:
   x=next(x for x in summary if x['seed']==seed and x['method']==method and x['policy']=='Set');i=next(x for x in summary if x['seed']==seed and x['method']==method and x['policy']=='Independent');lines.append(f"|{seed}|{method}|{x['pooled_improvement_pct']:+.3f}%|{i['pooled_improvement_pct']:+.3f}%|{x['mean_improvement_pct']:+.3f}%|{x['degraded']}/{x['noop']}|")
 lines+=['','## 条件交互','', '|seed|方法|双重中心化相关|逐事件中位数|符号翻转召回|','|---|---|---:|---:|---:|']
 for x in conditions:lines.append(f"|{x['seed']}|{x['method']}|{x['doublecentered']:.4f}|{x['median_event']:.4f}|{x['sign_flip_recall']:.2%}|")
 lines+=['','全部范围按trajectory/layer/chunk/step/head分列于generalization.csv；每事件执行、正负收益、no-op、P10与最差值见decision_summary.csv。未调用Test，不根据这些诊断选择或训练新模型。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
