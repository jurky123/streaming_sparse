from .run import *

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def report():
 summary=[];general=[];quality=[];conditions=[];decomp=[];offsets=[];rows=[];condsummary=[];qsummary=[];ds=[]
 for seed in [17,29]:
  dest=DEST/f's{seed}';assert (dest/'complete.json').exists();dr=json.loads((dest/'decisions.json').read_text());qr=json.loads((dest/'quality.json').read_text());cr=json.loads((dest/'conditional.json').read_text());dd=json.loads((dest/'decomposition.json').read_text());rows+=dr;quality+=qr;conditions+=cr;decomp+=dd;offsets+=json.loads((dest/'offsets.json').read_text())
  for method in METHODS:
   for policy in ['Set','Independent']:
    rr=[x for x in dr if x['method']==method and x['policy']==policy];summary.append(dict(seed=seed,method=method,policy=policy,**aggregate(rr),worst_absolute_nmse_increase=max(-x['executed_gain'] for x in rr)))
    for dim in ['trajectory','layer','chunk','step','head']:
     for val in sorted({x[dim] for x in rr}):general.append(dict(seed=seed,method=method,policy=policy,dimension=dim,value=val,**aggregate([x for x in rr if x[dim]==val])))
   for group in ['all','candidate','background']:
    for layer in ['ALL',1,15,29]:
     rr=[x for x in qr if x['method']==method and x['group']==group and (layer=='ALL' or x['layer']==layer)];qsummary.append(dict(seed=seed,method=method,group=group,layer=layer,n=len(rr),beta_mae_mean=float(np.mean([x['beta_mae'] for x in rr])),beta_rmse_mean=float(np.mean([x['beta_rmse'] for x in rr])),global_mass2_mse=sum(x['weighted_sse'] for x in rr)/sum(x['weight_sum'] for x in rr),event_mean_mass2_mse=float(np.mean([x['weighted_mse'] for x in rr])),event_mean_bias=float(np.mean([x['weighted_bias'] for x in rr])),event_mean_centered_variance=float(np.mean([x['centered_variance'] for x in rr])),median_gain_spearman=float(np.nanmedian([x['gain_spearman'] for x in rr])),mean_gain_sign_accuracy=float(np.mean([x['gain_sign_accuracy'] for x in rr]))))
   cc=[x for x in cr if x['method']==method];tc=[];pc=[];dt=[];dp=[]
   for i in range(288):
    p=np.load(dest/f'panel_{i:04d}.npz');t=p['true_gain'];v=p[method];tt=t-t.mean(0);pp=v-v.mean(0);tc.extend(tt.ravel());pc.extend(pp.ravel());dt.extend((tt-tt.mean(1,keepdims=True)).ravel());dp.extend((pp-pp.mean(1,keepdims=True)).ravel())
   o=dict(seed=seed,method=method,conditional_raw=stablecorr(tc,pc),double_centered=stablecorr(dt,dp),median_event_corr=float(np.median([x['conditional_corr'] for x in cc if x['conditional_corr'] is not None])))
   for kind in ['sign_flip','rank_flip']:
    n=sum(x[kind+'_true'] for x in cc);o[kind+'_recall']=sum(x[kind+'_tp'] for x in cc)/n if n else None
   condsummary.append(o)
  for layer in ['ALL',1,15,29]:
   rr=[x for x in dd if layer=='ALL' or x['layer']==layer];total=sum(x['total_sse'] for x in rr);ds.append(dict(seed=seed,layer=layer,query_events=len(rr),total_sse=total,global_shared_fraction=sum(x['shared_explained'] for x in rr)/total,global_extra_two_group_fraction=sum(x['candidate_background_extra'] for x in rr)/total,global_remaining_fraction=sum(x['within_group_sse'] for x in rr)/total,balanced_shared_fraction=float(np.mean([x['shared_fraction'] for x in rr])),balanced_two_fraction=float(np.mean([x['two_fraction'] for x in rr]))))
 gates=[]
 for seed in [17,29]:
  get=lambda m:next(x for x in summary if x['seed']==seed and x['method']==m and x['policy']=='Set');old,new,true=get('E29'),get('CandidateBackground-Q'),get('TrueBeta');gap=(new['R_executed']-old['R_executed'])/(true['R_executed']-old['R_executed']);checks=dict(gap=gap>=.5,gain=new['R_executed']>old['R_executed'],degradation=new['degraded']<old['degraded'],p10=new['p10_pct']>=old['p10_pct'],absolute_tail=new['worst_absolute_nmse_increase']<=old['worst_absolute_nmse_increase'],mean_relative=new['mean_improvement_pct']>=old['mean_improvement_pct']);gates.append(dict(seed=seed,gap_recovery=gap,checks=checks,passed=all(checks.values())))
 gate=dict(passed=all(x['passed'] for x in gates),seeds=gates,rule=json.loads((DEST/'protocol.json').read_text())['gate']);dump(DEST/'A_gate.json',gate)
 for name,data in [('decision_summary.csv',summary),('decision_per_event.csv',rows),('generalization.csv',general),('quality_per_event.csv',quality),('quality_summary.csv',qsummary),('conditional_per_event.csv',conditions),('conditional_summary.csv',condsummary),('decomposition_per_query.csv',decomp),('decomposition_summary.csv',ds),('offsets.csv',offsets)]:table(name,data)
 lines=['# E37-A：Query共享β校准的离线能力诊断','','冻结E29两个种子，每种子288个开发head–tile（两条轨迹各144例）。原R4、局部响应、Risk mask、Medium512、M16 probes、一次交换/no-op均保持不变。最终使用原真实Q64 NMSE收益；无训练、新QKV、原始KV读取或Test。','','## 预登记与实现','','选择摘要mass²作为唯一拟合权重，沿用既有质量诊断的尺度口径。对每个Query分别计算 b=Σm²(β真实−β预测)/Σm²；A2在原48候选页与背景页内分别计算。稳定计算采用组内softmax(2logmass)，不裁剪、不搜索偏移或权重、不以动作收益调参。四种方法全部使用相同预测局部响应；教师偏移使用当前Query真实β，不可在线获得。','', '五状态共同面板沿用E30。候选分组固定为原Medium的48页，跨状态不重新选池或重新分组；每次重新计算完整预测集合目标。教师拟合仅涉及原16个probes，不读取其余48 Query标签来拟合。独立评分和Set共享同一β与响应信息。','','## 实际决策','','|方法|seed|Set Oracle恢复|Independent恢复|Set汇总NMSE改善|逐例平均改善|退化/288|no-op|P10|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for seed in [17,29]:
  for m in METHODS:
   x=next(x for x in summary if x['seed']==seed and x['method']==m and x['policy']=='Set');ii=next(x for x in summary if x['seed']==seed and x['method']==m and x['policy']=='Independent');lines.append(f"|{m}|{seed}|{x['R_executed']:.2%}|{ii['R_executed']:.2%}|{x['pooled_improvement_pct']:.3f}%|{x['mean_improvement_pct']:.3f}%|{x['degraded']}|{x['noop']}|{x['p10_pct']:.3f}%|")
 lines+=['','恢复率统一为Σ实际收益/Σ原Medium真实最佳正收益。近零初始NMSE（≤1e−12）不计算相对改善，但绝对收益与退化计数保留。提议动作的真门控恢复、最差相对/绝对尾部、执行率见decision_summary.csv。','','## β误差分解','','对原误差d，严格分解Σw d²=共享解释项+候选/背景额外解释项+组内剩余。这里是相对零误差的MSE分解（包括偏差），不是把全部误差当作已中心化方差。candidate/background的偏差、中心化方差、MSE另在quality表中分列。','','|seed|层|共享解释比例|双组额外解释|剩余|事件/Query均衡双组解释|','|---|---|---:|---:|---:|---:|']
 for x in ds:lines.append(f"|{x['seed']}|{x['layer']}|{x['global_shared_fraction']:.2%}|{x['global_extra_two_group_fraction']:.2%}|{x['global_remaining_fraction']:.2%}|{x['balanced_two_fraction']:.2%}|")
 lines+=['','前三列比例按原mass² SSE汇总；最后一列对事件/Query等权，避免少数高质量上下文主导解释。β误差下降不等于指数质量或动作排序改善。','','## 集合条件信息','','|方法|seed|双重中心化相关|逐事件条件相关中位数|符号翻转召回|','|---|---:|---:|---:|---:|']
 for x in condsummary:lines.append(f"|{x['method']}|{x['seed']}|{x['double_centered']:.4f}|{x['median_event_corr']:.4f}|{x['sign_flip_recall']:.2%}|")
 lines+=['','## 继续门槛','','A2须在两个种子恢复至少50%的E29→TrueBeta执行收益缺口，实际收益增加；可靠性预先落实为退化数严格下降、P10与最差绝对NMSE增量不恶化、逐例平均不恶化。未根据结果改变门槛。']
 for x in gates:lines.append(f"- seed{x['seed']}：缺口恢复{x['gap_recovery']:.2%}；各项检查{x['checks']}；{'通过' if x['passed'] else '未通过'}。")
 lines+=['','整体：'+('通过，允许考虑轻量在线校准B。' if gate['passed'] else '未通过，不启动B，不增加校准组数、参数或其他统计。Test保持封存。'),'','## 解释边界与成本','','这是固定mass²回归下的解析教师诊断，不是这一类模型对真实动作收益的最优上界。失败只限制当前共享偏移方案，不能证明R4信息论不足或所有Query校准必然无效。共同β平移在全精确预测输出的页面softmax中会抵消，但混合输出同时含未修正摘要页，因此仍改变精确/近似相对质量，不能当作无作用的全局归一化。','', '当前阶段没有部署模块。若未来预测两个标量，逐Query–Page加法为16×410×864=5,667,840次/层，约0.00567 GFLOPs；无新增每页缓存或KV读取。聚合特征、校准器、额外扫描与融合成本尚未实现，不能将加法量当作总延迟。','', '来源为E30冻结全Query响应/β缓存与原面板，读取规范事件仅为Dense能量离线单位对齐和元数据；不加载raw KV文件、不重新执行Attention。正Dense归一化常数不进入模型，不影响排序或no-op。详见审计、源哈希与逐事件CSV。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gate,indent=2))
if __name__=='__main__':report()
