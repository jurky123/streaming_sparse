from .analyze import *

def table(name,rows):
 if not rows:return
 with (E27/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def report():
 rows=[];fits=[];summary=[];condsummary=[];training=[];checkpoints=[];gate=[]
 baselines=json.loads((E26/'s17'/'decisions.json').read_text())+json.loads((E26/'s29'/'decisions.json').read_text())
 rows.extend(r for r in baselines if r['method'] in ['P1','P3','NoOp'])
 for seed in [17,29]:rows.extend(json.loads((E27/f'baselines_stable_s{seed}.json').read_text()))
 for seed in [17,29]:
  dest=E27/f's{seed}';assert (dest/'analysis_complete.json').exists();rows+=json.loads((dest/'decisions.json').read_text());fits+=json.loads((dest/'fit_quality.json').read_text());training.extend(dict(seed=seed,**x) for x in json.loads((dest/'curves.json').read_text()));done=json.loads((dest/'training_complete.json').read_text())
  for ep in [-1,0,1,2,3]:
   rr=json.loads((dest/('development_initial.json' if ep==-1 else f'development_epoch{ep}.json')).read_text());gain=sum(x['set_gain'] for x in rr);ora=sum(x['oracle_gain'] for x in rr);checkpoints.append(dict(seed=seed,epoch=ep,selected=ep==done['best']['epoch'],R=gain/ora,absolute_gain=gain,mean_reconstruction_loss=float(np.mean([r['reconstruction_loss'] for r in rr]))))
  cc=json.loads((dest/'conditional.json').read_text());panels={idx:np.load(dest/f'panel_{idx:04d}.npz') for idx in range(288)}
  for method in ['Adaptive64','Adaptive16','Untrained64']:
   tc=[];pc=[];tr=[];pr=[];med=[]
   for p in panels.values():
    t=p['true_gain'];v=p[method];t=t-t.mean(0);v=v-v.mean(0);tc.extend(t.ravel());pc.extend(v.ravel());t=t-t.mean(1,keepdims=True);v=v-v.mean(1,keepdims=True);tr.extend(t.ravel());pr.extend(v.ravel());co=stablecorr(t,v)
    if co is not None:med.append(co)
   sub=[r for r in cc if r['method']==method];cs=dict(seed=seed,method=method,conditional_raw=stablecorr(tc,pc),doublecentered_raw=stablecorr(tr,pr),median_event_interaction=float(np.median(med)))
   for kind in ['sign_flip','rank_flip']:
    t=sum(r[kind+'_true'] for r in sub);p=sum(r[kind+'_pred'] for r in sub);tp=sum(r[kind+'_tp'] for r in sub);cs.update({kind+'_recall':tp/t if t else None,kind+'_precision':tp/p if p else None})
   condsummary.append(cs)
 for seed in [17,29]:
  sub=[r for r in rows if r['seed']==seed]
  for method,policy in sorted({(r['method'],r['policy']) for r in sub}):summary.append(dict(seed=seed,method=method,policy=policy,**aggregate([r for r in sub if r['method']==method and r['policy']==policy])))
  ss={r['method']:r for r in summary if r['seed']==seed and r['policy']=='Set'};frac=ss['Adaptive64']['executed_gain_sum']/ss['P1']['executed_gain_sum'];improve=ss['Adaptive64']['R_executed']-ss['Cphi_trueBeta']['R_executed'];gate.append(dict(seed=seed,adaptive_recovery=ss['Adaptive64']['R_executed'],P1_fraction=frac,improvement_over_fixed_pp=100*improve,passed=frac>=.8 and improve>=.05 and ss['Adaptive64']['executed_gain_sum']>0))
 dump(E27/'B_gate.json',dict(passed=all(r['passed'] for r in gate),seeds=gate,rule='both seeds >=80% P1 gains and >=5 percentage points Oracle recovery over Cphi_trueBeta, positive gain'))
 table('decision_per_event.csv',rows);table('decision_summary.csv',summary);table('fit_per_event.csv',fits);table('conditional_summary.csv',condsummary);table('training_curves.csv',training);table('checkpoints.csv',checkpoints)
 general=[]
 for seed in [17,29]:
  for method in ['Adaptive64','Adaptive16','Untrained64','Cphi_trueBeta','SVD5_trueBeta','P1']:
   rr=[r for r in rows if r['seed']==seed and r['method']==method and r['policy']=='Set']
   for dim in ['trajectory','layer','chunk','step','head']:
    for val in sorted({r[dim] for r in rr}):general.append(dict(seed=seed,method=method,dimension=dim,value=val,**aggregate([r for r in rr if r[dim]==val])))
 table('generalization.csv',general)
 cost=dict(page_coeff_bytes=64,history_cache_MiB=64*338*12*29/2**20,query_projection_GFLOPs=2*128*8*16*864/1e9,page_query_products_GFLOPs=2*8*4*16*410*864/1e9,page_encoder_current_GFLOPs=2*(712*64+64*32)*72*12/1e9,page_encoder_allpages_GFLOPs=2*(712*64+64*32)*410*12/1e9,extra_raw_KV_reads=0,combined_existing85_and_new32_BF16_coeff_bytes=234);table('cost_estimate.csv',[cost])
 lines=['# E27：页面相关Query方向与条件性系数预测','', '冻结E24 C摘要、Risk与16维基；两个种子分别使用8条训练轨迹/1152事件训练U/G，开发集2轨迹/288事件选择检查点。没有新增QKV，Test封存。本节A表使用真实beta和逐事件Oracle LS系数，不是在线方法；若执行B，其预测系数结果单独列出。','', '| Set方法 | seed17 Oracle恢复 | seed29 Oracle恢复 |','|---|---:|---:|']
 for name in ['Cphi_trueBeta','Untrained64','Adaptive64','Adaptive16','SVD5_trueBeta','P1']:
  vals=[next(r for r in summary if r['seed']==s and r['method']==name and r['policy']=='Set') for s in [17,29]];lines.append(f"|{name}|{vals[0]['R_executed']:.2%}|{vals[1]['R_executed']:.2%}|")
 lines+=['','Untrained64使用完全相同的随机初始化新结构，仅作结构基线，不参与检查点选择。Adaptive64在全部64Query真标签上拟合Oracle系数，在固定M16上选择动作；Adaptive16只在16 probes上拟合系数，再用全部64Query检查残差和最终真实动作收益。两者都仍使用教师信息，不叫在线预测。','', '| seed | Adaptive64 Set汇总改善 | 同信息Independent汇总改善 | Set逐例平均改善 | 正收益/退化/no-op | P10 | 最差 |','|---|---:|---:|---:|---|---:|---:|']
 for seed in [17,29]:
  s=next(r for r in summary if r['seed']==seed and r['method']=='Adaptive64' and r['policy']=='Set');i=next(r for r in summary if r['seed']==seed and r['method']=='Adaptive64' and r['policy']=='Independent');lines.append(f"|{seed}|{s['pooled_improvement_pct']:+.3f}%|{i['pooled_improvement_pct']:+.3f}%|{s['mean_improvement_pct']:+.3f}%|{s['positive']}/{s['degraded']}/{s['noop']}|{s['p10_pct']:+.3f}%|{s['worst_pct']:+.3f}%|")
 lines+=['','## 条件变化','', '| seed | 方法 | 去动作均值相关 | 双重中心化相关 | 逐事件交互相关中位数 | 正负翻转召回 |','|---|---|---:|---:|---:|---:|']
 for r in condsummary:lines.append(f"|{r['seed']}|{r['method']}|{r['conditional_raw']:.4f}|{r['doublecentered_raw']:.4f}|{r['median_event_interaction']:.4f}|{r['sign_flip_recall']:.2%}|")
 lines+=['','正/负/no-op计数之外还有绝对收益≤1e-12的已执行近零动作，完整数量见CSV。原样使用E25共同合法面板与五个集合。同一事件的Oracle系数在不同集合间固定，不重新拟合。相关性沿用E25的单位与汇总方法，原尺度相关可能被极端幅度主导，因此保留逐事件中位数。不能只用静态排序或残差NMSE宣称集合条件学习成功。','', '## 训练与选择','', '页面映射712→64 GELU→32，输出8×4 BF16系数；共享Query128→8，phi=tanh(uA)。不增加原始KV读取，也不额外使用R4 logits。四轮4608更新，Adam1e-4，梯度clip1；不搜索超参。内层沿用E26质量因子设计X=mass[1,phi]和相对ridge1e-6，通过FP64五维solve求导。','', '外层按页面目标能量归一化，并用64×16×原训练zscale²作分母下限，防止巨大残差主导；这是明确的训练加权选择，不等同于未归一化总MSE。仅使用训练真标签求梯度；开发集LS系数是评价教师，未用于更新U/G。检查点按四个训练epoch中最大开发汇总真实交换收益选择，初始随机结构不参与选择。每个epoch均保留。','', '## 进入B的参考门槛','']
 for g in gate:lines.append(f"- seed{g['seed']}：恢复P1的{g['P1_fraction']:.2%}，比固定Cphi_trueBeta高{g['improvement_over_fixed_pp']:+.2f}个百分点；通过={g['passed']}。")
 lines+=['','理论新增页面系数64B，历史29层约7.18MiB；Query–Page主乘加约0.363GFLOPs/层，另含Query投影和页面编码。与原85个BF16读出系数合计234B/page，不能声称仍满足E24的192B目标。这里只核算新增特征，不把Oracle LS当作可在线获得，也没有实测融合延迟。','', '数据仍是研究中使用过的开发场景；只说明该冻结分布下的诊断结果，不是Test泛化结论。']
 (E27/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gate,indent=2))
if __name__=='__main__':report()
