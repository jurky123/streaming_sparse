from .run import *
import csv

def table(name,rows):
 if not rows:return
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
def topagg(rows):
 return dict(n=len(rows),recall=sum(x['best_gain'] for x in rows)/sum(x['oracle_gain'] for x in rows),positive_fraction=sum(x['has_positive'] for x in rows)/len(rows),regret_sum=sum(x['regret'] for x in rows))
def report():
 stages=['A']+(['B'] if all((DEST/f'B_s{s}/complete.json').exists() for s in [17,29]) else [])
 decisions=[];tops=[];quality=[];conds=[];summary=[];ts=[];general=[];cs=[]
 for stage in stages:
  for seed in [17,29]:
   dest=DEST/f'{stage}_s{seed}';assert (dest/'complete.json').exists()
   dr=json.loads((dest/'decisions.json').read_text());tr=json.loads((dest/'topL.json').read_text());qr=json.loads((dest/'quality.json').read_text());cr=json.loads((dest/'conditional.json').read_text());decisions+=dr;tops+=tr;quality+=qr;conds+=cr
   for method in sorted({r['method'] for r in dr}):
    for policy in sorted({r['policy'] for r in dr if r['method']==method}):
     rr=[r for r in dr if r['method']==method and r['policy']==policy];ss=aggregate(rr);ss['worst_absolute_nmse_increase']=max(-x['executed_gain'] for x in rr);summary.append(dict(seed=seed,method=method,policy=policy,**ss))
     for dim in ['trajectory','layer','chunk','step','head']:
      for val in sorted({x[dim] for x in rr}):general.append(dict(seed=seed,method=method,policy=policy,dimension=dim,value=val,**aggregate([r for r in rr if r[dim]==val])))
    for L in [1,4,8]:
     rr=[r for r in tr if r['method']==method and r['L']==L];ts.append(dict(seed=seed,method=method,L=L,trajectory='ALL',**topagg(rr)))
     for traj in sorted({r['trajectory'] for r in rr}):ts.append(dict(seed=seed,method=method,L=L,trajectory=traj,**topagg([r for r in rr if r['trajectory']==traj])))
    cc=[r for r in cr if r['method']==method];tc=[];pc=[];dt=[];dp=[]
    for idx in range(288):
     p=np.load(dest/f'panel_{idx:04d}.npz');t=p['true_gain'];v=p[method];tt=t-t.mean(0);pp=v-v.mean(0);tc.extend(tt.ravel());pc.extend(pp.ravel());dt.extend((tt-tt.mean(1,keepdims=True)).ravel());dp.extend((pp-pp.mean(1,keepdims=True)).ravel())
    out=dict(seed=seed,method=method,conditional_raw=stablecorr(tc,pc),double_centered=stablecorr(dt,dp),median_event_corr=float(np.median([x['conditional_corr'] for x in cc if x['conditional_corr'] is not None])))
    for kind in ['sign_flip','rank_flip']:
     true=sum(x[kind+'_true'] for x in cc);tp=sum(x[kind+'_tp'] for x in cc);out[kind+'_recall']=tp/true if true else None
    cs.append(out)
 def get(s,m):return next(x for x in summary if x['seed']==s and x['method']==m and x['policy']=='Set')
 def top(s,m):return next(x for x in ts if x['seed']==s and x['method']==m and x['L']==8 and x['trajectory']=='ALL')['recall']
 gates=[]
 for seed in [17,29]:
  actual=get(seed,'E29-Q64')['R_executed']-get(seed,'E29-M16')['R_executed'];recall=top(seed,'E29-Q64')-top(seed,'E29-M16');traj=[]
  for t in sorted({x['trajectory'] for x in decisions}):
   rr=lambda m:[x for x in decisions if x['seed']==seed and x['method']==m and x['policy']=='Set' and x['trajectory']==t]
   tt=lambda m:[x for x in tops if x['seed']==seed and x['method']==m and x['L']==8 and x['trajectory']==t]
   traj.append(dict(trajectory=t,actual_delta=aggregate(rr('E29-Q64'))['R_executed']-aggregate(rr('E29-M16'))['R_executed'],top8_delta=topagg(tt('E29-Q64'))['recall']-topagg(tt('E29-M16'))['recall']))
  passed=(actual>=.05 and all(x['actual_delta']>0 for x in traj)) or (recall>=.1 and all(x['top8_delta']>0 for x in traj));gates.append(dict(seed=seed,actual_delta=actual,top8_delta=recall,trajectories=traj,passed=passed))
 gate=dict(passed=all(x['passed'] for x in gates),seeds=gates,rule=json.loads((DEST/'protocol.json').read_text())['A_gate']);dump(DEST/'A_gate.json',gate)
 if 'B' in stages:
  bg=[]
  for seed in [17,29]:
   old=get(seed,'E29-M16');new=get(seed,'E29-Coreset16');ind=next(x for x in summary if x['seed']==seed and x['method']=='E29-Coreset16' and x['policy']=='Independent');checks=dict(actual=new['R_executed']>old['R_executed'],combination=new['R_executed']>ind['R_executed'],top8=top(seed,'E29-Coreset16')>top(seed,'E29-M16'),degradation=new['degraded']<=old['degraded'],p10=new['p10_pct']>=old['p10_pct']);bg.append(dict(seed=seed,checks=checks,passed=all(checks.values())))
  dump(DEST/'B_gate.json',dict(passed=all(x['passed'] for x in bg),seeds=bg))
 table('decision_summary.csv',summary);table('decision_per_event.csv',decisions);table('topL_summary.csv',ts);table('topL_per_event.csv',tops);table('generalization.csv',general);table('information_quality.csv',quality);table('conditional_summary.csv',cs);table('conditional_per_event.csv',conds)
 lines=['# E36：Query 覆盖与集合决策诊断','','冻结 E29 两个模型种子，每种子288个开发 head–tile，来自两条轨迹（不是每条288例）。不训练、不采新QKV、不访问Test。原 Uniform16 Risk、own保护、精算预算与Medium512动作逐事件保持不变。全部实际动作使用真实Q64 NMSE评价。','','|方法|种子|实际Oracle恢复|汇总NMSE改善|逐例平均改善|退化/288|no-op|P10|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for x in summary:
  if x['policy']=='Set':lines.append(f"|{x['method']}|{x['seed']}|{x['R_executed']:.2%}|{x['pooled_improvement_pct']:.3f}%|{x['mean_improvement_pct']:.3f}%|{x['degraded']}|{x['noop']}|{x['p10_pct']:.3f}%|")
 lines+=['','|方法|种子|Top1真门控|Top4事后最佳|Top8事后最佳|','|---|---:|---:|---:|---:|']
 for seed in [17,29]:
  for m in A_METHODS+(B_METHODS if 'B' in stages else []):
   values=[next(x['recall'] for x in ts if x['seed']==seed and x['method']==m and x['L']==L and x['trajectory']=='ALL') for L in [1,4,8]];lines.append(f'|{m}|{seed}|'+ '|'.join(f'{v:.2%}' for v in values)+'|')
 lines+=['','Top-L使用真实标签从提议中选最佳并允许no-op，仅为召回诊断；不能当作在线收益。恢复率统一为Σ收益/Σ原Medium真实最佳正收益，不能与逐例平均相对改善混用。True-M16/True-Q64使用真实128维页面贡献与共同分母；TrueBeta-Q64仅替换质量比，局部响应仍是E29预测16维。','','A预登记要求每个种子实际恢复增加≥5个百分点或Top8增加≥10个百分点，并且对应指标在各自两条开发轨迹都增加。']
 for x in gates:lines.append(f"- seed{x['seed']}：实际恢复变化{x['actual_delta']*100:+.3f}个百分点，Top8变化{x['top8_delta']*100:+.3f}个百分点；门槛{'通过' if x['passed'] else '未通过'}。")
 lines+=['','A整体：'+('通过，进入唯一预登记Coreset16。' if gate['passed'] else '未通过，停止Query选择路线，不运行Coreset16、不训练Query Selector。'),'','逐轨迹和层/chunk/step/head分列见generalization.csv，逐轨迹Top-L见topL_summary.csv。全部逐事件执行、绝对NMSE尾部与no-op记录保留。','', '|方法|种子|汇总双重中心化相关|逐事件条件相关中位数|符号翻转召回|','|---|---:|---:|---:|---:|']
 for x in cs:lines.append(f"|{x['method']}|{x['seed']}|{x['double_centered']:.4f}|{x['median_event_corr']:.4f}|{x['sign_flip_recall']:.2%}|")
 lines+=['','条件诊断沿用E30原五状态共同合法动作面板，先去掉动作跨状态均值，双重中心化另去掉状态主效应。原尺度汇总相关与逐事件相关都保留；不能用总相关掩盖事件差异。预测分数的真实Dense能量正常数仅用于离线NMSE单位对齐，不进入Reader、不改变排序/接受判断。','', '数值路径使用FP64稳定正混合，不直接构造可能溢出的exp(beta)，无beta裁剪。M16分数逐事件重放E30缓存，True-Q64逐动作重放同池Oracle。Q64预测直接读取E30保存的同一冻结模型全64 Query输出，未添加教师输入。','', '成本：Q64把Query相关R4/Reader/候选评分从16扩至64，约4倍，仅作诊断；原Risk仍使用16，不能称整个流水线4倍。候选数512、KV信息、页面缓存均不变。Coreset若获准，只从现有64×128 Q计算距离；16×64×128距离差/平方/求和约0.340 GFLOPs/层（864 head–tiles），均值/首代表与控制另计，不是实测延迟。未运行的Coreset不得声称质量或速度收益。','', '原始来源：E24 C检查点/规范QKV、E29 selected Reader、E27 Adaptive phi、E30 beta/panel全64缓存、E25教师与E35候选召回基线。审计和源文件哈希见audit.json、source_hashes.csv；Test继续封存。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gate,indent=2))
if __name__=='__main__':report()
