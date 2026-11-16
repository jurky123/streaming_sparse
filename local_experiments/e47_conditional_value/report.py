from .run import *
from ..e41_training_signal.report import decision_summary
import shutil,tarfile
E45=ROOT/'artifacts/e45_frozen_replay_v1'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run():
 summaries=[];values=[];conditions=[];eventvalues=[]
 for seed in [17,29]:
  dest=D/f's{seed}';assert json.loads((dest/'audit.json').read_text())['passed'];npz=np.load(dest/'scores.npz');old=np.load(E45/f's{seed}/scores.npz');assert set(npz.files)==set(old.files)
  for key in npz.files:
   if any(z in key for z in ['truth','mask','remove','add']):assert np.array_equal(npz[key],old[key]),key
  for model,source in [('E43_P',E45),('E44_Pset',D)]:
   rows=[r for r in read(source/f's{seed}/decisions.csv') if r['arm']=='P']
   for role in ['Risk','BgSwap_train','BgSwap_holdout','trained_combined']:
    for policy in ['Set','Independent','Reuse_other']:
     rr=[r for r in rows if r['policy']==policy and (r['state_role']==role or role=='trained_combined' and r['state_role']!='BgSwap_holdout')];summaries.append(dict(seed=seed,model=model,state_role=role,policy=policy,**decision_summary(rr)))
  for e in range(48):
   g=np.stack([npz[f'{e}_truth_{s}'] for s in range(3)]);assert g.shape==(3,512)
   for scope,ix in [('all3',[0,1,2]),('trained2',[0,1]),('risk_holdout',[0,2])]:
    t=g[ix];weights=np.ones(len(ix))/len(ix);adapt=float(np.maximum(0,t.max(1))@weights);fixed=float((np.maximum(0,t)*weights[:,None]).sum(0).max());conditional=adapt-fixed;assert conditional>=-1e-12
    eventvalues.append(dict(seed=seed,event_index=e,event=read(dest/'decisions.csv')[e*27]['event'],scope=scope,V_adaptive=adapt,V_fixedplus=fixed,V_conditional=max(0.,conditional),conditional_fraction=conditional/adapt if adapt>0 else 0.))
  for scope in ['all3','trained2','risk_holdout']:
   v=[r for r in eventvalues if r['seed']==seed and r['scope']==scope];a=sum(r['V_adaptive'] for r in v);c=sum(r['V_conditional'] for r in v);values.append(dict(seed=seed,scope=scope,events=len(v),V_adaptive=a,V_fixedplus=sum(r['V_fixedplus'] for r in v),V_conditional=c,conditional_fraction_pct=100*c/a if a else None,events_with_positive_conditional=sum(r['V_conditional']>1e-12 for r in v),max_event_fraction=max(r['V_conditional'] for r in v)/c if c>0 else 0.))
  for model,source in [('E43_P',old),('E44_Pset',npz)]:
   p=np.stack([np.stack([source[f'{e}_P_{s}'] for s in [0,1]]) for e in range(48)]);t=np.stack([np.stack([source[f'{e}_truth_{s}'] for s in [0,1]]) for e in range(48)]);tc=t-t.mean(1,keepdims=True);pc=p-p.mean(1,keepdims=True);tc-=tc.mean(2,keepdims=True);pc-=pc.mean(2,keepdims=True);conditions.append(dict(seed=seed,model=model,double_center_corr=float(np.corrcoef(tc.ravel(),pc.ravel())[0,1]),same_action_count=sum((int(p[e,0].argmax()) if p[e,0].max()>0 else -1)==(int(p[e,1].argmax()) if p[e,1].max()>0 else -1) for e in range(48))))
 table(D/'summary.csv',summaries);table(D/'conditional_value.csv',values);table(D/'conditional_value_events.csv',eventvalues);table(D/'condition_summary.csv',conditions)
 lines=['# E47：P_set冻结回放与真实条件价值分母','', '本轮只使用E44最终P_set，沿用E45锁定的48事件、96训练状态及48邻近留出状态；原E43 P直接重放作参照。无训练、新状态、新Query或Test评价。候选与真实收益矩阵逐位匹配E45；P_set在原96状态的评分逐位匹配E44最终矩阵。','', 'V_adaptive对每个事件的状态等权，逐状态选择真实最佳动作/no-op；V_fixed+选一个固定动作，但每个状态允许真实门控拒绝。差值仅度量排除接受判断后的“改变动作”价值，是离线教师参照，不能部署。不把它当模型收益的可加分解，也不拿不同状态的恢复率直接平均。','', '|seed|状态范围|V_adaptive|V_fixed+|V_conditional|占真实自适应收益|有条件价值事件|','|---|---|---:|---:|---:|---:|---:|']
 for r in values:lines.append('|%d|%s|%.6g|%.6g|%.6g|%.3f%%|%d/48|'%(r['seed'],r['scope'],r['V_adaptive'],r['V_fixedplus'],r['V_conditional'],r['conditional_fraction_pct'],r['events_with_positive_conditional']))
 lines+=['','|seed|模型|状态|策略|恢复率|H_loss|退化|P10|','|---|---|---|---|---:|---:|---:|---:|']
 for r in summaries:lines.append('|%d|%s|%s|%s|%.2f%%|%.6f|%d/%d|%.2f%%|'%(r['seed'],r['model'],r['state_role'],r['policy'],r['recovery_pct'],r['H_loss'],r['harm'],r['events'],r['p10_relative_pct']))
 lines+=['','完整动作索引/端点、评分、true gains及mask均保存。Reuse_other接受判断也来自偏好源状态，不能用目标真值挑动作。Independent与Set使用相同模型信息。V_conditional很小时，本协议主要是邻近稳定性检查；不能据小增量否定其他大范围集合条件价值。V_conditional明显而模型收益差时，也须区分页面预测、动作提议与门控，不直接新增模块。','', 'P_set输出是结构化决策代理，未默认其响应和β物理校准。所有结果限于已见Query/页面和同候选池，不称跨Query或跨轨迹泛化。Test封存。']
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n');out=ROOT/'docs/research/evidence/e47_conditional_value_v1';out.mkdir(parents=True,exist_ok=True)
 for n in ['REPORT.md','summary.csv','conditional_value.csv','conditional_value_events.csv','condition_summary.csv']:shutil.copy2(D/n,out/n)
 for seed in [17,29]:
  for n in ['protocol.json','audit.json','decisions.csv','conditional.csv']:shutil.copy2(D/f's{seed}'/n,out/f's{seed}_{n}')
 with tarfile.open(D/'e47_review.tar.gz','w:gz') as tf:
  for p in D.rglob('*'):
   if p.is_file() and p.suffix in ['.csv','.json','.npz','.md']:tf.add(p,arcname=str(p.relative_to(D)))
 print('REPORT DONE',flush=True)
if __name__=='__main__':run()
