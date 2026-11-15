from .report import *

def report_b():
 assert json.loads((E27/'B_gate.json').read_text())['passed'];rows=[];conds=[];curves=[];cks=[];summary=[]
 for seed in [17,29]:
  dest=E27/f'B_s{seed}';done=json.loads((dest/'training_complete.json').read_text());assert (dest/'analysis_complete.json').exists();rows+=json.loads((dest/f'development_epoch{done["best"]["epoch"]}.json').read_text());curves.extend(dict(seed=seed,**r) for r in json.loads((dest/'curves.json').read_text()))
  for ep in range(4):
   rr=json.loads((dest/f'development_epoch{ep}.json').read_text())
   for met in ['PredictedBeta','TrueBeta']:
    cks.append(dict(seed=seed,epoch=ep,selected=ep==done['best']['epoch'],method=met,**aggregate([r for r in rr if r['method']==met and r['policy']=='Set'])))
  cr=json.loads((dest/'conditional.json').read_text());panels={i:np.load(dest/f'panel_{i:04d}.npz') for i in range(288)}
  for name in ['PredictedBeta','TrueBeta']:
   tc=[];pc=[];tr=[];pr=[];med=[]
   for p in panels.values():
    t=p['true_gain'];v=p[name];t=t-t.mean(0);v=v-v.mean(0);tc.extend(t.ravel());pc.extend(v.ravel());t=t-t.mean(1,keepdims=True);v=v-v.mean(1,keepdims=True);tr.extend(t.ravel());pr.extend(v.ravel());co=stablecorr(t,v)
    if co is not None:med.append(co)
   sub=[r for r in cr if r['method']==name];t=sum(r['sign_flip_true'] for r in sub);tp=sum(r['sign_flip_tp'] for r in sub);pred=sum(r['sign_flip_pred'] for r in sub);conds.append(dict(seed=seed,method=name,conditional_raw=stablecorr(tc,pc),doublecentered_raw=stablecorr(tr,pr),median_event_interaction=float(np.median(med)),sign_flip_recall=tp/t if t else None,sign_flip_precision=tp/pred if pred else None))
 for seed in [17,29]:
  for met in ['PredictedBeta','TrueBeta']:
   for policy in ['Set','Independent']:summary.append(dict(seed=seed,method=met,policy=policy,**aggregate([r for r in rows if r['seed']==seed and r['method']==met and r['policy']==policy])))
 table('B_decision_per_event.csv',rows);table('B_decision_summary.csv',summary);table('B_checkpoints.csv',cks);table('B_training_curves.csv',curves);table('B_conditional_summary.csv',conds);table('B_information.csv',[r for s in [17,29] for r in json.loads((E27/f'B_s{s}'/'information.json').read_text())]);gen=[]
 for seed in [17,29]:
  for met in ['PredictedBeta','TrueBeta']:
   rr=[r for r in rows if r['seed']==seed and r['method']==met and r['policy']=='Set']
   for dim in ['trajectory','layer','chunk','step','head']:
    for v in sorted({r[dim] for r in rr}):gen.append(dict(seed=seed,method=met,dimension=dim,value=v,**aggregate([r for r in rr if r[dim]==v])))
 table('B_generalization.csv',gen)
 with (E27/'REPORT.md').open('a') as f:
  f.write('\n## E27-B：冻结新φ，预测页面系数\n\n两个A种子通过参考门槛后执行；C摘要、U/G和投影基冻结。仅训练原容量712→64→85 Page Encoder，输出85个BF16系数。每种子四轮4608更新，按真实β未知时PredictedBeta Set的开发汇总收益选择；TrueBeta不参与检查点选择。Test仍封存。\n\n|seed|信息|Set Oracle恢复|提议＋真门控恢复|Set汇总改善|Independent汇总改善|逐例平均改善|正/负/no-op|P10|最差|\n|---|---|---:|---:|---:|---:|---:|---|---:|---:|\n')
  for seed in [17,29]:
   for met in ['PredictedBeta','TrueBeta']:
    s=next(r for r in summary if r['seed']==seed and r['method']==met and r['policy']=='Set');i=next(r for r in summary if r['seed']==seed and r['method']==met and r['policy']=='Independent');f.write(f"|{seed}|{met}|{s['R_executed']:.2%}|{s['R_proposal']:.2%}|{s['pooled_improvement_pct']:+.3f}%|{i['pooled_improvement_pct']:+.3f}%|{s['mean_improvement_pct']:+.3f}%|{s['positive']}/{s['degraded']}/{s['noop']}|{s['p10_pct']:+.3f}%|{s['worst_pct']:+.3f}%|\n")
  f.write('\nB训练使用有符号残差和beta的Huber监督、真实交换收益Huber及包含no-op的listwise项。为处理此前极大动态范围，残差/beta使用每个训练页面真实目标RMS与原训练尺度二者较大值作尺度，收益使用训练状态真实候选RMS与原训练gain尺度较大值；这些尺度只用于训练损失，不进入推理。没有裁剪beta或预测分母，没有训练额外门控网络。归一化和预算在B运行前固定，未根据B结果搜索。\n\nB不改变初选；所有本轮改善均相对对应冻结C的Risk初选，不把E24已有的初选改善计入新收益。TrueBeta是教师替换诊断，不是部署结果，也不保证优于预测beta（两个分支误差可能相互补偿）。跨集合相关性见B_conditional_summary.csv；所有同信息Independent/Set共享相同z与beta。\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':report_b()
