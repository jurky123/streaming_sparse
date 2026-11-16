from .core import *
from .report import read
import shutil,tarfile

def finish():
 for seed in [17,29]:assert json.loads((D/f's{seed}/audit.json').read_text())['passed']
 rows=read(D/'summary.csv');cond=read(D/'conditional_summary.csv');tail=[];trend=[]
 for seed in [17,29]:
  for step in [0,384,768,1152,1536,1920,2304]:
   r=read(D/f's{seed}/evaluation_{step}.csv')
   for arm in ARMS:
    ss=[x for x in r if x['arm']==arm and x['policy']=='Set'];worst=min(ss,key=lambda x:float(x['executed_gain']));state93=next(x for x in ss if int(x['state'])==93)
    for which,x in [('worst',worst),('state93',state93)]:tail.append(dict(seed=seed,step=step,arm=arm,which=which,**{k:x[k] for k in ['state','event','layer','initial','executed_gain','action']}))
    if arm.endswith('physical'):trend.append(next(x for x in rows if int(x['seed'])==seed and int(x['step'])==step and x['arm']==arm and x['policy']=='Set' and x['dimension']=='overall'))
 table(D/'tail_states.csv',tail);table(D/'physical_trajectory.csv',trend)
 report=(D/'REPORT.md').read_text();report+='\n## 最终结果解读与停止范围\n\n'
 report+='|seed|分支|恢复率|H_loss|局部MSE|β MAE|\n|---|---|---:|---:|---:|---:|\n'
 for seed in [17,29]:
  for arm in ['L_joint_sparse','L_physical','L_set','P_physical','P_set','P_contrast']:
   r=next(x for x in rows if int(x['seed'])==seed and int(x['step'])==2304 and x['arm']==arm and x['policy']=='Set' and x['dimension']=='overall');report+='|%d|%s|%.2f%%|%.6f|%.6g|%.6g|\n'%(seed,arm,float(r['recovery_pct']),float(r['H_loss']),float(r['local_mse']),float(r['beta_mae']))
 report+='\n物理目标分支的全部7个检查点见physical_trajectory.csv，可直接核验误差和收益是否同向；不能根据其中较好的检查点选择主结果。state93和每组实际最差状态均在tail_states.csv，包括原始初始误差与绝对损害，未移除尾部。\n\n独立审计核对最终矩阵重放、原E43实际动作收益重放、两种正质量评分公式、实际最差/固定状态的直接Q64目标及count/own。物理身份审计通过，活动组步数均96，无开发或Test读取。审计为数值和来源核验，不证明模型收敛。\n\n本轮不自动追加学习率、clip、对比权重或训练时长搜索；仍需根据损失分支、信息误差和实际损害的联合结果决定下一轮。Teacher四格明确显示Layer29教师在这批事件上更受Query覆盖限制，但预测模型上的改进尚需独立验证。\n'
 (D/'REPORT.md').write_text(report);dest=ROOT/'docs/research/evidence/e44_fit_diagnostics_v1'
 for n in ['REPORT.md','tail_states.csv','physical_trajectory.csv']:shutil.copy2(D/n,dest/n)
 for seed in [17,29]:
  for n in ['audit.json','independent_checks.csv']:shutil.copy2(D/f's{seed}'/n,dest/f's{seed}_{n}')
 manifestp=ROOT/'docs/research/evidence/manifest.json';manifest=json.loads(manifestp.read_text());manifest=[x for x in manifest if 'e44_fit_diagnostics_v1/' not in x['snapshot']]
 for p in dest.iterdir():
  if p.is_file():manifest.append(dict(source=str((D/p.name).relative_to(ROOT)) if not p.name.startswith('s') else 'artifacts/e44_fit_diagnostics_v1/'+p.name,snapshot=str(p.relative_to(ROOT)),sha256=sha(p)))
 # Correct per-seed source paths.
 for x in manifest:
  if 'e44_fit_diagnostics_v1/' in x['snapshot']:
   name=Path(x['snapshot']).name
   for seed in [17,29]:
    if name.startswith(f's{seed}_'):x['source']=f'artifacts/e44_fit_diagnostics_v1/s{seed}/'+name[len(f's{seed}_'):]
 dump(manifestp,manifest)
 with tarfile.open(D/'e44_review.tar.gz','w:gz') as tf:
  for p in D.rglob('*'):
   if p.is_file() and p.suffix in ['.csv','.json','.npz','.md']:tf.add(p,arcname=str(p.relative_to(D)))
 print('FINISHED',flush=True)
if __name__=='__main__':finish()
