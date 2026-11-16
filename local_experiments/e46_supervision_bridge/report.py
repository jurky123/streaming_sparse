from .core import *
from ..e41_training_signal.report import decision_summary
import shutil,tarfile

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run():
 rows=[];infos=[];small=[]
 for seed in [17,29]:
  dest=D/f's{seed}';assert (dest/'aligned_complete.json').exists() and (dest/'layer_complete.json').exists()
  for phase in ['frozen','trained']:
   assert (dest/f'{phase}_evaluation_complete.json').exists();rows+=read(dest/f'{phase}_decisions.csv');infos+=read(dest/f'{phase}_information.csv')
  chosen=set(json.loads((E43/f's{seed}/protocol.json').read_text())['events']);metas={m['event']:m for m in json.loads((CACHE/f'train_s{seed}/metadata.json').read_text())+json.loads((CACHE/f'development_s{seed}/metadata.json').read_text())};heads={(m['trajectory'],m['layer']):m['head'] for e,m in metas.items() if e in chosen}
  for arm,source in [('E29',ROOT/f'artifacts/e41_training_signal_v1/s{seed}/decisions.csv'),('Shared_T0',ROOT/f'artifacts/e42_local_coverage_v1/T0_s{seed}/decisions.csv')]:
   for r in read(source):
    mm=metas[r['event']];split=mm['split'];cohort='dev_unseen_trajectory' if split=='development' else 'fitted48' if r['event'] in chosen else 'same_chunk_step_otherhead' if mm['chunk']==5 and mm['step']==0 else 'same_head_othercontext' if mm['head']==heads[(mm['trajectory'],mm['layer'])] else 'other_head_othercontext';r.update(arm=arm,cohort=cohort);rows.append(r)
  for step in [0,768,1536,2304]:
   for dimension in ['overall','layer']:
    rr=read(dest/f'aligned_eval_{step}.csv');groups=collections.defaultdict(list)
    for r in rr:groups[(r['policy'],'all' if dimension=='overall' else r['layer'])].append(r)
    for (policy,value),v in groups.items():
     ss=decision_summary(v);ss.update(seed=seed,step=step,arm='Aligned_M16',dimension=dimension,value=value,policy=policy)
     for k in ['local_mse','local_mass_mse','beta_mae']:ss[k]=float(np.mean([float(r[k]) for r in v]))
     small.append(ss)
  old=read(E43/f's{seed}/evaluation_1.csv')
  for dimension in ['overall','layer']:
   groups=collections.defaultdict(list)
   for r in old:
    if r['arm']=='P':groups[(r['policy'],'all' if dimension=='overall' else str(metas[r['event']]['layer']))].append(r)
   for (policy,value),v in groups.items():
    ss=decision_summary(v);ss.update(seed=seed,step=2304,arm='Original_Q64',dimension=dimension,value=value,policy=policy)
    for k in ['local_mse','beta_mae']:ss[k]=float(np.mean([float(r[k]) for r in v]))
    ss['local_mass_mse']=None;small.append(ss)
 summary=[]
 for dim in ['overall','cohort','layer','head','trajectory','unused_train']:
  gs=collections.defaultdict(list)
  for r in rows:
   if dim=='unused_train' and (r['split']!='train' or r['cohort']=='fitted48'):continue
   value='all' if dim in ['overall','unused_train'] else r[dim];gs[(r['seed'],r['split'],r['arm'],r['policy'],value)].append(r)
  for (seed,split,arm,policy,value),v in gs.items():summary.append(dict(seed=seed,split=split,arm=arm,policy=policy,dimension=dim,value=value,**decision_summary(v)))
 inf=[];gs=collections.defaultdict(list)
 for r in infos:gs[(r['seed'],r['split'],r['arm'],r['layer'],r['cohort'])].append(r)
 for (seed,split,arm,layer,cohort),v in gs.items():
  row=dict(seed=seed,split=split,arm=arm,layer=layer,cohort=cohort,events=len(v))
  for k in ['local_mse','local_mass_mse','beta_mae']:row[k]=float(np.mean([float(r[k]) for r in v]))
  inf.append(row)
 table(D/'bridge_summary.csv',summary);table(D/'small_alignment_summary.csv',small);table(D/'information_summary.csv',inf)
 lines=['# E46：监督Query对齐与小集到全量桥接','', 'A：原E43固定48事件/96状态，新增M16真值setloss监督2304步，与原E43 Q64监督P比较。local/quality与架构不变。M16真实收益以16Query均值除原Q64 dense均值能量，保持预测评分单位；setloss的gain scale仍来自原Q64标签，仅替换目标，避免尺度更换混杂。','', 'B：冻结E43 P、E44 P_set，完整回放1152训练/288开发Risk状态。小集以外1104事件是在原E29预训练中已见的训练轨迹，不能叫从未训练数据；它们只是未参加小集续训。开发集才是轨迹留出，且历史上曾用于选择原E29检查点，故不是新的无污染最终Test。','', 'C：三个按层Reader各从同一E29初始化，原E39两轮计划共2304次更新；每层768步。共享T0复用E39最终检查点。按层解耦增加存储参数至153471（3倍），每次活动参数仍51157；同时改变共享和Adam时钟，因此仅为机制组合对照，不是坐标失配因果证明。','', '统一固定C/φ/B16、Risk、M16、Medium、budget/own、原损失、Adam/clip和最终检查点，Test封存。没有做新的Query Selector：Q64摘要contrast不是原M16路径里已免费得到的量，须先定义可在线取得的指标和成本。','', '## A：监督对齐，小集Q64实际动作收益','', '|seed|setloss标签|Set恢复|H_loss|局部MSE|β MAE|','|---|---|---:|---:|---:|---:|']
 for r in small:
  if r['step']==2304 and r['dimension']=='overall' and r['policy']=='Set':lines.append('|%d|%s|%.2f%%|%.6f|%.6g|%.6g|'%(r['seed'],r['arm'],r['recovery_pct'],r['H_loss'],r['local_mse'],r['beta_mae']))
 lines+=['','## B/C：完整Risk评价','', '|seed|划分|模型|Set恢复|H_loss|退化|P10|','|---|---|---|---:|---:|---:|---:|']
 for r in summary:
  if r['dimension']=='overall' and r['policy']=='Set':lines.append('|%s|%s|%s|%.2f%%|%.6f|%d/%d|%.2f%%|'%(r['seed'],r['split'],r['arm'],r['recovery_pct'],r['H_loss'],r['harm'],r['events'],r['p10_relative_pct']))
 lines+=['','同信息Independent、逐轨迹/layer/head以及48拟合/1104其余训练分列见bridge_summary.csv。cohort进一步区分同chunk/step不同head、同head其他context、不同head其他context；不能把其中差异直接当成单因果泛化损失。Risk单状态与E43的96状态合并恢复分母不同，不能直接比较百分点。','', '## 可核验与待解释范围','', '逐事件目标、checkpoints、loss曲线、所有实际选择和信息误差保存。原B16是训练集拟合一次后冻结的共享输出基，不是每层独立输出基；Value detail的basis确实按layer/head不同，但这不自动证明Reader无法从摘要学习其分布，也不证明旋转是唯一信息缺口。','', 'M16目标对齐不增加输入Query，不能使未观测Q48自动可见；若回归改善而Q64决策变差，只支持任务目标差异，不证明Q64监督必须扭曲某个具体页面。小集Pset到开发评价未训练/选checkpoint。按层对照没有新增head embedding或按head网络，不追加架构搜索。','', '本轮所有结果是开发机制诊断，仍非部署结论。增量标签沿用既有数值路径；另核验稳定正质量评分和count/own，异常事件不删除。']
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n');out=ROOT/'docs/research/evidence/e46_supervision_bridge_v1';out.mkdir(parents=True,exist_ok=True)
 for n in ['REPORT.md','bridge_summary.csv','small_alignment_summary.csv','information_summary.csv']:shutil.copy2(D/n,out/n)
 for seed in [17,29]:
  for n in ['protocol.json','aligned_complete.json','layer_complete.json','frozen_checks.csv','trained_checks.csv','frozen_evaluation_complete.json','trained_evaluation_complete.json']:shutil.copy2(D/f's{seed}'/n,out/f's{seed}_{n}')
 with tarfile.open(D/'e46_review.tar.gz','w:gz') as tf:
  for p in D.rglob('*'):
   if p.is_file() and p.suffix in ['.csv','.json','.npz','.md']:tf.add(p,arcname=str(p.relative_to(D)))
 print('REPORT DONE',flush=True)
if __name__=='__main__':run()
