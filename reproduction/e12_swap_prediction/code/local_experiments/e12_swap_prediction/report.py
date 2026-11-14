import json,csv,tarfile,shutil,platform
import numpy as np
import torch,scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import *
from .train import load_states,decision_metrics
from .evaluate import model_scores

def table(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 torch.set_num_threads(1);audit=json.loads((OUT/'audit.json').read_text());assert audit['passed'];lock=json.loads((OUT/'evaluation_lock.json').read_text());ev=json.loads((OUT/'evaluation_summary.json').read_text());evrows=json.loads((OUT/'evaluation_per_instance.json').read_text());cost=json.loads((OUT/'reference_cost.json').read_text());summaries=[];action_stats=[]
 for model in lock['models']:
  name=model['name'];mroot=OUT/'models'/name;selected=json.loads((mroot/'selection.json').read_text())['selected'];threshold=float('inf') if selected['threshold'] is None else selected['threshold']
  for sp in ('train','validation','evaluation'):
   ids,F,g,j=load_states(sp)
   scores=np.load(mroot/'evaluation_scores.npz')['scores'] if sp=='evaluation' else model_scores(name,F)
   for label,t in [('frozen',threshold),('threshold0',0)]:
    stats,dec=decision_metrics(scores,g,j,t);gainpct=100*dec['gain']/j;bad=gainpct< -1e-9
    row=dict(model=name,split=sp,calibration=label,threshold=None if np.isinf(t) else t,**stats,worst_gain_pct=float(gainpct.min()),mean_degraded_gain_pct=float(gainpct[bad].mean()) if bad.any() else None)
    summaries.append(row)
    if sp=='evaluation':
     for i,eid in enumerate(ids):
      positive=g[i]>1e-12;pred=scores[i]>t;tp=int((positive&pred).sum());fp=int((~positive&pred).sum());action_stats.append(dict(event_id=eid,model=name,calibration=label,true_positive=tp,false_positive=fp,positive_actions=int(positive.sum()),negative_actions=int((~positive).sum()),predicted_positive_actions=int(pred.sum())))
 table(OUT/'all_split_metrics.csv',summaries);dump(OUT/'all_split_metrics.json',summaries);table(OUT/'positive_action_metrics.csv',action_stats)
 action_summary=[]
 for name in sorted({r['model'] for r in action_stats}):
  for label in ('frozen','threshold0'):
   rr=[r for r in action_stats if r['model']==name and r['calibration']==label];tp=sum(r['true_positive'] for r in rr);fp=sum(r['false_positive'] for r in rr);pos=sum(r['positive_actions'] for r in rr);neg=sum(r['negative_actions'] for r in rr)
   action_summary.append(dict(model=name,calibration=label,micro_recall=tp/pos if pos else None,micro_precision=tp/(tp+fp) if tp+fp else None,micro_false_positive_rate=fp/neg if neg else None,states_without_positive=sum(r['positive_actions']==0 for r in rr)))
 table(OUT/'positive_action_summary.csv',action_summary)
 # Paired model differences, descriptive only. No new threshold/model fitting.
 paired=[]
 for sp in ('validation','evaluation'):
  ids,F,g,j=load_states(sp);gains={}
  for arch in ('A0','A1','A2'):
   name=arch+'_s17';sel=json.loads((OUT/'models'/name/'selection.json').read_text())['selected'];t=float('inf') if sel['threshold'] is None else sel['threshold'];sc=model_scores(name,F);_,d=decision_metrics(sc,g,j,t);gains[arch]=d['gain']/j
  for a,b in [('A1','A0'),('A2','A0'),('A2','A1')]:
   delta=100*(gains[a]-gains[b]);paired.append(dict(split=sp,comparison=a+' minus '+b,mean_gain_difference_pp=float(delta.mean()),median_gain_difference_pp=float(np.median(delta)),wins=int((delta>1e-9).sum()),losses=int((delta<-1e-9).sum()),ties=int((abs(delta)<=1e-9).sum())))
 table(OUT/'set_feature_ablation.csv',paired)
 failed=[dict(r) for r in evrows if r['gain']< -1e-12];table(OUT/'degraded_actions.csv',failed)
 def get(name,sp='evaluation',cal='frozen'):return next(r for r in summaries if r['model']==name and r['split']==sp and r['calibration']==cal)
 lines=['E12 — Set-Conditional Swap Prediction','',
 '结论：未得到稳定有效的单步交换预测器；未证明显式集合信息有可靠额外收益。A0第一种子仅靠极少数交换取得很小正收益，第二种子未通过开发门槛；A1迁移后退化；A2校准后全部no-op。因此按计划停止多轮、扩展状态训练和融合kernel开发。','',
 '1. 固定范围',
 '复用本地完整set_oracle_v1；GitHub此前发布的是精简评估包，不含数GB完整残差。本轮没有采集新QKV、生成视频或扩展全局求解。178f95a8/summary_pretrained/G-KVR，Q64/Page64、R4/rank8、own保护、固定10%。chunk2为P216/K21，其余P410/K41。Risk初始mask，Medium=16×32=512个交换，每状态另有no-op。',
 'city_walk/seed101的36个完整状态训练；city_walk/seed303的36个状态选阈值；horse_field两seed共72例评价。全部误差是64Query、pre-W_O真实混合NMSE；输入固定16个实际probes。场景已有开发使用史，不能称全新场景泛化。','',
 '2. 模型、特征与监督',
 'A0/A1/A2均32→32 GELU→1，共1089参数，第一轮seed17相同初始化、相同2000次Adam更新。完整36状态full-batch训练；不把18432个候选行当成独立状态。固定lr1e-3、grad clip1、temperature0.3；未做温度或网络规模搜索。',
 '32槽特征：前20槽为移出/加入页面各10个在线统计（相对Risk、Mass、四项风险、排名、年龄、历史标记、摘要Value相对全摘要输出的距离）。4槽集合标量为质量覆盖、质量加权风险、风险方差和历史比例。8槽方向信息为当前head rank8 basis下H_S的模、候选与H_S的内积/cosine、候选方向关系和交换后的H模变化。H_S不是预测残差。A0只启用前20槽，A1前24槽，A2全部32槽；未用槽在训练集标准化后置零。',
 '教师收益直接读取既有初始Risk完整交换面，并以FP64残差重新核验。训练标签按每状态512收益的标准差归一化，只用于softmax教师分布；512个交换加no-op一起做分布交叉熵。no-op真实收益和logit固定0。标签归一化尺度从不作为推理输入。',
 '只在开发集从预登记阈值0/.1/.25/.5/1/2/4/8/16/infinity中选择：先限制退化比例≤15%，再最大化平均相对改善；若无正收益方案，保留no-op。非零阈值表示额外的决策置信门槛，分数不是已校准的NMSE收益单位。评价前模型、阈值、SHA256统一写入evaluation_lock.json。','',
 '3. 开发集与门槛（36例）',
 '模型 | 阈值 | 平均改善% | 交换数 | 退化数 | 门槛']
 for name in ['A0_s17','A1_s17','A2_s17','A0_s29']:
  r=get(name,'validation');lines.append(f"{name} | {r['threshold']} | {r['mean_gain_pct']:.4f} | {r['swapped']}/36 | {r['degraded']}/36 | {r['gate']}")
 lines+=['A0第一种子虽形式上通过，但只交换1个状态；不能称稳定覆盖。按预登记规则，仅对开发集入选A0补seed29，第二种子选择全no-op。两个种子未同时过门槛，故不启动多轮和第二阶段状态扩充。','',
 '4. 冻结阈值后的72例评价',
 '模型 | 平均改善% | P10% | 最差改善% | 改善/退化/no-op | 平均池Oracle恢复%']
 for name in ['A0_s17','A1_s17','A2_s17','A0_s29']:
  r=get(name);lines.append(f"{name} | {r['mean_gain_pct']:.4f} | {r['p10_gain_pct']:.4f} | {r['worst_gain_pct']:.4f} | {r['improved']}/{r['degraded']}/{r['no_op']} | {100*r['mean_pool_recovery']:.4f}")
 oracle=next(r for r in ev if r['name']=='Medium_oracle');lines += [f"同Medium池一次真实Q64交换oracle平均改善{oracle['mean_gain_pct']:.4f}%，是离线可行决策对照，不是全局最优。Risk不交换为0。收益恢复按每状态gain/pool-best再平均，仅含pool-best>1e-12状态；可能与平均相对NMSE改善符号不同，不能用均值之比替换。另保存pooled_gain_recovery和中位数。",'P10为0不代表没有重尾风险：若退化不足10%，P10会被no-op覆盖。报告同时保留最差值、退化动作清单和退化条件均值。','',
 '5. 固定threshold0的未校准诊断（不用于改选评价模型）',
 '模型 | 训练平均改善% | 开发平均改善% | 评价平均改善% | 评价退化数/72']
 for name in ['A0_s17','A1_s17','A2_s17','A0_s29']:
  tr=get(name,'train','threshold0');va=get(name,'validation','threshold0');te=get(name,'evaluation','threshold0');lines.append(f"{name} | {tr['mean_gain_pct']:.3f} | {va['mean_gain_pct']:.3f} | {te['mean_gain_pct']:.3f} | {te['degraded']}/72")
 lines+=['三组可以在训练状态上获得收益，但没有稳定迁移。A2在评价集threshold0的均值略正，仍有较高退化率，且其开发集未过门槛；不能事后换用该阈值来宣称成功。','',
 '6. 集合信息的独立价值',
 '开发校准后的A1/A2均未显示可靠优于A0的收益。训练集上方向信息有更好的拟合，不代表迁移收益。这里只对每实例初始Risk集合训练，不能据此否定其他状态、其他损失或更充分数据下的集合信息价值。',
 '同一候选(r,a)，改变其他集合成员时：144/144实例A0输入保持不变，A1标量和A2方向均改变；加减更新与重新求和一致。因此不是“集合特征根本没接入”的实现问题。另一方面，当前结果也不足以证明摘要信息论上不充分；36个训练状态、输入压缩和当前损失均可能限制泛化。','',
 '7. 成本与验收',
 f"MLP主乘加每tile={cost['score_flops_per_tile']:,} FLOPs，每层72×12 tile/head约{cost['score_flops_per_layer']/1e9:.3f} GFLOPs，排除GELU、特征生成、候选选择、访存与启动。未开发融合kernel。",
 '参考实测：CPU完整参考流水线约13.7–14.4ms/head–tile，A100纯MLP约0.109–0.114ms；不是在线集成性能。当前参考复用缓存Risk特征，但为Value上下文重算摘要权重，尚未共享Risk内部权重或融合特征，不能据此声称满足在线预算。真实chunk5、layer1/15/29，预热3次、测量10次。CPU单线程总成本包含完整32槽特征、候选及网络；A100另测预驻留特征的纯MLP。两种测量不可相加冒充GPU端到端成本，也不能与上一轮GPU总router时间直接比较。完整采样见reference_cost.json。',
 f"73728个动作标签独立复核最大绝对差{audit['max_label_error']:.3e}；576个评价mask的own、预算、FP64目标通过。教师字段投毒测试通过、归一化仅训练集拟合、三组参数/更新/初始权重相同。734个冻结数据文件SHA256不变。",'',
 '已完成：E12-A/B/C单步对照、完整动作识别统计、no-op校准、第二种子条件性检查、错误动作清单、参考成本和审计。',
 '按门槛未启动：2/4轮自身轨迹、Oracle后续状态/扰动状态扩充训练、M64信息诊断、融合kernel、下游视频。没有训练新的逐页残差预测器，没有扩展全局证书。','',
 '动作识别统计另给positive_action_summary.csv的micro召回/精确率/FPR；没有预测正动作时precision为空，不把no-op当成正确识别了全部负动作。状态宏平均与动作微平均分列。',
 '文件入口：all_split_metrics.csv；evaluation_per_instance.csv/json；set_feature_ablation.csv；positive_action_metrics.csv；degraded_actions.csv；models/*/weights.npz、selection.json、training.json；protocol.json；evaluation_lock.json；audit.json；reference_cost.json。精简包不包含原始多GB数据，但保留模型和可复现代码。']
 (OUT/'REPORT.zh.txt').write_text('\n'.join(lines)+'\n')
 # Audit-approved plots keep raw diagnostic separate from frozen policy.
 fig,ax=plt.subplots(1,3,figsize=(16,4.6),layout='constrained');names=['A0_s17','A1_s17','A2_s17'];xx=np.arange(3)
 for i,sp in enumerate(['train','validation','evaluation']):ax[0].bar(xx+(i-1)*.25,[get(n,sp,'threshold0')['mean_gain_pct'] for n in names],width=.25,label=sp)
 ax[0].set_xticks(xx,['A0','A1','A2']);ax[0].set(title='Diagnostic: threshold 0',ylabel='Mean relative NMSE reduction (%)');ax[0].axhline(0,color='black',lw=.6);ax[0].legend()
 ax[1].bar(['A0','A1','A2','A0 seed29'],[get(n)['mean_gain_pct'] for n in names+['A0_s29']]);ax[1].axhline(0,color='black',lw=.6);ax[1].set(title='Frozen validation-selected policy',ylabel='Evaluation mean reduction (%)')
 low=[get(n)['improved'] for n in names+['A0_s29']];bad=[get(n)['degraded'] for n in names+['A0_s29']];noop=[get(n)['no_op'] for n in names+['A0_s29']];labels=['A0','A1','A2','A0 seed29'];ax[2].bar(labels,low,label='improved');ax[2].bar(labels,bad,bottom=low,label='degraded');ax[2].bar(labels,noop,bottom=np.array(low)+bad,label='no-op');ax[2].set(title='Evaluation decisions',ylabel='Instances / 72');ax[2].legend()
 fig.suptitle('E12: same1089 parameters; M16, Medium pool, one swap; evaluation is horse_field72');fig.savefig(OUT/'overview.png',dpi=170);plt.close(fig)
 # Complete source snapshot and reproducible review archive; no automatic Git push.
 code=OUT/'code'/'local_experiments'/'e12_swap_prediction';code.mkdir(parents=True,exist_ok=True);source=Path(__file__).parent
 for p in source.glob('*.py'):
  if p.resolve()!=(code/p.name).resolve():shutil.copy2(p,code/p.name)
 base=OUT/'code'/'local_experiments'/'set_oracle_v1';base.mkdir(parents=True,exist_ok=True)
 for name in ['__init__.py','common.py']:
  p=source.parent/'set_oracle_v1'/name
  if p.resolve()!=(base/name).resolve():shutil.copy2(p,base/name)
 dump(OUT/'provenance.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,sources={p.name:sha(p) for p in code.glob('*.py')},frozen_manifest_sha256=sha(DATA/'manifest.json'),model_hashes={m['name']:m['weights_sha256'] for m in lock['models']}))
 (OUT/'REPRODUCE.txt').write_text('Use the complete frozen set_oracle_v1 dataset, not only its previous review archive. Set SET_ORACLE_ROOT and E12_ROOT to absolute paths; use a NEW E12_ROOT for changed algorithms. Put code/ on PYTHONPATH or run from repository root. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8. One-command runner: python -m local_experiments.e12_swap_prediction.run --gpus 0,1,2. Individual stages: python -m local_experiments.e12_swap_prediction.prepare; train --arch A0/A1/A2 --gpu 0/1/2; inspect development gate before seed29. The supplied evaluation_lock records the actual frozen decisions and stopping rule. evaluate runs inference only after lock; audit validates labels/data; benchmark records reference costs; report produces outputs. No original contributions included. Threshold None means unconditional no-op. Current follow-up seed is A0 seed29; multi-round gate failed.\n')
 dump(OUT/'completion.json',dict(status='complete',scope='E12 single-step A/B/C plus gated seed29',primary_models=3,secondary_models=1,states=144,evaluation=72,multi_round='not run: two-seed validation gate failed',raw_data_unchanged=True))
 pack=OUT.parent/'e12_swap_prediction_review.tar.gz'
 with tarfile.open(pack,'w:gz') as tar:
  for p in sorted(OUT.iterdir()):
   if p.is_file() and p.suffix in ('.txt','.json','.csv','.png'):tar.add(p,arcname='e12_swap_prediction_review/'+p.name)
  tar.add(OUT/'models',arcname='e12_swap_prediction_review/models');tar.add(OUT/'code',arcname='e12_swap_prediction_review/code')
 print('REPORT',OUT/'REPORT.zh.txt','review_bytes',pack.stat().st_size)
if __name__=='__main__':main()
