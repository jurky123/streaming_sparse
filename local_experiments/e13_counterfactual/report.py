import csv,tarfile,shutil,platform
import numpy as np
import torch,scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import *
from .train import load,metrics
from .evaluate import NAMES

def table(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 audit=json.loads((OUT/'audit.json').read_text());assert audit['passed'];prep=json.loads((OUT/'preparation.json').read_text());lock=json.loads((OUT/'evaluation_lock.json').read_text());mech=json.loads((OUT/'mechanism_summary.json').read_text());cond=json.loads((OUT/'conditional_rank_summary.json').read_text());evrows=json.loads((OUT/'evaluation_per_state.json').read_text());summary=[];initial=[]
 for sp in ('train','validation','evaluation'):
  ids,F,g,j,meta,counts=load(sp);risk=np.array([m['state']==0 for m in meta])
  for name in NAMES:
   scores=np.load(OUT/'models'/name/(sp+'_scores.npz'))['scores'];chosen=scores.argmax(1);gain=g[np.arange(len(g)),chosen];after=j-gain
   for panel,mask in [('initial_risk',risk),('counterfactual_rows',np.ones(len(meta),bool))]:
    s=metrics(scores[mask],g[mask],j[mask]);s.update(mean_nmse_before=float(j[mask].mean()),mean_nmse_after=float(after[mask].mean()),mean_absolute_nmse_reduction=float(gain[mask].mean()),ratio_of_mean_nmse_reduction_pct=float(100*gain[mask].sum()/j[mask].sum()));summary.append(dict(split=sp,model=name,panel=panel,**s))
 table(OUT/'ranking_summary.csv',summary);dump(OUT/'ranking_summary.json',summary)
 for r in evrows:
  if r['state']==0:initial.append(r)
 table(OUT/'initial_risk_per_instance.csv',initial);table(OUT/'counterfactual_per_state.csv',evrows)
 mechanism=json.loads((OUT/'mechanism_per_event.json').read_text());table(OUT/'mechanism_per_event.csv',[{k:v for k,v in r.items() if k!='state_pairs'} for r in mechanism]);table(OUT/'ranking_flip_state_pairs.csv',[dict(event_id=r['event_id'],split=r['split'],**p) for r in mechanism for p in r['state_pairs']])
 conditional=json.loads((OUT/'conditional_rank_metrics.json').read_text());table(OUT/'conditional_rank_metrics.csv',[m for r in conditional for m in r['metrics']]);table(OUT/'actual_pool_repeat_coverage.csv',[{k:v for k,v in r.items() if k!='metrics'} for r in conditional])
 examples=[]
 labelsdir=OUT/'paired_labels';labelsdir.mkdir(exist_ok=True)
 for r in mechanism:
  eid=r['event_id'];a=np.load(OUT/'events'/eid/'anchors.npz');s=np.load(OUT/'events'/eid/'states.npz');G=a['gains'];order=np.argsort(-np.ptp(G,axis=0))[:3]
  for k in order:
   for st in range(len(G)):examples.append(dict(event_id=eid,split=split(eid),remove=int(a['remove'][k]),add=int(a['add'][k]),state=str(s['names'][st]),true_gain=float(G[st,k]),initial_risk_gain=float(G[0,k]),state_nmse=float(s['nmse'][st])))
  np.savez_compressed(labelsdir/f'{eid}.npz',gains=G,remove=a['remove'],add=a['add'],state_names=s['names'],state_masks=s['masks'],own=s['own'],K=s['K'])
 table(OUT/'counterfactual_pair_examples.csv',examples)
 def get(name,sp='evaluation'):return next(r for r in summary if r['model']==name and r['split']==sp and r['panel']=='initial_risk')
 e=mech['evaluation'];worst=audit['worst_case_high_precision'];lines=['E13｜反事实集合监督与排序/执行解耦','',
 '结论：真实交换收益的集合条件依赖得到直接验证；但加入反事实状态后，当前Set-Directional模型仍未可靠优于Pair-only。开发集排序未达到预登记门槛，本轮没有训练独立gating，没有把强制交换结果当作可部署Router。','',
 '范围与公平对照',
 '继续使用178f95a8/summary_pretrained/G-KVR的冻结set_oracle_v1；Q64/Page64、R4/rank8、10%精确预算、own保护；早期P216/K21，稳态P410/K41。输入M16、评价Q64、Medium池16×32。没有新视频/QKV、全局求解或kernel优化。GitHub旧包是精简交付；本轮实际读取本地完整冻结贡献。',
 f"144个head–tile构造{prep['states']}个去重集合：训练{prep['by_split']['train']}、开发{prep['by_split']['validation']}、评价{prep['by_split']['evaluation']}。训练仍只有city_walk/seed101的36个head–tile（一条轨迹、18组双tile事件），不是249条独立轨迹。开发city_walk/seed303；评价horse_field两seed72例，均为已参与开发的场景。",
 '状态为Risk、保存的Risk＋Exchange真实目标局部搜索第1/2/4步集合（oracle_step标签不表示全局OPT）、无标签参与的1/2/4次均匀合法随机扰动；终止后的重复Oracle状态去重。每状态保留全部512个候选收益。同一QKV下各状态的原始KV、摘要、位置、预算均相同，仅精确页集合变化。',
 '四组：A0_Risk、A2_Risk、A0_CF、A2_CF。A0使用页面对20槽；A2使用32槽含标量/方向上下文，沿用E12定义。全部32→32 GELU→1、1089参数、seed17同初始化、2000次Adam更新。每次更新每个训练head–tile抽一个状态，batch恒为36×512；两种CF架构使用相同抽样序列，Risk对照总看初始状态。归一化按训练head–tile等权、内部状态等权拟合。',
 '损失仅在512交换间做softmax教师排序，收益按状态标准差缩放、温度0.3；不加入no-op，也不根据绝对logit决定是否执行。强制选top1只用于排序诊断；真实标签拒绝负收益的结果只作为事后上限，不能部署。','',
 'E13-A：同一候选的集合条件依赖',
 '共同候选面板取所有状态共同选中且非own的移出页、共同未选中的加入页；按Risk确定最多16×32个固定合法页面对。该面板用于反事实机制，不保证这些页面对在每状态的Medium池内；实际训练仍使用各状态自己的Medium池。',
 f"评价72例的每事件等权平均：固定候选中{100*e['anchor_sign_flip_fraction_mean']:.2f}%跨集合出现正负收益翻转；相对Risk集合，{100*e['mean_rank_flip_vs_risk_mean']:.2f}%非平局动作对的相对排序翻转。",
 f"完整矩形g[集合,固定页面对]的双因素方差分解：页面对主效应{100*e['pair_main_fraction_mean']:.2f}%，集合整体主效应{100*e['set_main_fraction_mean']:.2f}%，页面对×集合交互{100*e['interaction_fraction_mean']:.2f}%。合计状态相关部分{100*e['state_dependent_fraction_mean']:.2f}%。这不是预测器R²；集合整体平移不改变排序，交互项才改变相对偏好。",
 f"实际训练Medium池中，{100*cond['train']['repeated_row_fraction']:.2f}%候选行对应在同一head–tile多个状态出现的页面对；这些重复页面对中{100*cond['train']['repeated_pair_sign_flip_fraction']:.2f}%存在正负翻转。因此训练数据确实包含了重复候选的条件监督，不只是另存一个未参与训练的反事实面板。",'',
 'E13-B：初始Risk状态下的排序（不使用gating）',
 '开发36例：模型 | 强制top1平均改善% | 事后只保留正收益% | 同池Oracle恢复%']
 for name in NAMES:
  r=get(name,'validation');lines.append(f"{name} | {r['mean_gain_pct']:.3f} | {r['positive_filtered_mean_pct']:.3f} | {100*r['positive_filtered_recovery']:.2f}")
 lines+=['开发门槛在训练前固定：CF模型的正收益事后过滤诊断至少恢复同池Oracle的30%，且比同架构Risk-only对照至少提高1个百分点；只用开发集判断是否值得继续训练独立接受模型。A0_CF为14.03%恢复、+0.179个百分点；A2_CF为14.60%、+0.510个百分点，均未通过。',
 '故没有训练gating，也没有进行阈值搜索。四个排序模型在评价前固定哈希；评价结果不能反过来改变此决定。','',
 '评价72例：模型 | 强制top1平均改善% | 中位改善% | P10% | 退化例数 | 事后正收益过滤% | 同池Oracle恢复%']
 for name in NAMES:
  r=get(name);lines.append(f"{name} | {r['mean_gain_pct']:.3f} | {r['median_gain_pct']:.3f} | {r['p10_gain_pct']:.3f} | {r['degraded']}/72 | {r['positive_filtered_mean_pct']:.3f} | {100*r['positive_filtered_recovery']:.2f}")
 lines += ['同Medium池一次交换Oracle平均改善12.631%；它是可行池内最佳动作，不是全P集合全局最优。Oracle恢复率此处定义为“平均相对正收益/平均相对Oracle收益”，与E12逐实例恢复率均值不同，不能混用。','',
 '极端相对误差的解释（保留在主表中）',
 f"A0_Risk最差实例{worst['event_id']}：NMSE从{worst['baseline_nmse']:.12g}变为{worst['after_nmse']:.12g}，相对改善{worst['relative_gain_pct']:.3f}%（即相对退化约20983%）。基准误差非常小，导致相对百分比极大；该结果已用np.longdouble扩展精度独立重算，不是分页错位或单位错误。",
 '没有截断、删除该实例或改换主要指标。ranking_summary.csv同时提供中位数、P10、最差值、平均绝对NMSE变化和均值NMSE之比，避免只凭极端均值解释模型。强制动作不是部署策略，也没有把其负收益伪装成no-op。','',
 '是否学到了条件排序？',
 f"共同候选面板上，A2_CF预测的排名变化与真实排名变化相关性：训练均值{cond['train']['models']['A2_CF']['mean_conditional_rank_change_correlation']:.3f}、开发{cond['validation']['models']['A2_CF']['mean_conditional_rank_change_correlation']:.3f}、评价{cond['evaluation']['models']['A2_CF']['mean_conditional_rank_change_correlation']:.3f}，仍然很弱。A0在固定页面对上预测排名不随集合变，相关性无定义而非虚报为有效0。",
 'A2_CF的全锚点排名Spearman略高于A0_CF（评价约0.322 vs0.303），但没有转化为更高的top1正收益上限（3.193% vs3.549%）。A2_Risk评价强制动作均值为正，仍有29/72退化，而且开发排序门槛未过；不以此补选gating或宣称成功。',
 '扩充状态使A0的评价排序潜力有所提升，但未出现Set-Directional相对Pair-only一致优势。它说明“补充状态”本身不够；不能进一步推断摘要信息论上不足。当前32槽压缩、排序损失、有限独立场景和训练分布都仍可能限制结果。CF均匀采样也把初始Risk的直接训练暴露分散给其他状态，需明确这项权衡。','',
 '验收与交付',
 f"全量复算{audit['medium_labels']:,}个Medium动作和{audit['anchor_labels']:,}个共同候选标签；每状态8个固定动作及全部{audit['evaluation_masks']}个评价mask另按完整选中集合直接求和校验，最大差异{audit['max_error']:.3e}。own/预算、同初始化/容量/更新数、事件等权训练、输入白名单投毒和仅训练集归一化检查通过。734个原冻结文件哈希未变。",
 '同一候选完整g(S,r,a)与集合身份保存在events/<event_id>/anchors.npz；所有训练池标签与特征在states.npz。另提供paired_labels仅标签包，去掉大特征矩阵，便于复核机制；示例表按每例收益变化最大3个页面对展示，仅为说明，汇总统计使用全部共同候选。',
 '已完成反事实构造、四组排序对照、共同候选条件排名诊断及全部审计。按开发门槛未启动独立接受模型；没有多轮在线交换、新数据采集、网络规模搜索、全局求解或kernel优化。原模型容量相同，未重复做硬件成本测量；本轮主要增加的是离线状态和标签。','',
 '建议结论：集合条件依赖确实存在；当前摘要派生的少量特征加小MLP尚不能可靠利用它。下一步应评估可观测摘要对交换排序的信息充分性，而不是仅继续提高no-op阈值或增加模型规模。']
 (OUT/'REPORT.zh.txt').write_text('\n'.join(lines)+'\n')
 fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained');vals=[e['pair_main_fraction_mean'],e['set_main_fraction_mean'],e['interaction_fraction_mean']];ax[0,0].bar(['Pair main','Set main','Interaction'],100*np.array(vals));ax[0,0].set(title='Fixed-pair gain variance decomposition',ylabel='Per-event mean fraction (%)')
 names=['Pair Risk','Set Risk','Pair CF','Set CF'];ax[0,1].bar(names,[get(n)['positive_filtered_mean_pct'] for n in NAMES]);ax[0,1].axhline(12.6306478463,ls='--',color='gray',label='Medium oracle');ax[0,1].set(title='Ranking potential: oracle rejection diagnostic',ylabel='Mean relative gain (%)');ax[0,1].legend()
 ax[1,0].bar(names,[get(n)['mean_gain_pct'] for n in NAMES]);ax[1,0].set_yscale('symlog',linthresh=1);ax[1,0].axhline(0,color='black',lw=.5);ax[1,0].set(title='Forced top1: no gating, includes all outliers',ylabel='Mean relative gain (%), symlog')
 ax[1,1].bar(names,[cond['evaluation']['models'][n]['mean_anchor_rank_spearman'] for n in NAMES]);ax[1,1].set(title='Common-anchor ranking correlation',ylabel='Mean Spearman correlation')
 fig.suptitle('E13: horse_field72; fixed QKV, M16 inputs / Q64 objective; not unseen-scene evidence');fig.savefig(OUT/'overview.png',dpi=170);plt.close(fig)
 code=OUT/'code'/'local_experiments';source=Path(__file__).parent
 for package,files in [('e13_counterfactual',[p.name for p in source.glob('*.py')]),('e12_swap_prediction',['__init__.py','core.py','train.py']),('set_oracle_v1',['__init__.py','common.py'])]:
  dest=code/package;dest.mkdir(parents=True,exist_ok=True)
  for name in files:
   p=source.parent/package/name
   if p.resolve()!=(dest/name).resolve():shutil.copy2(p,dest/name)
 dump(OUT/'provenance.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,source={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')},data_manifest_sha256=sha(DATA/'manifest.json'),weights=lock['models']))
 (OUT/'REPRODUCE.txt').write_text('Requires full frozen set_oracle_v1, not the prior Git review archive. Set SET_ORACLE_ROOT and E13_ROOT; use a new E13_ROOT for changed methods. Put code/ on PYTHONPATH or run in repository. Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8. Stages: python -m local_experiments.e13_counterfactual.prepare; train --arch A0/A2 --regime Risk/CF --gpu 0/1/2/3; evaluate (freezes development gate before test; stops if optional OOF gating is required); mechanism; diagnostics; audit; report. One-command runner: python -m local_experiments.e13_counterfactual.run. No no-op is used in ranking; positive-only rejection uses teacher labels and is never deployable. Actual gate failed, so no accepting classifier was fitted. Complete features/labels remain in events; label-only archive contains full fixed-pair gains and masks.\n')
 dump(OUT/'completion.json',dict(status='complete',scope='counterfactual supervision plus ranking factorial',events=144,states=prep['states'],models=4,gating='not_run_ranking_gate_failed',online_multiround='not_run',data_unchanged=True))
 with tarfile.open(OUT.parent/'e13_counterfactual_paired_labels.tar.gz','w:gz') as tar:tar.add(labelsdir,arcname='e13_counterfactual_paired_labels');tar.add(OUT/'protocol.json',arcname='protocol.json')
 pack=OUT.parent/'e13_counterfactual_review.tar.gz'
 with tarfile.open(pack,'w:gz') as tar:
  for p in OUT.iterdir():
   if p.is_file() and p.suffix in ('.txt','.json','.csv','.png'):tar.add(p,arcname='e13_counterfactual_review/'+p.name)
  tar.add(OUT/'code',arcname='e13_counterfactual_review/code')
  for name in NAMES:
   for f in ['weights.npz','training.json','development.json']:tar.add(OUT/'models'/name/f,arcname=f'e13_counterfactual_review/models/{name}/{f}')
 print('report',OUT/'REPORT.zh.txt','review_bytes',pack.stat().st_size,'labels_bytes',(OUT.parent/'e13_counterfactual_paired_labels.tar.gz').stat().st_size)
if __name__=='__main__':main()
