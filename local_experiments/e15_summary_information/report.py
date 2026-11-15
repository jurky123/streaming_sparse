from .run import *
def run():
 summary=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'audit.json').read_text());assert audit['passed']
 c=next(x for x in summary['concentration'] if x['split']=='evaluation');B={r['representation']:r for r in summary['B'] if r['split']=='evaluation'}
 acts=list(csv.DictReader((OUT/'B_actions.csv').open()));sr=[]
 for sp,rep,st in sorted({(r['split'],r['representation'],r['state']) for r in acts}):
  rr=[r for r in acts if (r['split'],r['representation'],r['state'])==(sp,rep,st)];g=np.array([float(r['true_gain']) for r in rr]);J=np.array([float(r['nmse']) for r in rr]);best=np.array([float(r['oracle_gain']) for r in rr]);gate=np.array([r['proxy_gate']=='True' for r in rr]);sr.append(dict(split=sp,representation=rep,state=st,events=len(rr),forced_relative_pct=float(100*np.mean(g/J)),forced_median_relative_pct=float(100*np.median(g/J)),positive_filtered_relative_pct=float(100*np.mean(np.maximum(g,0)/J)),positive_filtered_pooled_recovery=float(np.maximum(g,0).sum()/best.sum()) if best.sum()>0 else None,proxy_gate_relative_pct=float(100*np.mean(np.where(gate,g,0)/J))))
 table('B_by_state.csv',sr)
 state_rows=[r for r in summary['A_groups'] if r['split']=='evaluation' and r['group_by']=='state']
 state_table='\n'.join(f"| {r['group']} | {r['states']} | {100*r['strong_V_share']:.2f}% | {r['strong_relative_pp']:.3f} |" for r in state_rows)
 b_table='\n'.join(f"| {n} | {B[n]['set_full_cosine']:.4f} | {B[n]['candidate_full_cosine']:.4f} | {B[n]['gain_change_pearson']:.4f} | {B[n]['interaction_pearson']:.4f} |" for n in ['rank8','full_v','mean_only'])
 action_table='\n'.join(f"| {n} | {B[n]['proxy_gate_relative_pct']:.3f}% | {100*B[n]['gate_degradation_fraction']:.2f}% | {B[n]['positive_filtered_relative_pct']:.3f}% | {100*B[n]['pooled_positive_recovery']:.2f}% |" for n in ['rank8','full_v','mean_only'])
 text=f'''# E15：条件收益集中性与 Full-V 摘要信息诊断

**完成144个 head–tile、994个冻结反事实集合；不拟合任何参数，不生成新 QKV/视频，不执行新的全局求解。** 评价集为 horse_field 两个 seed 的72例/494集合；训练与开发场景仅作分列诊断，不参与本轮模型选择。数据、基线配置、共同合法动作面板沿用 E13/E14；在线16 probes、真实目标64 Query、Page64、R4/rank8、10%预算、own保护保持不变。

## 结论

1. 条件收益集中于少数状态。原 E14 门槛下14.68%的状态贡献58.10%的条件增益；但这不是已实现的在线触发规则。
2. **Full-V 提供了 rank8 之外的有用方向信息。** 完整空间集合余弦由0.174升至0.354，候选余弦由0.213升至0.410；仅完整 V 均值已达到0.346/0.400。
3. 固定方向评分仍无法可靠选动作。方向相关性改善没有解决幅度、分母和接受判断；三个方案直接执行都退化。不能称为在线 Router，也不能据此证明缓存信息不足。

当前结果支持优先研究如何使用已有128维 V 均值，而不是立即新增 K–V 交叉统计或优化 kernel。本轮没有证明某种新预测器会成功。

## A：首先修正 E14 的基线口径

E14 的7.6466%对应 j_mean=argmax_j mean_s(g_sj)，随后允许真实标签逐状态拒绝负收益。完美门控下真正最强的固定偏好是 j_plus=argmax_j mean_s(max(0,g_sj))；相对它的额外比例为 **7.5436%**。两者都是离线 Oracle，不是当前 Risk，也不能说 Risk 能恢复剩下的92%。

本轮以 j_plus 为主，同时保留全部 j_mean 结果。每个状态满足 G_cond=G_fixed+V；G_cond为共同面板内最佳单动作与no-op，不是全局最优集合。每个事件等权，事件内状态等权；分层贡献使用原1/状态数权重，不重新放大状态较少的事件。

- 原请求口径的明显优势状态占比：**{100*c['material_requested_weight']:.2f}%**；贡献该口径绝对条件增益的 **{100*c['material_requested_share_of_requested_V']:.2f}%**。
- 更强固定基线下，明显优势状态占比：**{100*c['material_strong_weight']:.2f}%**；贡献对应绝对条件增益的 **{100*c['material_strong_share_of_strong_V']:.2f}%**。
- 门槛均为 V≥max(1e-12,0.01×当前状态NMSE)。14.68%是等事件权重后的比例，不是简单的494个状态计数占比。

### 收益分布

| 状态 | 状态数 | 占总条件绝对收益 | 该类平均相对额外改善（百分点） |
|---|---:|---:|---:|
{state_table}

oracle_step1/2/4 是已有**局部交换**轨迹的集合，不代表这些集合获得全局最优证书。原始状态已去重，所以各步样本数不同。

初始 Risk 只贡献8.03%的总条件增益，在此类状态内条件收益约占条件 Oracle 单步收益3.54%；后续局部交换步骤合计贡献40.80%，扰动集合合计51.17%。第四次交换状态的相对额外改善约1.072个百分点，高于初始状态0.506个百分点，但最大单类绝对贡献来自四次扰动（30.17%）。**条件优势不只出现在后期自然交换，扰动状态对总量影响很大；不能外推线上状态频率。**

按层：layer15贡献76.01%的绝对条件收益，layer1/29分别13.59%/10.40%；但layer29的相对改善更高，约0.928个百分点。因此绝对贡献集中与相对难度集中不是同一回事。chunk2/5/8分别贡献43.53%/24.73%/31.74%，并非只在长历史出现。

按条件 Oracle 所选动作类型：历史→历史贡献57.14%，当前→历史30.89%，历史→当前9.75%，当前→当前2.22%。箭头表示移出→加入。页面年龄、step和逐状态完整统计见 A_groups.csv/A_states.csv。这是基于事后最优动作的描述性分类，不是可在线识别的触发条件，也不构成因果解释。

初始集合与后续集合的“最佳动作是否需要不同信息”不能仅靠动作编号回答。本轮保存初始最佳动作在后续集合的真实 regret，并按状态比较同一固定评分；这些结果能显示失配，不能证明哪组信息是必要或充分的。

## B：三种完全不拟合的方向代理

相同摘要 K 计算页面质量，四代表都具有相同token质量，所以在归一化质量中共同因子抵消。

- rank8：完整摘要输出差经现有 basis 投影，复现 E14；全空间余弦用伪逆解码回128维计算。
- Full-V：完整128维 V 均值＋现有rank8细节，构造 h=m(vhat−Ohat_all)。
- mean_only：同样的 K 质量，只保留128维 V 均值，并重新计算对应全摘要均值输出。

H是已选页面方向和；真实E是未选页面残差和。因此固定使用 −H 作为方向代理，候选代理为v=h_r−h_a。评分预先固定为 mean_q[2 H·v−||v||²]，等于该代理残差范数的下降量。没有真实残差、Dense输出、真实分母、真实NMSE归一化或拟合系数进入评分。

**这个评分忽略代理目标中的分母变化，不是原目标的等价式或下界。** 最终收益一律从冻结原始残差和实际变化分母计算完整64 Query的真实 J。这里检验的是一个预定义方向信息读出方式，不是摘要信息的最优解码上限。

### 方向与条件变化

| 表示 | 集合全128维余弦 | 候选全128维余弦 | 跨集合收益变化 Pearson | 去掉动作均值后的交互 Pearson |
|---|---:|---:|---:|---:|
{b_table}

收益变化相关性对同一候选跨所有无序状态对计算，然后每事件等权平均；**Pearson不是R²，不能与 E14 拟合探针的0.0869直接比较**。全部 Spearman、逐例和初始/后续状态结果也已保存。

Full-V的收益变化 Spearman 为0.4289，rank8为0.3614；动作内排名 Spearman从0.1325到0.1737，仍很低。完整均值已经解释了大部分方向改善，细节只提供小幅额外提升；这是本数据上的点估计，未作独立轨迹显著性声明。

### 最终动作质量

| 表示 | 代理分数>0才执行：平均相对改善 | 退化状态比例 | 真实标签事后拒绝负动作：相对改善 | 事后过滤收益／面板Oracle绝对收益 |
|---|---:|---:|---:|---:|
{action_table}

最后两列是使用真实标签的离线诊断，不能作为可部署收益。绝对汇总恢复率与相对NMSE均值的权重不同，不能混用。所有方案直接执行都失败，不能由方向余弦翻倍推断在线质量改善。

只看初始 Risk 状态，强制执行的平均相对改善分别为 rank8 −9.034%、Full-V −7.266%、均值 −6.119%；事后过滤后分别为1.096%、1.173%、1.351%。由此也不能说128维方向只对后期集合有信息。mean_only在部分后期状态有很大退化尾部，完整数据保留在 B_by_state.csv 与 B_actions.csv；不删除低NMSE导致的大相对误差。

方向代理既没有估计真实残差幅度，也没有估计分母修正。即使某种相关性改善，最高分动作仍可能错误。因此本轮**不能把失败归因于现有摘要信息不足，也不能确认需要增加 K–V 协方差缓存**。

## 审计与边界

144例独立选中动作的 FP64 直接重算，最大标签差 {audit['max_direct_gain_error']:.3e}；rank8评分差与E14点积公式独立核对，最大差 {audit['max_E14_score_difference_error']:.3e}。教师字段污染不改变在线代理；原数据734文件、E13缓存288文件哈希全部保持。未训练任何模型、未调符号或门槛、未按评价结果改变评分。

共同合法面板限制了动作空间；反事实集合并非独立注意力分布；场景已参与开发；无视频、post-W_O或线上性能结论。未实现触发器、K–V新统计、部署kernel或新Router，符合本轮范围。

## 交付与复现

- protocol.json、audit.json：固定定义和独立检查。
- A_states.csv/A_groups.csv：状态、层、历史长度、step、年龄及交换类型分布。
- B_events.csv/B_summary.csv/B_actions.csv/B_by_state.csv：方向、变化相关性、所有选中动作及真实收益。
- events/*/scores.npz：全部候选评分，留在完整工作目录，不放入精简包；原始贡献不重复拷贝。
- source：local_experiments/e15_summary_information/。

在已有原数据和E13/E14结果的仓库根目录，用原虚拟环境设置 OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1，依次运行：

```sh
python -m local_experiments.e15_summary_information.run
python -m local_experiments.e15_summary_information.audit
python -m local_experiments.e15_summary_information.report
```

run默认16进程，每进程单BLAS线程。代码会复用已有逐例结果；改变实现时应使用新的输出版本目录，不混用旧缓存。精简包是报告/代码/表格交付，复算仍依赖仓库冻结数据。
'''
 (OUT/'REPORT.md').write_text(text)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(12,4))
 names=['rank8','full_v','mean_only'];x=np.arange(3)
 ax[0].bar(x-.18,[B[n]['set_full_cosine'] for n in names],.36,label='Set');ax[0].bar(x+.18,[B[n]['candidate_full_cosine'] for n in names],.36,label='Candidate');ax[0].set_xticks(x,names);ax[0].set(ylabel='Full 128-D cosine',title='Frozen summary direction');ax[0].legend()
 ax[1].barh([r['group'] for r in state_rows],[100*r['strong_V_share'] for r in state_rows]);ax[1].set(xlabel='Share of absolute conditional value (%)',title='Evaluation states, equal-event weights');fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)
 package=OUT/'review';package.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.csv','.json','.md','.png']:shutil.copy2(p,package/p.name)
 src=package/'source'/'local_experiments'/'e15_summary_information';src.mkdir(parents=True,exist_ok=True)
 for p in (ROOT/'local_experiments/e15_summary_information').glob('*.py'):shutil.copy2(p,src/p.name)
 dump(package/'checksums.json',{str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p.name!='checksums.json'})
 target=OUT.parent/'e15_summary_information_review.tar.gz'
 with tarfile.open(target,'w:gz') as t:t.add(package,arcname='e15_summary_information_review')
 print(dict(report=str(OUT/'REPORT.md'),package=str(target),bytes=target.stat().st_size))
if __name__=='__main__':run()
