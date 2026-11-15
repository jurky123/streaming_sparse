from .core import *
import shutil,tarfile,ast

def readcsv(name):return list(csv.DictReader((OUT/name).open()))
def run():
 audit=json.loads((OUT/'audit.json').read_text());assert audit['passed'];pca_lock=json.loads((OUT/'pca_lock.json').read_text());B=[json.loads(p.read_text()) for p in sorted((OUT/'moments').glob('*/result.json'))];assert len(B)==2
 for name,key in [('B_page_metrics','page_metrics'),('B_performance','performance'),('B_selected_actions','selected')]:table(name+'.csv',[x for r in B for x in r[key]])
 dump(OUT/'B_logit_statistics.json',{r['event_id']:r['logit_statistics'] for r in B});dump(OUT/'B_snapshot_audits.json',{r['event_id']:r['audit'] for r in B})
 en=readcsv('A_energy_summary.csv');perf=readcsv('A_performance_summary.csv');sign=readcsv('A_interaction_sign_summary.csv')
 def energy(space,target):return float(next(r['energy_retention'] for r in en if r['split']=='evaluation' and r['scope']=='risk_set' and r['space']==space and r['target']==target))
 def pp(space,M=64,scope='risk'):return next(r for r in perf if r['split']=='evaluation' and r['scope']==scope and r['space']==space and r['probes']==str(M))
 spaces=['current_rank8','net_PCA16','V_PCA16','Z_PCA16','full128']
 et='\n'.join('| '+name+' | '+' | '.join(f'{100*energy(name,t):.2f}%' for t in ['net','V','Z'])+' |' for name in spaces)
 pt=[]
 for n in spaces:
  row=pp(n);si=next(r for r in sign if r['split']=='evaluation' and r['scope']=='risk' and r['space']==n and r['probes']=='64');pt.append(f"| {n} | {100*float(si['I_sign_agreement_Q64']):.2f}% | {float(row['gated_relative_pct']):.3f}% | {100*float(row['pooled_gated_recovery']):.2f}% | {100*float(row['gated_degradation_rate']):.2f}% |")
 pt='\n'.join(pt)
 bt=[];mt=[];ib=[]
 for result in B:
  case=result['event_id'].split('_s101')[0];rows=result['performance'];p={(r['panel'],r['method'],r['probes']):r for r in rows};oracle=p['full_legal','exact_dense_reference',64]['gated_relative_pct']
  for method in ['M0','M1','M2','exact_deltas_online_center','exact_dense_reference']:
   x=p['full_legal',method,16];y=p['full_legal',method,64];bt.append(f"| {case} | {method} | {x['gated_relative_pct']:+.3f}% | {y['gated_relative_pct']:+.3f}% | {100*y['gated_relative_pct']/oracle:.2f}% |")
   if method in ['M0','M1','M2']:ib.append(f"| {case} | {method} | {y['I_nmse']:.3f} | {y['B_nmse']:.3f} | {100*y['I_sign_accuracy']:.2f}% |")
  for method in ['M0','M1','M2']:
   x=next(r for r in result['page_metrics'] if r['method']==method and r['scope']=='all');un=next(r for r in result['page_metrics'] if r['method']==method and r['scope']=='risk_unselected');mt.append(f"| {case} | {method} | {x['deltaZ_nmse']:.4f} | {x['deltaN_nmse']:.4f} | {x['net_residual_nmse']:.4f} | {x['net_direction_cosine']:.4f} | {un['net_residual_nmse']:.4f} |")
 bt='\n'.join(bt);mt='\n'.join(mt);ib='\n'.join(ib)
 text=f'''# E19：净残差表示容量与K–V响应矩诊断

**完成阶段A：144个冻结head–tile、994个反事实集合；完成阶段B：仅两份已有原始QKV快照。没有新视频/QKV采集、Router训练、摘要修改、低秩响应缓存实现或全局优化。** 原Q64/Page64、R4/rank8、10%预算、own保护保持不变。

## 主要结论

1. **低维真实残差仍有决策价值，但不是高保真重建。** 训练拟合的净残差PCA16在评价集初始Risk仅保留17.28%的集合净残差能量，却恢复80.84%的共同面板Oracle汇总收益；交互符号相符87.88%。这些使用真实投影残差和真实分母，是表示容量诊断，不是在线预测结果。
2. **完整二阶矩出现了明确但很有限的正信号。** 两快照上，M2在全部14,760个合法交换中选择的动作，使真实NMSE分别降低11.27%和4.60%；同面板最佳真实单次交换为17.56%和9.20%。M1在horse样本仍退化，不能据此宣布单独协方差已经足够。
3. **还不能据此设计部署缓存。** M2使用完整δ²V混合响应，并非只补Key方差；当前I符号准确率约69%，完整矩存储比原始KV更大，且仅验证同一层/head/chunk/step的两例。

## 范围与冻结规则

A沿用city_walk seed101训练36例、seed303开发36例、horse_field两seed评价72例。三个PCA只在训练36例拟合，开发/评价不选择维度、权重或子空间。所有场景此前参与研究，不能称全新场景验证。

B仅使用：

- city_walk_s101_c5_l15_s0_h8_t61
- horse_field_s101_c5_l15_s0_h8_t61

两者均410页、41页精算，测试现有Risk初始集合；所有410页都作矩诊断，尚未实现仅历史页提交时缓存并与当前页路径整合。

## A：三种PCA与当前rank8

采用**未中心化PCA/SVD二阶矩**，不减/恢复全局均值；每个训练事件的二阶矩按其总能量归一化后等权平均。这样投影是通过零点的线性映射，保持 (eV+eZ)P=eP。每个PCA是跨head共享的16维空间；当前rank8是各head已有basis的QR正交化张成空间。没有单独拟合每个head的PCA。

净残差、V响应误差、质量误差分别产生一个PCA16。每个空间都投影同一份真实净残差用于决策，而不是用V-only或Z-only残差选页；各分量能量另作诊断。

### 初始Risk集合能量覆盖（评价72例，逐例等权）

| 空间 | 净残差 | V响应分量 | 质量分量 |
|---|---:|---:|---:|
{et}

当前head-specific rank8对V分量的覆盖甚至高于这次global PCA16。**本轮没有证明缺失V响应可以在共享16维空间中高保真重建，也不能据此证明它的本征维度很高**：共享空间、训练轨迹和跨head迁移都限制了这个结论。全部页面、未选页与后续集合的覆盖见CSV。

### 投影后的决策能力

对投影净残差执行原始真实变化分母下的单次交换评价，真实d和分母保持独立、未压缩；最终收益总是用完整128维和全部64 Query真实目标测量。使用原E13共同合法候选面板，所有方法同一页面集合和预算。

| 空间（64 Query） | I符号相符 | 平均相对改善 | 汇总Oracle收益恢复 | 退化率 |
|---|---:|---:|---:|---:|
{pt}

恢复率=汇总实际增益/汇总同面板Oracle增益；不是逐例恢复比例的均值。投影仍有退化，不能作为可用路由器。Full128严格复现原Oracle，用于核对数值。

16 probes下，净PCA16恢复58.04%、V-PCA16恢复64.36%、Z-PCA16恢复45.92%，当前rank8恢复35.70%；完整128维也只有69.78%。因此Query覆盖仍是独立限制，不能只看64 Query的表示容量。

后续反事实集合，净PCA16的64 Query收益恢复约75.75%，平均相对改善2.762%，退化约15.23%；全部分列数据保留。A阶段不是在测试矩统计是否能预测这些投影坐标。

## B：已有原始快照的矩展开

### 分组身份与实际定义

原始物理页按 reshape(P,4,16,128) 分组，匹配SummaryOps.build；没有重新分组。token IDs、Query、存储K代表、V均值、detail、BF16 basis与冻结缓存逐项相同。另用子组真实K/V均值核对缓存偏差统计，最大相对误差低于0.39%，符合BF16统计存储舍入。

κ取实际学习并重心校正后的K代表，ν取实际存储均值＋Haar/rank8 detail重建的代表V。δ=q·(k−κ)/√128，δ均值没有假定为零。各方法定义为：

- **M0**：Z0=16exp(ell)(1+meanδ)，N0=16exp(ell)(1+meanδ)meanV。只含线性K均值偏移与真实V均值偏移。
- **M1**：Z1=Z0；N1=16exp(ell)mean[(1+δ)V]，在M0上增加完整Cov(V,K)q/√128。
- **M2**：Z2=16exp(ell)mean[1+δ+δ²/2]；N2=16exp(ell)mean[(1+δ+δ²/2)V]。包含完整二阶混合项mean(δ²V)，不能将收益归功于Key方差一项。

先按组修正，再汇总页面；所有页面使用同一Query的**摘要总logZ尺度**，没有每页独立缩放失去相对质量。预测ΔN/ΔZ均减去原四代表摘要贡献，e_hat=ΔN_hat−Ohat_all·ΔZ_hat。

决策时混合分母由选中页的矩估计Z和未选页的原摘要Z构成，不读取真实Dense输出或真实d。只在离线评价时转为共同DenseZ归一化单位比较误差。没有将原始KV每Query扫描成本伪装成低成本线上实现。

### 近似误差

以下DeltaN、DeltaZ及净残差列都是平方误差/真实量平方能量（NMSE），不是幅度相对误差。方向余弦逐Query-page计算后平均。

| 快照 | 方法 | ΔZ NMSE | ΔN NMSE | 全页净残差NMSE | 净残差余弦 | Risk未选页净残差NMSE |
|---|---|---:|---:|---:|---:|---:|
{mt}

M2的净残差方向明显改善，但Risk未选页的幅度误差仍比全页结果更高，不能只报告0.95以上的余弦。

### 真实交换质量：全部合法页面对

40个可移出页×369个可加入页＝14,760个合法交换；own不可移出，精确页数保持41。每方法允许score≤0时no-op，使用各自估计的当前/交换后分母。没有删除难动作或裁剪分母。

| 快照 | 方法 | 16 probes真实NMSE改善 | 64 Query真实NMSE改善 | 相对64Q真实最佳单次动作 |
|---|---|---:|---:|---:|
{bt}

exact_deltas_online_center使用真实ΔN/ΔZ但仍以在线Ohat_all中心化，是分离中心化误差的离线参照；exact_dense_reference再使用真实OD，为该单次交换面板的完整Oracle。两者均不可部署，不叫全局最优集合。

共同候选面板结果也已保存：64Q M2在city/horse分别改善6.897%/4.355%，同面板Oracle为9.279%/5.695%。不能拿它与全14,760动作的Oracle混比。

### 交互与自身能量仍有误差

| 快照／全合法面板64Q | 方法 | I NMSE | B NMSE | I符号准确率 |
|---|---|---:|---:|---:|
{ib}

M2虽选到了好动作，I幅度误差仍大，符号准确率约69%，尚未解决完整误差预测。真实Δ但在线中心化的参照也存在交互误差，说明Ohat_all≠OD的影响需要保留。正收益动作召回、符号准确率、强制执行/门控和逐动作结果全部在B_performance.csv与B_selected_actions.csv。

### Taylor适用范围

两例子组max|δ|中位数分别为1.901和1.487，99分位为5.421和5.061，最大9.936和8.907。仅5.29%/15.65%的Query-group满足max|δ|≤1；δ的组内标准差中位数0.892/0.697，|meanδ|均值0.175/0.122。

因此不能声称普遍处于“小扰动”。按页面内四组最大偏移分层，M2在max|δ|≤2区间的净残差误差显著较小；>2区间误差更高。无页面满足预登记的≤0.5桶，空桶不补造数据，实际计数保留在CSV。

M0/M1在city有约0.678%的Query-group、0.324%的Query-page出现非正估计质量；horse对应0.233%/0.145%。这些不是合法概率质量，未修补或裁剪。本次所有候选**混合总分母仍为正**，所以诊断评分可计算，但不意味着M0/M1可直接部署。M2二次多项式始终为正，本次无负质量。

## 容量与成本边界

第一阶协方差使用完整128×128矩阵并核对收缩；第二阶通过原16 token的完整δ²V收缩计算，另在抽样组显式构造128×128×128张量独立验证，未省略混合项。没有永久物化所有二阶张量，避免无意义的数十GiB缓存。

如果直接按四子组存FP32完整矩：Cov(V,K)约256KiB/页；二阶K–K–V张量约32MiB/页（尚未计其他统计），而BF16原始Page64的K/V合计约32KiB。**这轮证明的是完整统计的离线信息价值，不是可行的存储设计。** 暂未验证低秩矩压缩、提交成本、跨去噪复用误差或线上延迟。

## 下一步判断

目前只有两个层次的有限积极结果：真实残差的16维投影可保留不少决策收益；两份已有快照上的完整二阶响应可产生正收益交换。但共享PCA16并不高保真重建V误差，一阶协方差也不稳定，二阶完整矩更不可能直接缓存。

这为继续做小规模的**响应矩压缩与适用域诊断**提供依据，尚不足以直接实现Residual Response Sketch、训练新Router或开展大规模闭环实验。本轮没有证明低阶矩在其他层/head/历史长度/去噪步有效。

## 审计、数据与复现

审计通过：A的PCA仅来自36训练例、特征向量满足独立重算二阶矩的本征方程、投影正交；Full128复现原收益，已选动作独立重算。B快照身份/子组统计/一阶协方差/完整二阶张量/原token贡献/冻结页贡献/真实动作收益均核验。原734个文件和288个E13文件哈希不变。

- A_*_per_event.csv / A_*_summary.csv：三种目标覆盖、交互符号、Q16/Q64与初始/后续决策。
- pca16.npz、pca_lock.json：冻结基与训练定义。
- B_page_metrics.csv、B_performance.csv、B_selected_actions.csv：两快照全部方法与面板。
- B_logit_statistics.json、B_snapshot_audits.json、audit.json：波动范围、异常质量与独立检查。
- capacity/*/scores.npz、moments/*/scores.npz：完整候选评分留在工作目录；精简包不重复存原始数据与大评分缓存。

在原仓库环境和冻结数据可用时，设置OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1，依次运行：

```sh
python -m local_experiments.e19_response_moments.fit_pca
python -m local_experiments.e19_response_moments.capacity
python -m local_experiments.e19_response_moments.moments --event city_walk_s101_c5_l15_s0_h8_t61 --device cuda:0
python -m local_experiments.e19_response_moments.moments --event horse_field_s101_c5_l15_s0_h8_t61 --device cuda:1
python -m local_experiments.e19_response_moments.audit
python -m local_experiments.e19_response_moments.report
```

注意：审计中用于核对增补I/B指标未改变动作的result_before_IB.json是本次运行记录；干净复现若不存在该历史文件，脚本会自动跳过这项历史一致性检查，其余数值审计照常执行。源码、协议与身份哈希随包提供，依赖仓库既有模块/冻结数据。本轮是本地交付，不声称已推送GitHub。
'''
 (OUT/'REPORT.md').write_text(text)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(12,4));names=spaces[:-1];x=np.arange(len(names));ax[0].bar(x-.2,[100*energy(n,'net') for n in names],.4,label='Net set energy retained');ax[0].bar(x+.2,[100*float(pp(n)['pooled_gated_recovery']) for n in names],.4,label='Oracle gain recovered');ax[0].set_xticks(x,names,rotation=15);ax[0].set(ylabel='Percent',title='A: true residual projection, evaluation72');ax[0].legend(fontsize=8)
 for offset,result in [(-.2,B[0]),(.2,B[1])]:
  vals=[next(r['gated_relative_pct'] for r in result['performance'] if r['panel']=='full_legal' and r['probes']==64 and r['method']==m) for m in ['M0','M1','M2']];ax[1].bar(np.arange(3)+offset,vals,.4,label=result['event_id'].split('_s101')[0])
 ax[1].set_xticks(np.arange(3),['M0','M1','M2']);ax[1].axhline(0,color='black',lw=.5);ax[1].set(ylabel='Actual relative NMSE improvement (%)',title='B: two snapshots, full legal swaps');ax[1].legend();fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)
 pkg=OUT/'review';pkg.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.png','.npz']:shutil.copy2(p,pkg/p.name)
 src=pkg/'source/local_experiments/e19_response_moments';src.mkdir(parents=True,exist_ok=True)
 for p in (ROOT/'local_experiments/e19_response_moments').glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,src/p.name)
 dump(pkg/'source_identities.json',{p.parent.name:json.loads(p.read_text())['audit'] for p in (OUT/'capacity').glob('*/result.json')})
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 target=OUT.parent/'e19_response_moments_review.tar.gz'
 with tarfile.open(target,'w:gz') as t:t.add(pkg,arcname='e19_response_moments_review')
 print(dict(report=str(OUT/'REPORT.md'),package=str(target),bytes=target.stat().st_size))
if __name__=='__main__':run()
