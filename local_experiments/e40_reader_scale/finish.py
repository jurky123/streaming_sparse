from .report import *
import tarfile

def md(rows,keys):
 out=['|'+'|'.join(keys)+'|','|'+'|'.join(['---']*len(keys))+'|']
 for r in rows:
  vals=[]
  for k in keys:
   v=r.get(k);vals.append('—' if v is None else f'{v:.5g}' if isinstance(v,float) else str(v))
  out.append('|'+'|'.join(vals)+'|')
 return '\n'.join(out)
def load(name):return list(csv.DictReader((DEST/name).open()))
def main():
 assert json.loads((DEST/'audit.json').read_text())['passed'];
 if not (DEST/'collision_contexts.csv').exists():(DEST/'collision_contexts.csv').write_text('seed,split,identity,n,unique_queries,beta_range\n')
 sat=load('saturation_summary.csv');bound=load('collision_bounds_summary.csv');sc=load('scale_error_association.csv');iv=load('scale_reconstruction.csv');pr=load('similar_page_summary.csv');near=load('near_phi_summary.csv');br=load('beta_interval_diagnostic.csv');main=[];inversion=[];association=[];pairs=[];limits=[];ranges=[]
 for seed in [17,29]:
  for split in ['train','development']:
   r=next(x for x in sat if int(x['seed'])==seed and x['split']==split and x['dimension']=='ALL' and x['role']=='all');b=next(x for x in bound if int(x['seed'])==seed and x['split']==split and x['dimension']=='ALL' and x['kind']=='all');done=json.loads((DEST/f'{split}_s{seed}/complete.json').read_text());main.append(dict(seed=seed,split=split,samples=int(r['samples']),max_abs_activation=float(r['abs_activation_max']),max_abs_phi=float(r['abs_phi_max']),exact_any_pct=100*float(r['exact_any_samples_rate']),near99_any_pct=100*float(r['near99_any_samples_rate']),near999_any_pct=100*float(r['near999_any_samples_rate']),mean_derivative=float(r['mean_derivative']),collision_groups=done['collision_groups'],collision_samples=done['collision_samples']))
   for role in ['harmful_predicted_endpoint','positive_predicted_endpoint','endpoint_pair','background']:
    rr=[x for x in sat if int(x['seed'])==seed and x['split']==split and x['dimension']=='ALL' and x['role']==role]
    if rr:
     x=rr[0];limits.append(dict(seed=seed,split=split,role=role,samples=int(x['samples']),near99_any_pct=100*float(x['near99_any_samples_rate']),exact_any_pct=100*float(x['exact_any_samples_rate']),beta_mae=float(x['beta_mae_event_equal'])))
   for block in ['K','mean','detail','stats']:
    zz=[x for x in iv if int(x['seed'])==seed and x['split']==split and x['block']==block];n=sum(int(x['n']) for x in zz);med=[float(x['median_relative_error']) for x in zz if x['median_relative_error']];inversion.append(dict(seed=seed,split=split,block=block,valid_pct=100*sum(int(x['n'])*float(x['recoverable_fraction']) for x in zz)/n,rms_one_pct=100*sum(int(x['n'])*float(x['normalized_rms_eq_one_fraction']) for x in zz)/n,head_median_relative_min=min(med) if med else None,head_median_relative_max=max(med) if med else None))
    zz=[x for x in sc if int(x['seed'])==seed and x['split']==split and x['role']=='all' and x['stratum']=='all' and x['block']==block];cb=[float(x['scale_beta_spearman']) for x in zz if x['scale_beta_spearman']];cl=[float(x['scale_local_spearman']) for x in zz if x['scale_local_spearman']];association.append(dict(seed=seed,split=split,block=block,head_groups=len(zz),median_scale_beta_rho=float(np.median(cb)) if cb else None,beta_positive_heads=sum(x>0 for x in cb),median_scale_local_rho=float(np.median(cl)) if cl else None))
   for b in ['ALL','0']:
    zz=[x for x in pr if int(x['seed'])==seed and x['split']==split and x['pair_kind']=='nearest' and x['feature_distance_train_quartile']==b];rho=[float(x['scale_contrast_beta_spearman']) for x in zz if x['scale_contrast_beta_spearman']];pairs.append(dict(seed=seed,split=split,distance_bin=b,head_groups=len(zz),median_scale_contrast_beta_rho=float(np.median(rho)) if rho else None,positive_heads=sum(x>0 for x in rho),median_beta_target_rmse=float(np.median([float(x['median_beta_target_rmse']) for x in zz]))))
   zz=[x for x in br if int(x['seed'])==seed and x['split']==split and x['role']=='all'];ranges.append(dict(seed=seed,split=split,current_beta_interval_target_outside_pct=100*sum(int(x['outside_true_target']) for x in zz)/sum(int(x['n']) for x in zz)))
 table(DEST/'headline.csv',main);table(DEST/'scale_headline.csv',association);table(DEST/'pair_headline.csv',pairs);table(DEST/'inversion_headline.csv',inversion);table(DEST/'beta_interval_summary.csv',ranges)
 collisions=sum(r['collision_groups'] for r in main);exact=sum(r['exact_any_pct'] for r in main)
 text=r'''# E40：Reader尺度保留与Query可辨识性诊断

**无训练诊断已完成，未启动尺度探针或Reader续训，Test保持封存。** 固定原E29两个种子、C摘要、Adaptive φ、B16、Risk和Medium。训练8条轨迹（每种子1152事件）用于主要统计，开发2条轨迹（每种子288事件）用于分列确认；不将页面、Query或模型种子当成独立视频样本。

## 主要发现

- 两种子训练与开发均无精确±1饱和、无同页四维φ精确碰撞；平均tanh导数约0.87–0.92。任一分量近饱和≥.99的比例约0.33%–2.39%，不能据此认定大量关键Query被压成同一输入。
- 四块FP32输入的尺度反解均有效；各layer/head中位相对误差范围约0.004%–0.462%。因此“尺度完全丢失”不符合本次实现数据；显式尺度的潜在作用更像改善学习条件，仍需对照验证。
- 最相近特征距离档中，尺度差异与真实β差异的相关性中位数约0.02–0.04，不足以直接推导添加尺度能提高交换收益。
- 约60%–63%的真实β落在当前已学系数区间之外。这是现有参数幅度覆盖不足的描述，不是所有五项读出器的能力上界，也不是tanh饱和导致失败的证明。

## 范围与实现

原缓存只保存开发φ、β和局部响应，没有训练C摘要、四块RMS和tanh前激活。用户明确授权“允许重建”后，离线读取已有KV，仅构建并保存冻结C摘要；真实页面N/Z/output仍使用规范缓存，没有重算原始Attention或采集数据。开发φ、预测β和真实β重放一致，动作身份保持一致。新增文件只是离线诊断缓存，不是部署摘要扩容。

## A：实际精度下的饱和与精确碰撞

'''+md(main,['seed','split','samples','max_abs_activation','max_abs_phi','exact_any_pct','near99_any_pct','near999_any_pct','mean_derivative','collision_groups','collision_samples'])
 if collisions==0:text+='\n全部已检查范围内没有同一物理页/相同摘要下的四维φ精确重复组；因此本次精确碰撞下界为0，不能用它解释现有β误差。零下界只是没有找到这种碰撞，不证明Reader表达充分。\n'
 if exact==0:text+='\n没有分量在FP32实际tanh计算中精确取到±1。有限的近饱和样本仍单独统计，不能把理论上有界直接解释成当前数据中发生了大规模精确压平。\n'
 text+=r'''
精确碰撞不做十进制舍入，四个FP32值必须逐项相等，仅把±0视为同值。历史页以已核验的token/KV与C摘要哈希识别；当前noisy页仅做同事件内比较。完全相同输入的加权组内β方差才构成相应回归下界；近似相同不作此声称。该下界也不是集合收益上界。

## 决策相关页面

'''+md(limits,['seed','split','role','samples','near99_any_pct','exact_any_pct','beta_mae'])
 text+=r'''
角色允许重叠。endpoint_pair含原预测和真实最佳动作端点；harmful/positive_predicted_endpoint按真实Q64动作收益区分。开发另保留E38正端点区域贡献标记，只有区域归因，不能把它当作每个页面的独立因果贡献；训练事件没有E38归因，不伪造对应标记。全层与轨迹分列见saturation_summary.csv。

## 同历史页跨上下文的近似特征

同chunk同step其他tile、同chunk其他step、其他chunk与严格过去chunk分别保存。每个Query只在对应合法支持中找最近φ，统计距离和真实β差异，不拟合系数、不读取当前标签预测其他上下文。样本支持不同，不能直接把这些表当作受控泛化恢复率。

开发严格过去chunk的描述性结果：

'''
 nr=[dict(seed=int(x['seed']),layer=int(x['layer']),page_groups=int(x['page_groups']),median_phi_distance=float(x['median_nearest_phi_distance']),median_beta_gap=float(x['median_mean_beta_gap'])) for x in near if x['split']=='development' and x['scope']=='strict_past_chunk']
 text+=md(nr,['seed','layer','page_groups','median_phi_distance','median_beta_gap'])
 text+=r'''
即使近φ对应不同β，也不能称为不可辨识：大系数可能放大微小差异，且当前φ只有4维，对Query方向的压缩与tanh饱和是不同限制。

## B：尺度未显式输入，不等于数学上消失

令原块RMS为s，归一化后RMS为u。在精确算术、s>0时：

\[
u=\frac{s}{s+10^{-4}},\qquad s=10^{-4}\frac{u}{1-u}.
\]

因此该映射并非严格尺度不变。s较大时u很接近1，尺度被编码成很小的范数差异，数值舍入或学习条件可能很差。以下用实际FP32输入块重新计算RMS并反解，只是数值诊断，未把反解尺度提供给Reader。

'''+md(inversion,['seed','split','block','valid_pct','rms_one_pct','head_median_relative_min','head_median_relative_max'])
 text+=r'''
valid指RMS<1且原尺度>0；原尺度为0也不计入相对误差统计，不应把所有invalid都解释成量化丢失。误差范围是12个layer/head组各自中位相对误差的最小/最大值，不是全样本极值。FP64独立重算用于区分归一化恒等式与FP32实现精度。显式增加四个尺度可能改变优化条件，但本轮没有证明它必需或有效。

## 尺度与错误的分层关联

按layer/head内计算Spearman，再列12组的中位数和正相关组数。没有用跨layer整体相关性声称尺度是主因。训练唯一页面的log尺度p10/p90用于低/中/高分层，开发沿用相同边界；轨迹单独统计，不把大量重复页面当独立证据。

'''+md(association,['seed','split','block','head_groups','median_scale_beta_rho','beta_positive_heads','median_scale_local_rho'])
 text+=r'''
详细角色、低/中/高尺度与逐轨迹结果见scale_error_association.csv。误差关联仍可能由Key方向、页面内容、Query分布等共同因素解释；没有执行因果干预，也没有训练一个尺度补偿器。

## 相似归一化输入页面的目标差异

每个事件在原712维归一化输入上选最近的另一页面，用相同64个Query比较真实β和局部响应目标。固定循环偏移页面作为描述性对照。距离四分位只由训练集确定，按layer/head固定。

'''+md(pairs,['seed','split','distance_bin','head_groups','median_scale_contrast_beta_rho','positive_heads','median_beta_target_rmse'])
 text+=r'''
ALL为最近页面全集，0为训练距离最低四分位。按距离分箱只降低一种混杂，不能控制所有归一化特征差异。这不是“同一归一化输入只有尺度不同”的干预，也不证明添四个标量一定改善选页。完整最近/固定对照与响应目标见similar_page_summary.csv。

## 当前β系数范围

'''+md(ranges,['seed','split','current_beta_interval_target_outside_pct'])
 text+=r'''
范围是当前已学系数b0±Σ|bj|，不是所有可能页面系数的表达上界。真值超出当前范围表示现有参数幅度不足，不能单独归因于tanh：重新学习更大系数也会改变该范围，而本轮没有做这种训练。

## 结论与后续边界

E40区分了三件事：理论有界、实际饱和碰撞、尺度通道的数值与优化条件。它不支持仅凭归一化和tanh形式就宣布必要信息已经被删除。另一方面，没有精确碰撞也不排除Query方向压缩、页面响应模型和尺度条件共同造成预测困难。

本轮只交付可辨识性与尺度关联诊断，不启动条件性尺度探针，不修改Reader或β分支，不运行新的集合收益实验。不能声称改善了Oracle恢复率或可靠性。若以后测试显式尺度，必须保持其余结构、数据、预算一致，用训练轨迹内部整轨迹留出比较；应把它作为优化/可预测性假设，而非已经证明的信息恢复方案。先前E39停止损失搜索的结论保持有效。

## 复现与审计

实际φ前激活和Page Encoder输出复算、四块RMS独立FP64核验、开发身份与E28全量一致、抽样Python精确集合计数验证碰撞、输入范围检查见audit.json。所有摘要/Reader/φ保持原检查点。审计不再读取原始KV。

protocol.json记录授权、阈值与范围。新增的跨tile/step/chunk最近Query分列是落实原方案的分析补全，不涉及模型或阈值搜索。原始逐页面和逐事件记录保存在本地子目录；评审包收录报告、汇总、协议和源码，不收录原始KV或大诊断数组。
'''
 (DEST/'REPORT.md').write_text(text)
 source=list((ROOT/'local_experiments/e40_reader_scale').glob('*.py'))+[ROOT/'local_experiments/e40_reader_scale/README.md'];items=[p for p in DEST.iterdir() if p.suffix in ['.csv','.json','.md'] and p.name!='package_manifest.json']+source;manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in items];dump(DEST/'package_manifest.json',manifest)
 with tarfile.open(DEST/'e40_review.tar.gz','w:gz') as tar:
  for p in items+[DEST/'package_manifest.json']:tar.add(p,arcname=str(p.relative_to(ROOT)))
 with tarfile.open(DEST/'e40_review.tar.gz') as tar:
  for x in manifest:assert hashlib.sha256(tar.extractfile(x['path']).read()).hexdigest()==x['sha256']
 print('FINISHED E40',flush=True)
if __name__=='__main__':main()
