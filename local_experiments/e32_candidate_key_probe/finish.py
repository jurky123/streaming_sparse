from .run import *
import csv,tarfile,shutil

def read(name):return list(csv.DictReader((DEST/name).open()))
def main():
 summary=json.loads((DEST/'summary.json').read_text());gates=json.loads((DEST/'gates.json').read_text());audit=json.loads((DEST/'independent_audit.json').read_text());ss={(r['seed'],r['method'],r['policy']):r for r in summary};methods=BASES+[f'KeyProbe-b{b}' for b in BS];rep=read('sampling_seed_summary.csv');st=read('sampling_stability.csv');qq=read('quality_summary.csv');co=read('conditional_summary.csv');gg=read('generalization.csv')
 lines=['# E32：候选页 Key-only 质量探测','', '**结论：'+('信息筛选通过，但' if gates['information_pass'] else '预登记信息筛选未通过，且')+('执行可靠性门槛未通过。' if not gates['full_pass'] else '全部预登记门槛通过。')+'** 本轮不训练、不访问 Test、不实现 Kernel，不增加到 b>4。','', '## 范围与固定设计','', '两个冻结 C/E29 模型种子17/29，各使用同一两条开发轨迹的288个 head–tile。M16决策、Q64真实 pre-W_O NMSE评价；Page64/R4、10%预算和 own保护不变。每事件初始Medium含16移出×32加入，共48个候选页，仅这些页面新增Key探测，背景β及预测局部响应完全不变。每个16-token子组均匀无放回抽 b=1/2/4，固定5个采样种子17/29/43/71/101，嵌套样本。采样位置由页面token身份和预登记种子决定，先于收益评价冻结。','', '每采样种子都是一次独立随机计划的完整实验；表中探测结果平均五次，不选择最佳种子、不集成五次信息。模型种子并非独立视频；1440事件×采样观测仍只对应288相关事件、两条轨迹。','', '质量估计为 logZprobe=logsumexp(Q·Ksample/√128)+log(16/b)，β=logZprobe−logZsummary。Z估计在抽样随机性下无偏，logZ与动作选择不是无偏；无beta裁剪、无教师校准。Set沿用解析局部响应与共同Softmax集合目标，Independent用完全相同的信息做页面独立残差能量交换；均允许no-op。','', '## 实际集合收益','', '恢复率=Σ真实执行收益/Σ同池真实最佳正收益。近零收益判断阈值1e−12；相对指标仅统计初始NMSE>1e−12，绝对汇总不删除事件。教师不是可部署方法。','', '|方法|seed17 Oracle恢复|seed29 Oracle恢复|seed17 汇总NMSE改善|seed29 汇总NMSE改善|','|---|---:|---:|---:|---:|']
 for m in methods:
  a,b=[ss[s,m,'Set'] for s in [17,29]];lines.append(f"|{m}|{a['R_executed']:.2%}|{b['R_executed']:.2%}|{a['pooled_improvement_pct']:.3f}%|{b['pooled_improvement_pct']:.3f}%|")
 lines+=['','|seed|方法|Set恢复|Independent恢复|提议＋事后真门控恢复|逐例平均改善|退化率|no-op率|P10|最差|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for seed in [17,29]:
  for m in methods:
   x=ss[seed,m,'Set'];y=ss[seed,m,'Independent'];lines.append(f"|{seed}|{m}|{x['R_executed']:.2%}|{y['R_executed']:.2%}|{x['R_proposal']:.2%}|{x['mean_improvement_pct']:.3f}%|{x['degraded']/x['n']:.2%}|{x['noop']/x['n']:.2%}|{x['p10_pct']:.3f}%|{x['worst_pct']:.3f}%|")
 lines+=['','b=2恢复Candidate-True增益缺口约37.8%／3.9%，b=4约76.1%／40.7%。较低成本的b=2在seed29几乎未弥补教师差距，b=4则需稳态14.6%的额外Key读取和约2.72 GFLOPs/层新增QK。','','Set−Independent 必须与 no-op 一起看。事后真门控读取真实收益，只诊断提议质量，不能作为在线结果。探测行的退化/no-op比例分母为1440，基线和教师为288；完整计数及真实汇总量见 summary.csv。','', '## 采样稳定性与预登记门槛','', '|seed|b|Candidate-True缺口恢复|五采样恢复率范围|超过E29的采样数|动作两两不同率|五次均正收益事件率|信息门槛|可靠性门槛|','|---|---:|---:|---:|---:|---:|---:|---|---|']
 for g in gates['results']:
  seed=g['seed'];b=g['b'];m=f'KeyProbe-b{b}';rs=[r for r in rep if int(r['seed'])==seed and r['method']==m and r['policy']=='Set'];rr=[float(r['R_executed']) for r in rs];stab=[r for r in st if int(r['seed'])==seed and r['method']==m];dis=np.mean([float(r['pair_disagreement']) for r in stab]);pos=np.mean([int(r['positive_samples'])==5 for r in stab]);lines.append(f"|{seed}|{b}|{g['candidate_gap_recovery']:.2%}|{min(rr):.2%}–{max(rr):.2%}|{sum(v>ss[seed,'E29','Set']['R_executed'] for v in rr)}/5|{dis:.2%}|{pos:.2%}|{'通过' if g['information_pass'] else '未通过'}|{'通过' if g['reliability_pass'] else '未通过'}|")
 lines+=['','信息门槛：两模型种子均超过E29和同信息Independent、恢复至少50%候选教师增益缺口、两开发轨迹均正汇总收益、至少4/5采样超过E29。b1为成本诊断，正式继续候选仅b2/b4。可靠性另要求逐例均值非负、退化≤15%、P10和最差优于E29。五次重复不是统计显著性检验。失败项见 gates.json。','', '|seed|方法|开发轨迹|汇总NMSE改善|Oracle恢复|','|---|---|---|---:|---:|']
 for r in gg:
  if r['dimension']=='trajectory':lines.append(f"|{r['seed']}|{r['method']}|{r['value']}|{float(r['pooled_improvement_pct']):.3f}%|{float(r['R_executed']):.2%}|")
 lines+=['','b=4的动作两两不同率仍约86%–87%，五次采样都获得正收益的事件不到一半。不同动作可能近似等价，因此动作不一致本身不是失败证明；这里必须结合仍达25%–26.5%的退化比例和明显不同的实际收益一起判断。','','## 信息质量与集合条件变化','', '|seed|方法|候选β MAE|候选β有符号偏差|交换Spearman均值|正收益动作召回|收益符号准确率|','|---|---|---:|---:|---:|---:|---:|']
 for r in qq:
  lines.append(f"|{r['seed']}|{r['method']}|{float(r['beta_mae']):.4f}|{float(r['beta_signed_bias']):.4f}|{float(r['gain_spearman']):.4f}|{float(r['positive_recall']):.2%}|{float(r['sign_accuracy']):.2%}|")
 lines+=['','β有符号偏差定义为估计−真值。不能因为MAE降低就判定集合收益有效。全候选动作的相关/召回与实际选中动作质量分开统计。','', '|seed|方法|汇总双重中心化相关|逐事件交互相关中位数|符号翻转召回|','|---|---|---:|---:|---:|']
 for r in co:lines.append(f"|{r['seed']}|{r['method']}|{float(r['doublecentered']):.4f}|{float(r['median_event_interaction']):.4f}|{float(r['sign_flip_recall']):.2%}|")
 lines+=['','反事实诊断沿用E25/E30五状态共同合法面板，继续使用初始48页的相同探测信息，不刷新候选池、不再增加探测。预测分数仅为离线量纲比较除以真实Dense正能量常数；该常数不进入Reader或采样，也不影响排序/no-op。不同事件尺度影响汇总相关，故同时报告逐事件相关。','', '## 新增成本','', '|b|每页Key|每head–tile Key|逻辑Key读取/head–tile|相对41页精算KV|相对21页精算KV|每层新增QK|每层逻辑Key读取|','|---|---:|---:|---:|---:|---:|---:|---:|']
 costs=read('cost_estimate.csv')
 for b in BS:
  x=next(r for r in costs if int(r['b'])==b and int(r['exact_pages'])==41);y=next(r for r in costs if int(r['b'])==b and int(r['exact_pages'])==21);lines.append(f"|{b}|{4*b}|{x['key_tokens_per_headtile']}|{int(x['logical_key_bytes_per_headtile'])/1024:.0f} KiB|{float(x['extra_relative_exact_KV']):.2%}|{float(y['extra_relative_exact_KV']):.2%}|{float(x['extra_QK_GFLOPs_layer']):.3f} GFLOPs|{float(x['logical_key_MiB_per_layer']):.1f} MiB|")
 lines+=['','每层按12 heads×72 tiles、M16、D128、BF16计数；无新增V读取或AV乘加、无新永久缓存。表只计算新增探测，已有摘要构建、Reader、Risk、Set/Independent评分仍有成本。非连续Gather可能放大HBM事务且增加同步/kernel开销，未测部署延迟，不能声称加速。最终精算页数量仍为21/41，但实际还读取了表中的Key，未假设与最终精算复用。五采样计划为重复实验，不计作每次推理五倍探测。','', '## 审计与可复现性','', f"独立审计检查全部{sum(r['decisions'] for r in audit['counts']):,}个Set/Independent动作的合法性、count和own；重建288份标签无关采样计划与SHA；{len(audit['probe_checks'])}次NumPy longdouble采样质量复算；{len(audit['checks'])}个选中动作独立真实混合NMSE复算与Set正质量混合评分核对。完整64Key仅用作少量数值恒等检查，不是新增实验配置。",'', 'E29/C/phi模型SHA保持不变；三个对照逐事件重放一致。离线读取完整KV用于冻结摘要和真值重建，新增探测函数仅接收Q、抽中K、摘要logZ和b，不接收V、Dense或真实页贡献。Test未读取，无训练。','', '原始数据在本目录 s17/s29：panel_*.npz含候选β、初始与共同面板预测/真实收益；plans包含token身份哈希与嵌套位置。精简包只包含报告、表格、协议、审计和复现源码，不包含原始QKV、模型或大型面板缓存。','', '## 判定边界','', '该实验检验的是固定预测局部响应下补充候选质量信息的价值，不能证明β是唯一瓶颈。Candidate-True/Full-True仍使用有误差的预测响应，本身不是完整真实残差Oracle；采样偶尔超过它们不意味着更准确。若信息收益改善但退化仍多，应按预设停止条件暂停扩展与Kernel，不通过增加b接近完整精算来获得“成功”。两条开发轨迹不足以声明全新场景泛化。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
 # Scientific cost-quality figure, not an online latency chart.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,3.8))
 for ax,seed in zip(axes,[17,29]):
  xx=[2*864*16*(48*4*b)*128/1e9 for b in BS];yy=[100*ss[seed,f'KeyProbe-b{b}','Set']['R_executed'] for b in BS];ax.plot(xx,yy,'o-',label='KeyProbe Set');ax.axhline(100*ss[seed,'E29','Set']['R_executed'],color='gray',linestyle='--',label='E29');ax.axhline(100*ss[seed,'Candidate-True','Set']['R_executed'],color='green',linestyle=':',label='Candidate true beta');ax.set(title=f'C seed {seed}',xlabel='Additional QK GFLOPs/layer',ylabel='Oracle gain recovery (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(DEST/'cost_quality.png',dpi=180);plt.close(fig)
 # Package self-contained review files, with a verified inventory.
 names=['REPORT.md','protocol.json','gates.json','summary.json','independent_audit.json','cost_quality.png']+[p.name for p in DEST.glob('*.csv')];entries={name:DEST/name for name in names};entries['plans_manifest.json']=DEST/'plans/manifest.json'
 for seed in [17,29]:entries[f's{seed}_complete.json']=DEST/f's{seed}/complete.json'
 for p in (ROOT/'local_experiments/e32_candidate_key_probe').glob('*'):
  if p.is_file():entries['source/'+p.name]=p
 inventory={name:sha(path) for name,path in entries.items()};dump(DEST/'review_inventory.json',inventory);entries['review_inventory.json']=DEST/'review_inventory.json'
 with tarfile.open(DEST/'e32_review.tar','w') as tar:
  for name,path in entries.items():tar.add(path,arcname=name)
 with tarfile.open(DEST/'e32_review.tar') as tar:
  for name,h in inventory.items():assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==h
 dump(DEST/'package_audit.json',dict(files=len(entries),bytes=(DEST/'e32_review.tar').stat().st_size,sha256=sha(DEST/'e32_review.tar'),all_inventory_hashes_verified=True));print('FINISHED',len(entries),gates['information_pass'],gates['full_pass'])
if __name__=='__main__':main()
