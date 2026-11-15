import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .core import OUT

def csvwrite(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def run():
 results=[json.loads(p.read_text()) for p in sorted((OUT/'events').glob('*/result.json'))]
 audit=json.loads((OUT/'audit.json').read_text());assert audit['passed']
 tables=OUT/'tables';tables.mkdir(exist_ok=True)
 a=[];states=[];thresholds=[];directions=[]
 for r in results:
  ident={k:r[k] for k in ['event_id','split']}
  a.append(ident|{k:v for k,v in r['A'].items() if not isinstance(v,list)})
  states.extend(ident|x for x in r['A']['state_details'])
  thresholds.extend(ident|x for x in r['A']['thresholds'])
  directions.append(ident|r['B_direction'])
 for name,rows in [('A_events',a),('A_states',states),('A_thresholds',thresholds),('B_directions',directions),('B_fullspace',audit['fullspace_and_panel_scope'])]:csvwrite(tables/(name+'.csv'),rows)
 reg=json.loads((OUT/'regression_summary.json').read_text());csvwrite(tables/'B_regression.csv',reg)
 ranks=json.loads((OUT/'ranking_per_event.json').read_text());csvwrite(tables/'B_ranking_events.csv',ranks)
 summary=[]
 for sp,pa,mo in sorted({(r['split'],r['panel'],r['model']) for r in ranks}):
  rr=[r for r in ranks if (r['split'],r['panel'],r['model'])==(sp,pa,mo)]
  row=dict(split=sp,panel=pa,model=mo,events=len(rr))
  for k in rr[0]:
   if isinstance(rr[0][k],(int,float)):
    row[k+'_mean']=float(np.mean([r[k] for r in rr]));row[k+'_median']=float(np.median([r[k] for r in rr]))
  summary.append(row)
 csvwrite(tables/'B_ranking_summary.csv',summary)
 ev=[r for r in a if r['split']=='evaluation'];full=[r for r in audit['fullspace_and_panel_scope'] if r['split']=='evaluation']
 fs=np.mean([r['negative_H_fullspace_cosine'] for r in full]);fc=np.mean([r['candidate_fullspace_cosine'] for r in full])
 text=f'''# E14：集合条件决策价值与可观测性

完成范围：复用144个 head–tile、994个去重反事实集合。训练36例/249集合，开发36例/251集合，评价72例/494集合。没有新增视频、QKV、全局优化、神经 Router 或多轮推理。保持 Q64/Page64、R4/rank8、10%预算与 own 保护。在线输入16 probes，真实目标评价全部64 Query。chunk2为216页/21精确页；chunk5/8为410页/41精确页。

## 主要判断

**集合条件信息存在实际决策价值，但在本次共同合法候选面板上的额外价值有限；摘要具有部分方向信号，却尚未转化为更好的动作选择。** 不能从本实验推断集合机制无价值，也不能断言当前摘要完全不可观测。

## A：条件信息的决策价值

同一 head–tile 内所有集合共享固定合法候选页面对，最多16×32；不是每个状态各自重新生成的 Medium 池，也不是全页面动作空间。条件 Oracle 是该面板内最佳单次动作，不是全局最优集合。

按请求选固定动作 j_mean=argmax mean(g)，再对各状态用真实收益拒绝负动作。因此此对照的排序固定，但执行门控使用完美条件信息。另列严格固定动作/no-op，以及同样完美门控下最强固定偏好 j_plus=argmax mean(max(0,g))，避免混淆排序和接受决策。

评价集结果以 head–tile 等权、例内集合等权汇总：

| 指标 | 结果 |
|---|---:|
| 条件 Oracle 相对请求的固定偏好：平均绝对 NMSE 增益 | 4.4528e-5 |
| 相对严格固定动作/no-op：平均绝对增益 | 5.1492e-5 |
| 相对最强固定偏好＋完美门控：平均绝对增益 | 4.3928e-5 |
| 逐状态相对 NMSE 额外改善，均值／中位数（先例内平均） | 0.5427／0.2451个百分点 |
| 每例 V/平均状态 NMSE，均值／中位数 | 0.3121%／0.1504% |
| 每例 V/条件 Oracle 收益，均值／中位数 | 12.21%／6.49% |
| 汇总绝对 V / 汇总条件 Oracle 收益 | 7.65% |
| 最佳动作相对初始 Risk 状态改变频率 | 55.30% |
| 改变动作且收益差达到当前 NMSE 的1% | 21.74% |
| 条件额外优势达到当前 NMSE 的1%的状态比例 | 14.68% |
| 有意义正负翻转（双方均超过1% NMSE门槛） | 4.13% |
| 有意义排序翻转／全部非平局动作对 | 2.10% |

不同归一化回答不同问题，不能相互替换。完整0.1%/1%/5%门槛及逐例结果见 tables/A_*.csv。频繁的最佳动作变化并不等于大幅 NMSE 改善。固定 Oracle 已读取反事实标签，不是在线可实现的页面排序器。

## B：方向信号与预测能力

H 是已选页面的摘要质量加权 Value 方向；真实 E 是未选页面的残差和。全部页面的摘要中心化方向和为零，因此同时报告 H 和 −H，不能将负相关简单判作无信息，也不能把 H 称为预测残差。

| 评价集方向指标 | 均值 |
|---|---:|
| rank8坐标内 cos(H,E) | −0.4425 |
| rank8坐标内 cos(−H,E) | +0.4425 |
| 候选方向与真实残差变化的 rank8 余弦 | +0.5383 |
| 集合绝对余弦／随机公共旋转参照 | 0.5376／0.2969 |
| 候选绝对余弦／随机公共旋转参照 | 0.5864／0.2956 |
| 解码回128维后 cos(−H,E) | {fs:.4f} |
| 解码回128维后候选余弦 | {fc:.4f} |
| 当前 basis 保留的真实集合／候选残差能量 | 14.41%／14.00% |

先逐 Query 计算再汇总；不把不同 Query 残差相加。坐标相关性不能冒充完整残差重建能力。14%是现有 basis 的能量覆盖，并非所有缓存信息的上限；缓存仍含完整 V 均值。

对同一页面对的跨集合收益差训练无截距 ridge 探针：集合标量、既有12维上下文、5维点积、68维双线性。使用全部无序状态对；按 head–tile 等权。仅训练集拟合归一化，正则在开发集选定，评价前冻结。另将每个状态对的候选均值去掉，专门评价影响排序的交互变化。

| 特征 | 完整收益变化 R²（对零预测） | 纯排序交互 R² |
|---|---:|---:|
| 不使用集合（零变化） | 0 | 0 |
| 集合标量 | 0.0365 | 0 |
| 既有上下文 | 0.1282 | 0.0869 |
| 点积 | 0.0862 | 0.0476 |
| 完整双线性 | 0.0473 | −0.0080 |

收益差用每例标签标准差归一化；该尺度只用于监督和评价，绝不作为推理输入。这些是解释能力诊断，不是已校准的绝对收益预测器。不能把这里的 R² 与 E13 的交互方差占比直接相乘解释为决策收益。

## 实际选动作：解释能力没有转为收益

使用共享的在线线性页面对评分，再追加条件变化预测。下表用真实标签事后拒绝负动作，仅作排序诊断，**不是可部署门控结果**。

| 模型 | 评价集事后正收益过滤后的平均相对改善 |
|---|---:|
| Pair-only 线性探针 | 0.9551% |
| ＋完整变化上下文 | 0.9369% |
| ＋纯交互上下文 | 0.9400% |
| ＋纯交互点积 | 0.8765% |
| ＋纯交互双线性 | 0.9341% |

集合标量不改变动作内排序；其余已注册探针均未提高这个指标。额外用真实初始收益作为固定排序锚点的离线面板，基线为4.0495%，上下文纯交互为3.9819%，也没有改善。

强制执行最高分动作时存在严重退化：在线 Pair-only 平均−2.5447%，若干双线性/点积方案达到约−431%。这些相对均值受低初始 NMSE 状态影响，不能省略；CSV同时保存绝对 NMSE 增益、逐例数据与中位数。本轮没有训练执行置信度，也没有通过阈值选择回避这些失败。

## 审计、边界与下一步

审计144例通过，抽检合法动作与原 FP64 残差重算最大差0；在线特征独立重建一致，禁止教师字段污染不影响特征；训练正规方程残差通过，冻结模型哈希通过，原始734个文件和E13的288个状态/面板文件哈希不变。详见 audit.json。

反事实集合来自初始 Risk、局部 Exchange 第1/2/4步及合法扰动，并非新独立 QKV。只有36个独立训练 head–tile、同一条训练基础轨迹；评价场景此前已使用，不能称未见场景泛化。本轮既没有全动作空间结论，也没有视频质量或统计显著性结论。

当前证据支持：条件排序有额外价值，但不能用交互方差夸大它；现有方向代理有部分可观测信号，不能直接宣布应放弃；简单线性/双线性条件校正尚无动作收益。暂不升级 Router 或修改 KV 摘要结构。若继续，应先定位哪些状态有实质条件收益，并研究现有可观测信号为何不能识别这些动作，而非仅增加网络规模。

## 文件与复现

- protocol.json：预登记定义与门槛；model_lock.json：开发选择、冻结权重哈希。
- tables/：逐例、逐状态、门槛、方向、回归和动作表。
- models/：全部线性探针；events/：在线特征缓存和标签；predictions/及selected_actions.json：完整预测与动作记录（精简包不重复收录大缓存）。
- 源码：local_experiments/e14_value_observability/。
- 在仓库根目录、原 set_oracle_v1 和 E13 数据可用时，依次执行 `python -m local_experiments.e14_value_observability.{{prepare,fit,evaluate,audit,report}}`（花括号表示依次运行五个模块，不是单条 Python 模块名）。设置 OPENBLAS_NUM_THREADS=4、OMP_NUM_THREADS=4。重新拟合会覆盖本地模型，应使用新 E14_ROOT 保存复现实验；基线输入目录不修改。
'''
 (OUT/'REPORT.md').write_text(text)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 ax[0].hist([100*r['V_statewise_relative'] for r in ev],bins=18);ax[0].set(xlabel='Additional NMSE improvement (percentage points)',ylabel='Head-tiles',title='Conditional oracle vs fixed preference + perfect gate')
 rr=[r for r in reg if r['split']=='evaluation' and r['mode']=='interaction_only' and r['model']!='pair_only'];ax[1].bar([r['model'].split('__')[-1] for r in rr],[r['R2_vs_zero'] for r in rr]);ax[1].axhline(0,color='black',lw=.5);ax[1].set(ylabel='Interaction R²',title='Held-out explanation of gain changes');fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)
 package=OUT/'review';package.mkdir(exist_ok=True)
 for name in ['REPORT.md','overview.png','protocol.json','preparation.json','audit.json','model_lock.json','development_trials.json','regression_summary.json'] :shutil.copy2(OUT/name,package/name)
 for name in ['tables','models']:shutil.copytree(OUT/name,package/name,dirs_exist_ok=True)
 root=Path(__file__).resolve().parents[2]
 for name in ['e14_value_observability','e13_counterfactual','e12_swap_prediction','set_oracle_v1']:
  dest=package/'source'/'local_experiments'/name;dest.mkdir(parents=True,exist_ok=True)
  for p in (root/'local_experiments'/name).glob('*.py'):shutil.copy2(p,dest/p.name)
 from ..set_oracle_v1.common import sha
 hashes={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p.name!='checksums.json'}
 (package/'checksums.json').write_text(json.dumps(hashes,indent=2))
 target=OUT.parent/'e14_value_observability_review.tar.gz'
 with tarfile.open(target,'w:gz') as tar:tar.add(package,arcname='e14_value_observability_review')
 print(dict(report=str(OUT/'REPORT.md'),package=str(target),bytes=target.stat().st_size,fullspace_set=fs,fullspace_candidate=fc))
if __name__=='__main__':run()
