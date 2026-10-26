import json,statistics,hashlib
from pathlib import Path
from .setup import ROOT

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 shapes=['480x832_chunks4','480x832_chunks7','720x1280_chunks4','720x1280_chunks7','960x1664_chunks4','960x1664_chunks7'];summary={};all_errors=[]
 for shape in shapes:
  summary[shape]={}
  for phase in ['confirm','module','module_v2']:
   blocks=[read(ROOT/phase/shape/f'layer{layer:02d}.json') for layer in [1,14,24,29]];assert all(d['completed'] for d in blocks)
   assert all('error' not in r for d in blocks for r in d['rows'])
   options={tuple(r['options'][key]for key in ['q','k','warp_q'])for d in blocks for r in d['rows']};rows=[]
   for g,k,w in sorted(options):
    vals=[r for d in blocks for r in d['rows'] if tuple(r['options'][key]for key in ['q','k','warp_q'])==(g,k,w)];assert len(vals)==4
    row=dict(q=g,k=k,warp_q=w,warps=g//w)
    for key in ['ms','baseline_ms','dense_ms','nmse_vs_dense','nmse_vs_q32','baseline_nmse_vs_dense']:
     row[key]=statistics.mean(r[key] for r in vals)
    row.update(speedup=row['baseline_ms']/row['ms'],time_vs_dense=row['ms']/row['dense_ms'],reference_max_nmse=max(r['reference_max_nmse']for r in vals));all_errors.append(row['reference_max_nmse'])
    if phase=='confirm':
     for key in ['true_keep','execution']:row[key]=statistics.mean(r[key]for r in vals)
    else:
     assert all(all(r['checks'].values())for r in vals)
     row['components_ms']={key:statistics.mean(r['components_ms'][key] for r in vals)for key in vals[0]['components_ms']}
    rows.append(row)
   summary[shape][phase]=rows
  scan=[r for r in read(ROOT/'k64_final'/shape/'layer24.json')['rows'] if r['options']['k']==64]+read(ROOT/'k128'/shape/'layer24.json')['rows']
  assert len(scan)==22 and all('error'not in r for r in scan)
  summary[shape]['scan22_best']=min(scan,key=lambda r:r['ms'])
 selection=read(ROOT/'selection_verification.json');assert selection['completed'] and len(selection['rows'])==24 and all(len(r['checks'])==10 for r in selection['rows'])
 # Archive and report the failed exploratory configurations; never count as valid speed measurements.
 locked=read(ROOT.parent/'opt20_all29_tile_finetune_20261025/chosen_checkpoints.json');assert len(locked)==58 and all(sha(p)==s for p,s in locked.items())
 for name,key in [('opt25_720p_e2e_20261025','source_sha256'),('opt26_route_20261025','files'),('opt27_fused_refine_20261025','files')]:
  assert all(sha(p)==s for p,s in read(ROOT.parent/name/'FINAL_VERIFICATION.json')[key].items())
 adapter=read(ROOT/'adapter.json');assert adapter['completed'] and len(adapter['rows'])==72 and all(r['exact'] for r in adapter['rows'])
 result=dict(adapter=adapter,completed=True,scope='A100-SXM4-40GB; six resolutions/history shapes; real final-chunk step3 captured Q/K/V; layers1/14/24/29; 5 balanced timing rounds per block, medians then equal-weight mean4 layers. Sparse uses INT8 Q/K, FP16 P/V, BF16 output; Dense is BF16 FlashAttention. Fixed ~20% execution budget. Frozen existing Tile20 weights; no training or new full-video rollouts.',selected=dict(q=64,k=128,warp_q=16,warp_k=128,warps=4),shapes=summary,selection_checks=240,native_reference_max_nmse=max(all_errors),new_video_evaluations=0,training_updates=0,default_changed=False)
 write(ROOT/'summary.json',result)
 lines=['Opt28：先选高效硬件tile，再设计Query聚合','',result['scope'],'','结论：Q64×K128、4 warps（WARP_Q16，WARP_K128）作为后续算法/联合学习的首选硬件目标。',
 '这是本轮22个可用配置中的胜出者，不代表达到A100硬件极限；未取得Nsight硬件counter。Q64×K64保留作较低选页误差对照，Q128×K64次优，Q256不采用。',
 '复用现有Sparge SM80 kernel，扩展非因果tile和K128打包/尾页处理；未修改原Opt25/26/27默认代码或checkpoint。',
 'K128将同一chunk内相邻两个K64页合为物理页，不能跨chunk边界拼接。原型仍按K64逻辑页计算，尚未重新学习K128原型。','',
 '首选Q64×K128相对当前Q32×K64，四层平均：',
 '形状                       attn旧→新(ms)       attn加速  模块旧→新(ms)       模块加速  执行率  实际保留']
 for shape,d in summary.items():
  a=next(r for r in d['confirm'] if(r['q'],r['k'])==(64,128));m=next(r for r in d['module_v2']if(r['q'],r['k'])==(64,128))
  lines.append(f'{shape:25} {a["baseline_ms"]:7.3f}→{a["ms"]:7.3f} {a["speedup"]:7.3f}x {m["baseline_ms"]:7.3f}→{m["ms"]:7.3f} {m["speedup"]:7.3f}x {a["execution"]*100:6.3f}% {a["true_keep"]*100:6.3f}%')
 lines += ['','attn路径包含当前Q量化、页LUT、native attention和Q恢复；KV已打包。不等于纯kernel时间。',
 '完整模块包含当前KV分组/量化/原型准备、Q分组/投影/量化、粗筛、逐query精筛、选页适配、attention和恢复；历史准备排除。',
 '本轮无新视频端到端测量，不能把上述模块加速比当作整段视频加速比。上版Opt27的720p实测1.353x仍是最近完整生成记录。','',
 'attention路径相对BF16 Dense FlashAttention的时间比例：']
 for shape,d in summary.items():
  a=next(r for r in d['confirm'] if(r['q'],r['k'])==(64,128));lines.append(f'{shape:25} {a["ms"]:.4f}/{a["dense_ms"]:.4f}ms = {100*a["time_vs_dense"]:.2f}%')
 lines += ['稀疏用INT8 QK/FP16 PV，Dense用BF16；时间比例接近执行比例属于操作层面的结果，不能据此宣称两种精度下的硬件利用率相同或理论性能已完全兑现。','',
 '质量代价：每层NMSE独立归一后取四层均值，非视频指标，也不是所有层的均值。',
 '形状                       Q32K64       Q64K64       Q128K64      Q64K128']
 for shape,d in summary.items():
  rr=d['module_v2'];base=rr[0]['baseline_nmse_vs_dense'];vals=[next(r['nmse_vs_dense']for r in rr if(r['q'],r['k'])==key)for key in [(64,64),(128,64),(64,128)]]
  lines.append(f'{shape:25} {base:12.6f}'+''.join(f'{x:12.6f}'for x in vals))
 lines += ['直接合并Q/K页改变共享选页，并扩大部分量化scale范围；没有训练补偿，不能声称视频质量通过。',
 'Q64K128在本轮所有形状最快；Q64K64误差通常更低，但480p完整模块几乎没有收益。若后续联合学习无法修复K128选页损失，回到K64作质量折中。','',
 '首选方案完整模块各阶段（独立CUDA Graph计时，求和不一定等于整模块）：']
 for shape,d in summary.items():
  r=next(r for r in d['module_v2']if(r['q'],r['k'])==(64,128));lines.append(shape+' '+', '.join(f'{k}={v:.4f}ms'for k,v in r['components_ms'].items()))
 lines += ['','下一阶段算法设计：见QUERY_AGGREGATION_DESIGN.txt。',
 '1. 先做Q32子组聚合，保持原粗筛语义；再升级到整个Q64聚合，并统一候选预算。R4/R8/R16都计入实际GPU布局，不能仅按逻辑FLOPs宣称加速。',
 '2. 联合学习token→槽位权重A和Q分组，让选择相似的Q进入同一硬件tile。每步以当前Q重新计算加权query；复用的是A，不是Q、聚合query或旧选页。',
 '3. A按原始token位置缓存，重新排序后仍可正确索引；每步重算当前tile的槽位质量和混合权重。先验证fresh A，再训练/评估跨去噪步复用。',
 '4. 先按每个代表query做候选softmax，再按槽位质量混合概率；最终attention仍用真实QKV。主loss/best必须用实际attention输出误差，联合训练包含Q聚合/分组与KV路由，后续backbone微调也对齐部署前向。',
 '5. 保持第一chunk与第0层dense；其他层按现有历史距离衰减P8/P4/P2，当前逻辑K64页P16。缓存不能跨视频/chunk或泄漏未来step。','',
 '验证与失败记录：',
 '最终扫描22配置×6形状=132条有效结果；四层复核96条attention路径、72条最终完整模块，另有72条旧适配器模块对照。',
 f'独立量化attention参考（3头×首/中/末tile×3个Q位置）最大NMSE={max(all_errors):.8g}，门槛1e-4；所有完整模块Q投影、量化与scale逐元素匹配独立准备结果。',
 '独立PyTorch选页参考覆盖24块×10种Q/K几何，每种检查3头×首/中/末tile；索引有序、不越界、top-k阈值满足容差。',
 '初始K128把K方向拆成多个warp，未正确合并其结果，数值校验失败，已废弃。修正为WARP_K128，4warps沿Q分工。',
 '第二个问题是将radix histogram直接用于跨Q组归约后的张量时产生非法页号，长历史触发越界。先改为安全bitonic选择，再将合并分数写为小型连续矩阵，调用独立1D top-k（B>512用已验证radix，其余bitonic）。',
 '最终retile_v2与安全retile的72组选页完全一致，72个完整模块attention输出也完全一致；加入native前索引断言。错误尝试保存在invalid_inline_radix/，不计入速度结论。',
 '独立top-k增加少量分数物化，但避免在低效切片布局中排序；720p四chunk额外分数仅约2.76MB，960p七chunk约14.47MB。完整模块最终表包含这部分开销。',
 '初次完整模块编译漏设TORCH_CUDA_ARCH_LIST而尝试sm52，构建失败；统一设8.0后通过，源码基线未改动。',
 '现有58个checkpoint与Opt25/26/27已记录源码hash全部保持一致。新默认/训练更新/视频端到端评估次数均为0。','',
 '查看：./opt28_status.sh',
 '复现：TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 CUDA_VISIBLE_DEVICES=0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt28.module_probe --height 720 --width 1280 --chunks 4 --layer 24 --adapter-v2',
 '纯attention路径复核：同上替换module_probe为probe，将--adapter-v2替换为--confirm。独立选页校验：migration.unified_router.opt28.validate_selection。']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 write(ROOT/'FINAL_VERIFICATION.json',dict(completed=True,valid_scan_configurations=132,confirmed_attention=96,confirmed_modules=72,legacy_module_controls=72,adapter_exact_checks=72,selection_checks=240,max_reference_nmse=max(all_errors),all58_checkpoints_unchanged=True,baseline_sources_unchanged=True,training_updates=0,new_video_evaluations=0,default_changed=False,files={str(p.resolve()):sha(p)for p in Path(__file__).parent.glob('*')if p.suffix in ['.py','.cu','.cuh']}))
 print('\n'.join(lines[:21]))
if __name__=='__main__':main()
