"""Audit and summarize router optimization without changing reference results."""
import json,statistics,hashlib
from pathlib import Path
from .setup import ROOT

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 checks=read(ROOT/'radix_correctness.json');assert checks['passed'];assert read(ROOT/'verify/720x1280_chunks4/layer24.json')['previous_drift_regression_exact'];shapes=['480x832_chunks4','720x1280_chunks4','720x1280_chunks7','960x1664_chunks7'];operators={}
 for shape in shapes:
  rr=[read(ROOT/'verify'/shape/f'layer{layer:02d}.json') for layer in [1,14,24,29]];assert all(r['completed'] and r['exact'] for r in rr)
  before=statistics.mean(r['base_ms'] for r in rr);after=statistics.mean(r['ms'] for r in rr)
  operators[shape]=dict(before_ms=before,after_ms=after,speedup=before/after,selected_pages_exact=True,layers=[1,14,24,29])
 cases=[]
 for i in range(8):
  path=ROOT/'e2e/720x1280'/f'case{i}';done=read(path/'completed.json');assert done['completed'] and done['all_full_latents_bit_exact'];cases.append(read(path/'timing.json'))
 times={label:statistics.mean(r['results'][label]['seconds'] for r in cases) for label in ['Dense','Sparge32','Route26']}
 sparse=statistics.mean(r['results']['Route26']['keep'] for r in cases);execution=statistics.mean(r['results']['Route26']['execution'] for r in cases)
 pilot=read(ROOT/'e2e/480x832/case0/completed.json');assert pilot['all_full_latents_bit_exact']
 p=read(ROOT/'e2e/720x1280/case0/profile.json');groups={}
 for r in p['kernels']:
  name=r['name'];label='other'
  if 'gemm' in name.lower():label='GEMM'
  elif 'qk_int_sv_f16_block_sparse_attn_kernel' in name:label='sparse_attention'
  elif name in ['_coarse_grouped','_fine','_fine_age','_tiles','_top','_project']:label='router'
  groups[label]=groups.get(label,0)+r['ms']
 summary=dict(completed=True,scope='A100-SXM4-40GB, frozen Q32/K64,20% budget,first chunk/layer0 Dense,remaining29 layers sparse. Whole video720x1280,45 frames,8 prompts,seed0; median5 warmed replays then mean8cases. Each case compares all3 variants on same GPU; order rotated. Generation includes history preparation, model,router,scheduler and latent transfers; excludes VAE,loading,compilation,graph capture,encoding.',operators=operators,video_seconds=times,video_speedup_vs_dense=times['Dense']/times['Route26'],video_speedup_vs_previous=times['Sparge32']/times['Route26'],video_time_reduction_vs_previous=1-times['Route26']/times['Sparge32'],keep=sparse,execution=execution,video_all_bit_exact=True,profile_groups_ms=groups,profile_scope=p['scope'],radix_boundary_cases=len(checks['cases']),cases=cases)
 write(ROOT/'summary.json',summary)
 lines=['Opt26 路由优化结果','',summary['scope'],'','路由耗时（ms，四层平均；含投影、粗筛、top-k、精筛及最终选页）：']
 for shape,r in operators.items():lines.append(f'{shape:24} {r["before_ms"]:.4f} → {r["after_ms"]:.4f} ms；{r["speedup"]:.3f}x；选页完全一致')
 lines+=['','720p完整45帧生成（不含VAE）：']
 for label,t in times.items():lines.append(f'{label:12} {t:.6f}s；相对Dense {times["Dense"]/t:.4f}x')
 lines += [f'相对上一版Sparge32再加速 {summary["video_speedup_vs_previous"]:.4f}x，耗时减少 {summary["video_time_reduction_vs_previous"]*100:.2f}%。',f'实际保留率 {sparse*100:.4f}%；执行率 {execution*100:.4f}%。','',
 '实现：','1. 粗筛一次处理8个原Q32组，每组仍使用原来的8个代表query；共享prototype读取，将矩阵M维填满。仅执行调度改变，各组候选仍独立。',
 '2. 大候选集采用8-bit radix阈值选择，按IEEE浮点整数序定位精确第k大值；保持原索引顺序和并列值规则。小集合保留bitonic排序。',
 '3. 精筛保留原GP8/4warp kernel与FP32分数，较小最终候选集直接调用原选页kernel。',
 '4. 中间精筛分数矩阵仍存在（720p最后chunk约123MB）。转置布局和更少warp等配置未通过完整视频精确性条件，未用于部署。',
 '',f'正确性：16个真实块选页完全一致；{len(checks["cases"])}组radix边界测试通过；8条720p及1条480p完整视频与上一版latent逐元素完全一致。',
 '新路由的质量与上一版Sparge32完全一致；这不表示Sparge32与Dense无损。没有新增训练，也无需为本轮执行优化微调。',
 '审计：原Opt22/24/25源码及58个checkpoint SHA校验；原默认入口不改，可选入口migration.unified_router.opt26.inference.load()仅在进程内安装路由优化，480p/720p四chunk单流推理。',
 '实验未采用项：2warp精筛在case4第22层chunk2/step2出现1.9e-6分数变化，导致选页及闭环latent变化；case4/7未通过完整视频回归，恢复原精筛重跑。原记录保存在rejected_fine_schedule。GP4在P2时矩阵尺寸不满足Tensor Core要求；4-bit histogram原型一致性失败，正式路径只使用已验证8-bit实现。',
 '最初独立Torch参考对±0并列值的处理与原kernel不同，检查改为显式IEEE整数序，保留原语义。','',
 '最新profile（720p最后chunk step3，全30层；GPU kernel时间加和，不等于完整视频墙钟）：']
 for label,t in groups.items():lines.append(f'{label}: {t:.4f} ms')
 lines+=['','路由内部的新热点是精筛及其FP32分数写出/softmax读取。内存带宽占比尚无硬件计数器测量，不能仅凭张量大小认定全部耗时来自物化；下一步可比较融合精筛与归约、局部重算的收益。',
 '查看状态：./opt26_status.sh','复现：CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt26.evaluate --case 0']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 locked=read(ROOT.parent/'opt20_all29_tile_finetune_20261025/chosen_checkpoints.json');assert len(locked)==58 and all(sha(p)==s for p,s in locked.items())
 for name,key in [('opt22_tile20_backbone_fusion_20261025','files'),('opt24_attention_backends_20261025','files'),('opt25_720p_e2e_20261025','source_sha256')]:
  v=read(ROOT.parent/name/'FINAL_VERIFICATION.json');assert all(sha(p)==s for p,s in v[key].items())
 write(ROOT/'FINAL_VERIFICATION.json',dict(completed=True,real_blocks_exact=16,video720_cases_exact=8,video480_cases_exact=1,radix_tests_passed=len(checks['cases']),previous_drift_regression_exact=True,all58_checkpoints_unchanged=True,baseline_sources_unchanged=True,files={str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py')}))
 print(json.dumps({k:summary[k] for k in ['operators','video_seconds','video_speedup_vs_dense','video_speedup_vs_previous','profile_groups_ms']},indent=2))
if __name__=='__main__':main()
