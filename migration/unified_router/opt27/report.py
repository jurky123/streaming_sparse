"""Report rejected refinement fusions and adopted exact preparation fusions."""
import json,statistics,hashlib
from pathlib import Path
from .setup import ROOT

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 shapes=['480x832_chunks4','720x1280_chunks4','720x1280_chunks7','960x1664_chunks7'];modules={};rejected={}
 for shape in shapes:
  blocks=[read(ROOT/'confirm'/shape/f'layer{layer:02d}.json') for layer in [1,14,24,29]];assert all(d['completed'] and all(d['checks'].values()) and d['rows'][0]['exact'] for d in blocks)
  old=statistics.mean(d['rows'][0]['base_ms'] for d in blocks);new=statistics.mean(d['rows'][0]['ms'] for d in blocks)
  modules[shape]=dict(before_ms=old,after_ms=new,speedup=old/new,all_intermediates_and_attention_exact=True)
 for family in ['scan','mass','shared']:
  rejected[family]={}
  for p in (ROOT/family).glob('*/*.json'):
   data=read(p);assert data['completed'];valid=[r for r in data['rows'] if 'ms' in r];best=min(valid,key=lambda r:r['ms'])
   rejected[family][p.parent.name]=dict(configurations=len(data['rows']),errors=sum('error' in r for r in data['rows']),best=best)
 cases=[]
 for i in range(8):
  p=ROOT/'e2e/720x1280'/f'case{i}';assert read(p/'completed.json')['all_full_latents_bit_exact'];cases.append(read(p/'timing.json'))
 times={k:statistics.mean(c['results'][k]['seconds'] for c in cases) for k in ['Dense','Route26','Prep27']}
 assert read(ROOT/'e2e/480x832/case0/completed.json')['all_full_latents_bit_exact']
 keep=statistics.mean(c['results']['Prep27']['keep']for c in cases);execution=statistics.mean(c['results']['Prep27']['execution']for c in cases)
 profile=read(ROOT/'e2e/720x1280/case0/profile.json');groups={}
 for r in profile['kernels']:
  name=r['name'];label='other'
  if 'gemm' in name.lower():label='GEMM'
  elif 'qk_int_sv_f16_block_sparse_attn_kernel' in name:label='sparse_attention'
  elif name in ['_fine_age','_coarse_grouped','_tiles','_top']:label='router'
  elif name in ['_q','_kv']:label='fused_Q_KV_preparation'
  groups[label]=groups.get(label,0)+r['ms']
 result=dict(completed=True,scope='A100-SXM4-40GB;720x1280,45 frames,8 Val prompts,seed0;median5 warmed full rollouts/case then mean8cases. Same GPU and inputs for all3 variants per case; order rotated. Generation includes history encoding/preparation,router,whole network,scheduler and latent transfers; excludes VAE,loading,compilation,graph capture,video encoding. First chunk/layer0 Dense,remaining29layers sparse,Q32K64,20% budget,currentP16/historyP8P4P2,frozen weights.',modules=modules,rejected_refinement_fusions=rejected,video_seconds=times,speedup_vs_dense=times['Dense']/times['Prep27'],speedup_vs_previous=times['Route26']/times['Prep27'],time_reduction_vs_previous=1-times['Prep27']/times['Route26'],all_video_latents_bit_exact=True,keep=keep,execution=execution,profile_groups_ms=groups,profile_scope=profile['scope'],cases=cases)
 write(ROOT/'summary.json',result)
 lines=['Opt27：精筛融合探索与精确 Q/KV 数据准备融合','',result['scope'],'','正式采用方案：',
 'Q：一次kernel完成按已有Q分组索引读取、FP16路由投影、INT8 attention量化及scale计算；不再写出完整重排BF16 Q。',
 '当前KV：一次kernel完成按已有K分组索引读取、写入分组KV、INT8 K量化、scale计算、填充页K及FP16 V打包。历史刷新保持原路径。',
 '保留原fine精筛、softmax与选页预算；不改变router训练目标和attention数学语义。',
 '按720p每chunk10800 token、12头、D128计算，融合消除约182MB/层/次的逻辑中间读写（Q约99.5MB，KV约82.9MB）；这是按张量大小估计的搬运量，不是实测HBM流量。','',
 '完整稀疏模块（含当前KV更新、Q分组、路由、Sparge attention与Q恢复；不含历史准备；四层均值）：']
 for shape,r in modules.items():lines.append(f'{shape:24} {r["before_ms"]:.4f} → {r["after_ms"]:.4f} ms；{r["speedup"]:.4f}x')
 lines+=['','720p完整45帧生成（不含VAE）：']
 for label,t in times.items():lines.append(f'{label:12} {t:.6f}s；相对Dense {times["Dense"]/t:.4f}x')
 lines += [f'相对Opt26再加速{result["speedup_vs_previous"]:.4f}x，耗时减少{result["time_reduction_vs_previous"]*100:.2f}%。',f'实际保留率{keep*100:.4f}%；执行率{execution*100:.4f}%。',
 '16个真实块逐项检查分组K/V、量化K和scale、打包K/V、prototype/count、Q投影/量化/scale，全部逐元素一致；attention输出也完全一致。',
 '8条720p及1条480p完整45帧latent与独立保存的上一版输出逐元素一致，每个方案5次计时重复也一致。本轮不需要训练补偿。',
 '这表示没有新增生成误差，不表示上一版稀疏生成与Dense无损。','',
 '精筛融合未采用（下面只含fine+最终选页，不含coarse或attention）：']
 for family,shapes0 in rejected.items():
  for shape,r in shapes0.items():
   best=r['best'];lines.append(f'{family:8} {shape:24} best {best["ms"]:.4f} ms，原{best["base_ms"]:.4f} ms，{best["speedup"]:.3f}x；{best["options"]}')
 lines += ['Triton candidate-loop把全部分数保留片上；prototype-loop与mass-loop按prototype流式累加；CUDA WMMA方案把分数存入shared memory，只写出页权重。',
 '这些实现虽减少或消除了大分数矩阵写出，但本轮实测均更慢，没有接入正式推理。串行循环、片上数据重排/同步及部分方案的额外prototype计算是实现层面的代价；未测得硬件带宽/occupancy计数，不将推测当作硬件profile结论。',
 '正式方案仍保留约123MB的720p末chunk精筛FP32分数。减少物化不自动等于更快；这轮收益来自消除Q/KV准备阶段重复读写和kernel调度。','',
 '最新profile：720p最后chunk step3，全30层GPU kernel时间加和（不是整段视频墙钟）：']
 for label,t in groups.items():lines.append(f'{label}: {t:.4f} ms')
 lines+=['','没有重新训练，原Opt25/26默认入口与58个checkpoint保持不变。可选入口：migration.unified_router.opt27.inference.load()；同原接口，仅支持单流四chunk/45帧。',
 '查看状态：./opt27_status.sh','复现：CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt27.evaluate --case 0']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 locked=read(ROOT.parent/'opt20_all29_tile_finetune_20261025/chosen_checkpoints.json');assert len(locked)==58 and all(sha(p)==s for p,s in locked.items())
 for name,key in [('opt25_720p_e2e_20261025','source_sha256'),('opt26_route_20261025','files')]:
  assert all(sha(p)==s for p,s in read(ROOT.parent/name/'FINAL_VERIFICATION.json')[key].items())
 write(ROOT/'FINAL_VERIFICATION.json',dict(completed=True,blocks_exact=16,video720_cases_exact=8,video480_cases_exact=1,all58_checkpoints_unchanged=True,baseline_sources_unchanged=True,files={str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*') if p.suffix in ['.py','.cu']}))
 print(json.dumps({k:result[k]for k in ['modules','video_seconds','speedup_vs_dense','speedup_vs_previous','profile_groups_ms']},indent=2))
if __name__=='__main__':main()
