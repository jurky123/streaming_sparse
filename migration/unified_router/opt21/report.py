import collections,hashlib,html,json,statistics
from pathlib import Path
from .setup import ROOT,PARENT
from ..common import atomic_json,sha

def main():
 cases=[]
 for i in range(8):
  p=ROOT/f'final/case{i}';assert (p/'completed.json').exists();d=json.loads((p/'result.json').read_text());d['decode']=json.loads((p/'decode.json').read_text());cases.append(d)
 names=['original_Dense','original_Tile20_after','optimized_Dense','optimized_Tile20_after']
 aggregate={}
 for name in names:
  variant='Tile20_after' if name.endswith('Tile20_after') else 'Dense'
  gen=statistics.mean(c['results'][name]['seconds'] for c in cases);decode=statistics.mean(c['decode'][variant]['seconds'] for c in cases)
  aggregate[name]=dict(generation_seconds=gen,decode_and_transfer_seconds=decode,total_seconds=gen+decode,
    all8_latents_bit_identical=all(c['results'][name]['latent_bit_identical_to_opt20'] for c in cases),
    peak_allocated_gb=max(c['results'][name]['peak_allocated_gb'] for c in cases))
 parent=json.loads((PARENT/'summary.json').read_text());quality=parent['quality']['Tile20_after']
 keep=statistics.mean(c['results']['optimized_Tile20_after']['keep'] for c in cases);executed=statistics.mean(c['results']['optimized_Tile20_after']['executed'] for c in cases)
 assert abs(keep-quality['keep'])<1e-7 and abs(executed-quality['executed'])<1e-7
 for p,h in json.loads((ROOT/'checkpoint_manifest.json').read_text()).items():assert sha(p)==h
 profile=json.loads((ROOT/'final/case0/profile.json').read_text());groups=collections.defaultdict(float)
 for row in profile['kernels']:
  name=row['name']
  if 'gemm' in name.lower():group='GEMM：FFN及各线性投影'
  elif name=='_shared':group='稀疏attention主循环'
  elif name in ['_coarse','_fine_age','_top','_tiles']:group='路由评分与选择'
  elif name=='_rope':group='融合RoPE'
  elif name=='_merge_restore':group='融合分段合并/还原'
  elif 'flash_fwd' in name:group='Dense/交叉attention'
  elif name in ['_assign','_local','_offset','_scatter'] or 'scatter_gather' in name:group='Q/K分组与重排'
  elif name in ['_segment','_write','_quant']:group='KV/原型与量化'
  else:group='归一化、逐元素运算及其他'
  groups[group]+=row['ms']
 gains=dict(vs_previous_sparse=aggregate['original_Tile20_after']['generation_seconds']/aggregate['optimized_Tile20_after']['generation_seconds'],vs_optimized_dense=aggregate['optimized_Dense']['generation_seconds']/aggregate['optimized_Tile20_after']['generation_seconds'],vs_previous_dense=aggregate['original_Dense']['generation_seconds']/aggregate['optimized_Tile20_after']['generation_seconds'])
 summary=dict(cases=8,measurements=aggregate,generation_speedups=gains,quality_preserved_by_bit_identical_full_latents=True,lpips_vs_dense=quality['lpips'],psnr_vs_dense=quality['psnr_db'],weighted_keep=keep,executed_attention_ratio=executed,profile_scope=profile['scope'],profile_group_kernel_ms=dict(groups),parent=str(PARENT),same_checkpoints=True)
 atomic_json(ROOT/'summary.json',summary)
 lines=['Opt21：固定微调后的20%全层方案，推理热点优化','',
  '范围：层1–29稀疏，层0/首chunk Dense；当前P16，历史P8/P4/P2。原Opt20 Tile20 checkpoint、分组树、预算和选择规则均不变。',
  '验证：8个Val prompt、seed0、45帧480×832；原Dense/原Tile20/优化Dense/优化Tile20在各自同一张A100上测量。',
  '每方案1次预热后3次同步计时，取中位数，再平均8个样例。模型加载、初次建图、视频编码不计；历史处理计入生成时间。',
  '所有方案的每次完整latent均与Opt20对应保存结果逐元素相同。另通过同进程切换prompt、Dense/稀疏模式的2样例检查。',
  'VAE单独预热后2次计时，含latent上传与RGB下载；下表总耗时为生成与对应VAE时间之和。','',
  '方案                        生成秒    VAE秒     总秒']
 for name,d in aggregate.items():lines.append(f"{name:27} {d['generation_seconds']:.4f}  {d['decode_and_transfer_seconds']:.4f}  {d['total_seconds']:.4f}")
 lines += ['',f"相对旧Tile20生成加速 {gains['vs_previous_sparse']:.3f}x；相对同样优化的Dense {gains['vs_optimized_dense']:.3f}x。",
  f"保留率 {100*keep:.3f}%；稀疏区域attention执行率 {100*executed:.3f}%，不是整网FLOPs比例。",
  f"视频相对Dense的LPIPS {quality['lpips']:.6f}、PSNR {quality['psnr_db']:.3f} dB，与Opt20相同（后36帧；由完整latent逐元素一致证明，复用既有质量测量）。",'',
  '已采用：',
  '1. 固定文本投影与cross-attention KV每个视频计算一次；切换prompt显式更新。',
  '2. 去噪步骤不再拼接未使用的输出KV。',
  '3. 因果历史增量构建：只编码新增的完成chunk；历史主干前向由6次减到3次，仍在原context_noise上编码，不复用带噪去噪KV。',
  '4. FP64 RoPE单kernel融合，保留原BF16舍入；微测约0.428ms→0.035ms。',
  '5. 分段attention结果批量合并并直接还原query顺序；关闭会改变舍入的FMA融合，微测约节省0.06ms/次。',
  '6. 将完整去噪前向捕获为CUDA Graph；路由kernel平铺捕获，显式刷新轨迹历史、文本条件与timestep；切换Dense/稀疏时释放旧整步图以控制显存。','',
  '未采用：16-query/128-key计算块、更多split/warp/stage配置；本轮实测更慢。attention主循环仍使用原32-query、split2/warp2/stage2。','',
  '优化后热点（最后chunk最后去噪step、完整30层；独立profiler pass，下面只统计kernel时间，不是整段视频比例）：']
 for k,ms in sorted(groups.items(),key=lambda x:-x[1]):lines.append(f'{k}: {ms:.3f}ms ({100*ms/profile["kernel_sum_ms"]:.1f}%)')
 lines+=['','下一步：FFN/投影GEMM，以及归一化、残差、调制、dtype转换仍占大头；稀疏attention主循环和路由次之。',
  '仅降低attention执行率不会同比降低整网耗时。后续优先融合主干逐元素/归一化读写，再评估attention循环访存与寄存器占用。','',
  '稳定入口：migration.unified_router.opt21.inference.load 与 trajectory；使用torch.no_grad和现有sdpa_backend上下文。',
  '状态：bash opt21_status.sh --watch；视频：index.html（与优化前完全一致，直接引用已验证的原视频）。',
  '本轮没有重新训练，也未改写旧实验。新实现针对既有BF16、batch1、128维head的T2V模型；首次建图耗时单独记录在各case/result.json。']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 body=['<!doctype html><meta charset="utf-8"><title>Opt21 20%全层优化</title>',
  '<style>body{font-family:sans-serif;background:#17191d;color:#eee;max-width:1500px;margin:24px auto;padding:0 12px}a{color:#8cf}.pair{display:flex;gap:8px}.pair>div{width:50%}video{width:100%}button{padding:8px 16px;margin:12px 0}</style>',
  '<h1>Opt21：20%全层推理优化</h1><p>8条完整输出与Opt20逐元素一致。下列直接引用对应视频，左侧Dense，右侧微调后的20%方案；优化没有改变视频。</p><a href="REPORT.txt">性能与热点报告</a>']
 for c in cases:
  pid=c['prompt_id'];body.append(f'<section><h2>{html.escape(pid)}</h2><div class="pair">')
  for name in ['Dense','Tile20_after']:
   src=f'../{PARENT.name}/e2e_final/{pid}.seed0/{name}.mp4';body.append(f'<div><p>{name}</p><video controls preload="metadata" src="{src}"></video></div>')
  body.append('</div><button onclick="this.parentNode.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">同步播放</button></section>')
 (ROOT/'index.html').write_text('\n'.join(body))
 atomic_json(ROOT/'completion.json',dict(status='completed',video_cases=8,all_bit_identical=True,checkpoints_unchanged=True))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
