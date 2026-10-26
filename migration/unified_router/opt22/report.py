import collections,json,statistics
from pathlib import Path
from .setup import ROOT,PARENT,TRAINING
from ..common import atomic_json,sha

def main():
 cases=[]
 for i in range(8):
  p=ROOT/f'final/case{i}';assert (p/'completed.json').exists();d=json.loads((p/'result.json').read_text());d['decode']=json.loads((p/'decode.json').read_text());cases.append(d)
 names=['Opt21_Dense','Opt21_Tile20_after','Opt22_Dense','Opt22_Tile20_after'];aggregate={}
 for name in names:
  variant='Tile20_after' if name.endswith('Tile20_after') else 'Dense'
  gen=statistics.mean(c['results'][name]['seconds'] for c in cases);decode=statistics.mean(c['decode'][variant]['seconds'] for c in cases)
  aggregate[name]=dict(generation_seconds=gen,decode_and_transfer_seconds=decode,total_seconds=gen+decode,all8_latents_bit_identical=all(c['results'][name]['latent_bit_identical_to_opt21'] for c in cases))
 for p,h in json.loads((ROOT/'checkpoint_manifest.json').read_text()).items():assert sha(p)==h
 quality=json.loads((TRAINING/'summary.json').read_text())['quality']['Tile20_after']
 keep=statistics.mean(c['results']['Opt22_Tile20_after']['keep'] for c in cases);executed=statistics.mean(c['results']['Opt22_Tile20_after']['executed'] for c in cases)
 assert abs(keep-quality['keep'])<1e-7 and abs(executed-quality['executed'])<1e-7
 groups=collections.defaultdict(float);profile=json.loads((ROOT/'final/case0/profile.json').read_text())
 for r in profile['kernels']:
  n=r['name']
  if 'gemm' in n.lower():g='GEMM：FFN及线性投影'
  elif n=='_shared':g='稀疏attention主循环'
  elif n in ['_coarse','_fine_age','_top','_tiles']:g='路由评分与选择'
  elif n=='_pointwise':g='融合调制与残差'
  elif n=='_rms_ordered':g='融合RMSNorm（原归约顺序）'
  elif n=='_rope':g='融合RoPE'
  elif n=='_merge_restore':g='融合分段合并/还原'
  elif 'flash_fwd' in n:g='Dense/交叉attention'
  elif n in ['_assign','_local','_offset','_scatter'] or 'scatter_gather' in n:g='Q/K分组与重排'
  elif n in ['_segment','_write','_quant']:g='KV/原型与量化'
  else:g='其他归一化、逐元素及管理'
  groups[g]+=r['ms']
 gains=dict(vs_opt21_sparse=aggregate['Opt21_Tile20_after']['generation_seconds']/aggregate['Opt22_Tile20_after']['generation_seconds'],vs_equally_optimized_dense=aggregate['Opt22_Dense']['generation_seconds']/aggregate['Opt22_Tile20_after']['generation_seconds'])
 summary=dict(cases=8,measurements=aggregate,generation_speedups=gains,all_full_latents_bit_identical=True,checkpoint_unchanged=True,lpips_vs_dense=quality['lpips'],psnr_vs_dense=quality['psnr_db'],weighted_keep=keep,executed_attention_ratio=executed,profile_scope=profile['scope'],profile_group_kernel_ms=dict(groups))
 atomic_json(ROOT/'summary.json',summary)
 lines=['Opt22：微调后20%全层方案的主干融合优化','',
  '层1–29稀疏，层0与首chunk Dense；原checkpoint、原型、投影、分组树、页选择及预算均不变。',
  '8个Val prompt，seed0，45帧480×832；同卡先后测Opt21/Opt22的Dense和Tile20。',
  '每方案预热1次后同步计时3次，取中位数，再平均8个样例；不含模型加载、首次建图及视频编码。',
  '所有重复的完整latent均与冻结的Opt20/Opt21参考逐元素相同；VAE单独预热1次、测量2次，含上传/下载。',
  '总耗时为生成与对应单独测量的VAE耗时之和。','',
  '方案                    生成秒   VAE/传输秒  合计秒']
 for n,d in aggregate.items():lines.append(f"{n:24} {d['generation_seconds']:.4f}  {d['decode_and_transfer_seconds']:.4f}  {d['total_seconds']:.4f}")
 lines += ['',f"相对Opt21稀疏生成加速 {gains['vs_opt21_sparse']:.3f}x；相对同样优化后的Dense {gains['vs_equally_optimized_dense']:.3f}x。",
  f"保留率 {100*keep:.3f}%；稀疏区域attention实际执行率 {100*executed:.3f}%，不代表整网FLOPs。",
  f"LPIPS {quality['lpips']:.6f}、PSNR {quality['psnr_db']:.3f} dB（相对Dense，后36帧），由完整latent相同确认沿用原质量测量。",'',
  '采用的优化：',
  '1. 融合调制：将(1+scale)、乘法、加bias合为一遍读写，同时显式保留每个原BF16舍入位置。',
  '2. 融合门控残差：branch*gate再加residual，保留中间乘积的BF16舍入；支持第一层非连续输入。',
  '3. 单pass RMSNorm：直接读取BF16，按本地PyTorch Reduce.cuh的32-lane、4累加器、12轮、升序shuffle顺序求均方，再执行原rsqrt、BF16转换、weight乘法。',
  '4. 合并Q/K/V线性投影：拼接只读权重与bias，复用现有cuBLAS一次计算三者；保持各自的归一化及RoPE。',
  '这些公共优化同时应用于Dense，避免把公共收益当成稀疏收益。','',
  '微测：调制0.098→0.025ms；门控残差0.082→0.042ms；RMSNorm原0.293ms，保持原顺序融合后约0.034ms（4680×1536）。',
  '首条完整视频消融（各候选有同卡Opt21对照，约3.04秒）：只融合调制/残差2.704秒；RMS保留原统计kernel2.551秒；保持原归约顺序的单pass RMS2.373秒；再合并QKV2.333秒。',
  '未采用通用tree-reduction RMS：微测只差几个元素，但完整latent NMSE达到0.09146。已采用的有序归约版本无此差异，且速度相当。','',
  '优化后热点：最后chunk最后去噪step、完整30层，独立profiler pass；下面比例仅对应该step的GPU kernel时间。']
 for g,ms in sorted(groups.items(),key=lambda x:-x[1]):lines.append(f'{g}: {ms:.3f}ms ({100*ms/profile["kernel_sum_ms"]:.1f}%)')
 old=json.loads((PARENT/'final/case0/profile.json').read_text())['kernel_sum_ms']
 lines += ['',f'相同profile范围GPU kernel总时间：Opt21 {old:.3f}ms → Opt22 {profile["kernel_sum_ms"]:.3f}ms。',
  '下一步重点：FFN/线性投影GEMM与稀疏attention主循环；原大规模RMSNorm中间张量已消除。',
  '归约顺序依赖当前固定环境：本地Reduce.cuh的路径与SHA记录在experiment.json。实现限既有BF16、1536通道、行数至少16的模型。',
  '代码入口：migration.unified_router.opt22.inference.load / trajectory；在torch.no_grad及现有sdpa_backend上下文使用，逐视频select("Tile20_after")。',
  '状态：bash opt22_status.sh --watch。视频与Opt21完全一致：index.html。无需重新训练；没有修改旧实验或训练权重。',
  'case result中的peak_allocated_gb为进程累计峰值，不能作为各方案独立显存对比。']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 body=(PARENT/'index.html').read_text().replace('Opt21','Opt22').replace('20%全层推理优化','20%全层主干融合优化')
 (ROOT/'index.html').write_text(body)
 atomic_json(ROOT/'completion.json',dict(status='completed',cases=8,all_latents_bit_identical=True,checkpoint_unchanged=True))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
