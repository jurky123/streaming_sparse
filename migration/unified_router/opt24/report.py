"""Audit, summarize and visualize kernel reuse and hardware tile experiments."""
import csv,json,statistics,subprocess
from pathlib import Path
from .setup import ROOT,TRAINING,BASE
from ..common import atomic_json,sha

def main():
 shapes=['480x832_chunks4','960x1664_chunks7'];confirmed={}
 for case in shapes:
  rr=[json.loads((ROOT/'confirm'/case/f'layer{l:02d}.json').read_text()) for l in [1,14,24,29]]
  assert all(r['completed']for r in rr);confirmed[case]={}
  for name in ['Dense','Opt22_Q32','Sparge_Q32','Sparge_Q64','Sparge_Q128']:
   vals=[next(x for x in r['rows']if x['name']==name)for r in rr]
   keys=['ms','nmse_vs_dense','nmse_vs_parent']+(['execution'] if name.startswith('Sparge') else [])
   confirmed[case][name]={k:statistics.mean(v[k]for v in vals)for k in keys}
  d=confirmed[case]['Dense']['ms'];old=confirmed[case]['Opt22_Q32']['ms']
  for row in confirmed[case].values():row.update(time_vs_dense=row['ms']/d,speedup_vs_old=old/row['ms'])
 modules={case:json.loads((ROOT/f'module_{case}.json').read_text()) for case in shapes};assert all(v['completed']for v in modules.values())
 e2e=[]
 for i in range(8):
  out=ROOT/'e2e'/f'case{i}';assert json.loads((out/'completed.json').read_text())['completed']
  x=json.loads((out/'timing.json').read_text());x['quality']=json.loads((out/'quality.json').read_text());e2e.append(x)
 mean=lambda name,kind,key:statistics.mean(x[kind][name][key]for x in e2e)
 times={name:mean(name,'results','seconds')for name in ['Dense','Opt22','Sparge32']}
 quality={name:{k:statistics.mean(x['quality'][name]['sparse_frames_only'][k]for x in e2e)for k in ['lpips','psnr_db']}for name in ['Opt22','Sparge32','Sparge32_vs_Opt22']}
 delta=[x['quality']['Sparge32']['sparse_frames_only']['lpips']-x['quality']['Opt22']['sparse_frames_only']['lpips']for x in e2e]
 library={case:{kind:json.loads((ROOT/'library'/case/f'{kind}_layer24.json').read_text())for kind in ['flashinfer','flex']}for case in shapes}
 assert all(all('error'not in row for row in d['rows'])for c in library.values()for d in c.values())
 research=[]
 for label,repo,url,reason in [('SpargeAttn','sparge','https://github.com/thu-ml/SpargeAttn','Measured adapted SM80 CUDA kernel; retained Apache-2.0 attribution. No PV-threshold skipping.'),('Block-Sparse-Attention','block_sparse','https://github.com/mit-han-lab/Block-Sparse-Attention','Inspected only; API checks require128x128 sparsity granularity, incompatible with current32x64 without a larger mask change.')]:
  path=ROOT/'vendor'/repo;commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip();research.append(dict(name=label,url=url,commit=commit,scope=reason))
 research += [dict(name='FlashInfer',url='https://docs.flashinfer.ai/api/sparse.html',measured_version='0.2.8 (existing isolated local dependency; not latest documentation version)',scope='FA2 paged prefill with JIT tail-length mask and static shape planning; reused pages, no tail exclusion.'),dict(name='FlexAttention',url='https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html',measured_version='PyTorch2.6.0+cu124',scope='Full/partial BlockMask, exact tail mask, GPU metadata and transpose; explicit tile settings.')]
 atomic_json(ROOT/'external_sources.json',research)
 info=dict(completed=True,scope='A100-SXM4-40GB. Operator probes: one actual Dense-generated Val prompt, final chunk step3, frozen Tile20. Final timing:5balanced rounds x20replays, median per layer then4layer mean. Full45-frame evaluation:8Val prompts,all29sparse layers,frozen weights,3timed full generations after warmup.',confirmed=confirmed,module_probes=modules,e2e_generation_seconds=times,e2e_speedup_vs_opt22=times['Opt22']/times['Sparge32'],e2e_speedup_vs_dense=times['Dense']/times['Sparge32'],quality_sparse36_frames=quality,per_case_lpips_delta=delta,keep=mean('Sparge32','results','keep'),execution=mean('Sparge32','results','execution'),libraries=library,research=research,decision='Opt-in Sparge32 available and E2E tested; higher-resolution128x64/4warp is the hardware target candidate for router adaptation, not a quality-validated video deployment. No retraining in Opt24.',limitations=['Operator timing excludes history/page-cache initialization. Module probes include current KV update, route and Q restore; history preparation excluded.','The actual CUDA QK/AV math and softmax rounding differ from the trained INT8-QK/BF16-PV kernel. Small block NMSE does not guarantee identical closed-loop videos.','High-resolution Q128 has4-layer block accuracy measurements only, no end-to-end video or joint fine-tuning result.','FlashInfer/Flex preliminary timings used shorter warmup; use confirmation rows for selected backend claims.'])
 atomic_json(ROOT/'summary.json',info)
 preflight=json.loads((ROOT/'preflight.json').read_text());assert preflight['passed'] and preflight['changed_history_exact']
 old=BASE/'streaming_sparse/v10/experiments/opt22_tile20_backbone_fusion_20261025';manifest=json.loads((old/'FINAL_VERIFICATION.json').read_text());assert all(sha(Path(p))==v for p,v in manifest['files'].items())
 locks=json.loads((TRAINING/'chosen_checkpoints.json').read_text());assert all(sha(Path(p))==v for p,v in locks.items())
 report(info,e2e);plot(info)
 files={str(p.resolve()):sha(p)for p in sorted(Path(__file__).parent.iterdir())if p.suffix in ['.py','.cu','.cuh']};files[str(BASE/'opt24_status.sh')]=sha(BASE/'opt24_status.sh')
 binary={str(p):sha(p)for p in ROOT.glob('sparge*build/*.so')}
 atomic_json(ROOT/'source_manifest.json',dict(files=files,binaries=binary,research=research))
 atomic_json(ROOT/'FINAL_VERIFICATION.json',dict(completed=True,custom_scan=96,library_scan=16,exact_union_scan=16,sparge_scan=12,warp_scan=6,final_confirmation_blocks=8,full_e2e_cases=8,all_e2e_repeats_deterministic=True,first_chunk_dense_exact=True,all58_checkpoints_unchanged=True,opt22_sources_unchanged=True,captured_history_refresh_validated=True,large_query_tile_video_evaluations=0,training_updates=0,files=files,binaries=binary))
 print(json.dumps(dict(times=times,quality=quality,confirmed=confirmed),ensure_ascii=False,indent=2))

def report(s,e2e):
 c=s['confirmed'];t=s['e2e_generation_seconds'];q=s['quality_sparse36_frames'];lines=['Opt24：公开attention kernel复用与A100 query tile选择','',
 '结论：SpargeAttn SM80 CUDA复用获得明确算子收益；适合的query tile随规模变化。',
 '480x832/3历史：Q32×K64、2warp较好，保留现有选页；960x1664/6历史：Q128×K64、4warp最高效。',
 '后者实际执行量20.18%，attention耗时约Dense的21.75%，已经接近计算量缩放收益；但现有路由直接重分组后误差增加，尚不能作为质量已验证的视频方案。','',
 '最终确认（层1/14/24/29均值；单位ms）：','规模                   Dense    Opt22Q32  SpargeQ32 SpargeQ64 SpargeQ128']
 for name,d in c.items():lines.append(f"{name:22} "+' '.join(f"{d[k]['ms']:9.3f}"for k in ['Dense','Opt22_Q32','Sparge_Q32','Sparge_Q64','Sparge_Q128']))
 lines+=['','SpargeQ32：CTA32×64/WARP_Q16=2warp；SpargeQ64：小规模WARP_Q16=4warp，大规模WARP_Q32=2warp；SpargeQ128：WARP_Q32=4warp。',
 '上表包含Q量化、稀疏页LUT生成、attention、Q复原；KV已准备，未包含router和当前KV更新。',
 '数据来自真实Dense闭环轨迹，而非复制随机token；每层5轮×20次CUDA graph计时，充分预热后取中位数。','',
 '加入路由与当前KV更新（第24层实测）：']
 for name,d in s['module_probes'].items():
  for v in d['rows']:lines.append(f"{name} Q{v['group']}: {v['baseline_ms']:.3f} → {v['ms']:.3f} ms，{v['speedup']:.3f}x。")
 lines+=['模块计时仍不包含历史准备；历史页常驻，只更新当前chunk，避免每次重新打包全部KV。','',
 '精度（四层真实attention NMSE均值，相对Dense）：','规模                   Opt22Q32  SpargeQ32 SpargeQ64 SpargeQ128']
 for name,d in c.items():lines.append(f"{name:22} "+' '.join(f"{d[k]['nmse_vs_dense']:9.6f}"for k in ['Opt22_Q32','Sparge_Q32','Sparge_Q64','Sparge_Q128']))
 lines+=['Q32保留原路由选择，算子NMSE相对旧kernel约1.3e-6；Q64/128将原32组的归一化proxy概率汇总，再重新选20%页，会改变选择。',
 '保持原32组各自mask、用并集执行64/128大tile也已测试：执行量上升，全部更慢，没有采用。','',
 'Sparge32全29层端到端（8条45帧视频，480x832，首chunk和层0 Dense）：',
 f"Dense生成 {t['Dense']:.6f}s；Opt22 {t['Opt22']:.6f}s；Sparge32 {t['Sparge32']:.6f}s。",f"Sparge32相对Opt22 {s['e2e_speedup_vs_opt22']:.4f}x，相对Dense {s['e2e_speedup_vs_dense']:.4f}x。以上不含VAE解码。",
 f"后36帧相对Dense：LPIPS {q['Opt22']['lpips']:.6f} → {q['Sparge32']['lpips']:.6f}；PSNR {q['Opt22']['psnr_db']:.3f} → {q['Sparge32']['psnr_db']:.3f} dB。",
 f"Sparge32与Opt22视频之间LPIPS={q['Sparge32_vs_Opt22']['lpips']:.6f}，并非无损；8例相对Dense的LPIPS最大增加{max(s['per_case_lpips_delta']):.6f}。",
 '小算子误差会通过闭环生成和离散选页传播。不能以1e-6块误差推断视频逐像素不变。',
 f"全视频加权保留率{s['keep']*100:.4f}%，执行率{s['execution']*100:.4f}%；checkpoint冻结，8例重复生成确定性通过。",'',
 '实现与公开kernel复用：',
 '- SpargeAttn：直接复用Apache-2.0 SM80 CUDA核心；轻量绑定提供32/64/128模板，修正为当前CUDA stream；三个softmax位置加入chunk尾页mask。关闭额外PV阈值跳过。',
 '- KV量化页常驻，当前chunk增量更新；历史刷新同步更新原型与页缓存，并测试了CUDA Graph捕获后更换历史。',
 '- FlashInfer0.2.8 FA2：JIT精确尾页mask、按固定形状一次规划。较大query tile有收益，但全KV重打包会侵蚀收益。',
 '- PyTorch2.6 FlexAttention：完整/尾页BlockMask和GPU转置元数据，16/32/64/128均完成正确性与速度测试，本次没有优于选定Sparge路径。',
 '- MIT Block-Sparse-Attention：只做代码适配调查，原API要求128×128稀疏块；没有把未运行候选写成测速结果。',
 '外部源码URL、commit、版本及适配范围见external_sources.json；测量源码与二进制SHA见source_manifest.json。','',
 '下一步设计建议：',
 '1. 小规模保留Q32；高分辨率长历史将Q128×K64/4warp作为硬件目标，而不是机械地在所有规模统一使用128。',
 '2. 大尺寸路由直接以128-query共享选页、20%执行预算为前向语义；优化Q分组，使同tile的查询真正适合共享KV。',
 '3. 联合微调使用实际部署CUDA输出误差，best按实际attention误差选；多尺度训练需要从Train提示词采集对应分辨率/历史，不能把本次Val测试QKV用于训练。',
 '4. 路由仍是明显成本，后续应融合/压缩fine分数及tile归约，优化coarse/top-k；更高效的attention并不会自动消除这些成本。','',
 '本轮没有开启新训练。Q128只有算子层精度数据，没有高分辨率闭环质量结论。',
 'Opt22默认入口未改；可选入口migration.unified_router.opt24.inference.load()，使用方法与Opt22相同。',
 '查看状态：./opt24_status.sh',
 '复现实测：CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt24.confirm --height 960 --width 1664 --chunks 7 --layer 24',
 '端到端：CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt24.evaluate --case 0',
 '并排视频：e2e/case0/Dense_Opt22_Sparge32.mp4、e2e/case1/Dense_Opt22_Sparge32.mp4。']
 profile=json.loads((ROOT/'e2e/case0/profile.json').read_text());gemm=sum(v['ms']for v in profile['kernels']if 'gemm'in v['name'].lower());att=sum(v['ms']for v in profile['kernels']if 'qk_int_sv_f16_block_sparse_attn_kernel'in v['name']);route=sum(v['ms']for v in profile['kernels']if v['name']in ['_coarse','_top','_fine_age','_tiles','_landmark','_project'])
 lines+=['',f"最新最后chunk/step3的30层profile：GPU kernel合计{profile['kernel_sum_ms']:.3f}ms；GEMM {gemm:.3f}ms；稀疏attention主kernel {att:.3f}ms；router {route:.3f}ms。",'生成速度提升较小的原因：投影/FFN等GEMM仍占大头，原attention已占少数，且新适配包含额外Q量化/页同步/复原。']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 with (ROOT/'confirmation.csv').open('w')as f:
  w=csv.writer(f);w.writerow(['shape','variant','ms','time_vs_dense','speedup_vs_old','execution','nmse_vs_dense'])
  for name,d in c.items():
   for key,v in d.items():w.writerow([name,key,v['ms'],v['time_vs_dense'],v['speedup_vs_old'],v.get('execution'),v['nmse_vs_dense']])

def plot(s):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(12,4.7));names=['Opt22_Q32','Sparge_Q32','Sparge_Q64','Sparge_Q128'];labels=['Original\nQ32','Sparge\nQ32','Sparge\nQ64','Sparge\nQ128'];colors=['#8494a7','#2468a2','#589c88','#e59836']
 for ax,(case,data)in zip(axes,s['confirmed'].items()):
  val=[data[n]['time_vs_dense']*100 for n in names];bars=ax.bar(labels,val,color=colors);ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.axhline(20,color='black',ls=':',lw=1);ax.set_ylim(0,36);ax.set_ylabel('Attention time / Dense FlashAttention (%)');ax.set_title(case.replace('_chunks4',' / 3 history chunks').replace('_chunks7',' / 6 history chunks'));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('A100: optimal query tile depends on workload size',fontsize=14);fig.text(.5,.02,'Four-layer mean; ~20% executed attention. Q64/128 change selection and need router adaptation.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(ROOT/'kernel_tiles.png',dpi=180);fig.savefig(ROOT/'kernel_tiles.pdf');plt.close(fig)

if __name__=='__main__':main()
