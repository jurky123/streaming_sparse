"""Paired final timing, decode, and reports for measured fixed-budget search."""
import argparse,csv,json,os,statistics,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .intersection_search import OUT,OLD,SEEDS,normalize,policy_id
from .scope_experiment import configure_scope,write
from .weights import ROOT

def choices():return json.loads((OUT/'summary.json').read_text())['selected']
@torch.no_grad()
def timing(a):
 from .deployment import current,configure,load_weights,make_runtime
 from ..broad_common import Engine
 from ..rollout_suite import generate_chunk
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=512
 cfg=current(static_route=True,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg);model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 policies=[dict(name='base'),dict(name='full')]+[dict(name=k,**v['policy']) for k,v in choices().items()];cache={}
 def generate():
  rt.reset()
  for c in range(9):generate_chunk(engine,emb,a.seed,c)
  torch.cuda.synchronize()
 try:
  for spec in policies:
   key=policy_id(spec) if 'layers' in spec else spec['name'];path=OUT/'timing'/f's{a.seed}'/(spec['name']+'.json')
   if key not in cache:
    configure_scope(rt,spec);generate();times=[]
    for _ in range(3):
     torch.cuda.synchronize();start=time.perf_counter();generate();times.append(time.perf_counter()-start)
    cache[key]=dict(seconds=times,median_seconds=statistics.median(times),counters=rt.counters.cpu().tolist())
   write(path,dict(policy=spec,seed=a.seed,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),**cache[key],scope='same GPU per seed, one full warmup, 3 repetitions; generation only, excludes VAE and IO; duplicate policies reuse timing'))
   print(spec['name'],a.seed,cache[key]['median_seconds'],flush=True)
 finally:rt.close()
@torch.no_grad()
def decode_final(a):
 from ..stream_eval_common import native_vae,decode,save_video
 from ..high_sparse_experiment import prefix_quality
 import lpips
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache');metric=lpips.LPIPS(net='alex').eval().requires_grad_(False);vae=native_vae()
 dense=np.load(ROOT/f'artifacts/video_runtime_v1/quality_batch_9d4d33d_v1/videos/city_walk_s{a.seed}/chunks9/dense/reference_pixels.npy',mmap_mode='r');previous=np.load(OLD/f'selected/intersection/s{a.seed}/pixels.npy',mmap_mode='r')
 for name,row in choices().items():
  folder=OUT/'final'/name/f's{a.seed}';folder.mkdir(parents=True,exist_ok=True);z=torch.load(OUT/'results'/row['name']/f's{a.seed}/latents.pt',weights_only=True).cuda()
  frames=decode(vae,z,105);pixels=np.concatenate(frames);quality=prefix_quality(frames,[dense],metric);search=json.loads((OUT/'results'/row['name']/f's{a.seed}/result.json').read_text())['quality'];save_video(folder/'video.mp4',frames)
  delta=pixels.astype(np.float32)-previous.astype(np.float32)
  write(folder/'quality.json',dict(quality=quality,lpips_delta_vs_search=quality['lpips']-search['lpips'],rgb_mae_vs_original_255=float(np.abs(delta).mean()),rgb_equal_to_original=np.array_equal(pixels,previous)))
  print(name,a.seed,quality['lpips'],flush=True)
 paths=[ROOT/f'artifacts/video_runtime_v1/quality_batch_9d4d33d_v1/videos/city_walk_s{a.seed}/chunks9/dense/video.mp4']+[OUT/'final'/n/f's{a.seed}/video.mp4' for n in choices()]
 cmd=['ffmpeg','-y','-loglevel','error']
 for p in paths:cmd+=['-i',str(p)]
 subprocess.run(cmd+['-filter_complex',f'hstack=inputs={len(paths)}','-an','-c:v','libx264','-threads','4','-crf','20',str(OUT/f'comparison_s{a.seed}.mp4')],check=True)

def report():
 summary=json.loads((OUT/'summary.json').read_text());selected=summary['selected'];table=[];names=['base','full',*selected]
 for name in names:
  records=[json.loads((OUT/'timing'/f's{s}'/(name+'.json')).read_text()) for s in SEEDS];generation=float(np.mean([x['median_seconds'] for x in records]))
  if name in selected:row=dict(selected[name])
  else:
   rows=[json.loads((OLD/'individual'/name/f's{s}/result.json').read_text()) for s in SEEDS];row={k:float(np.mean([x['quality'][k] for x in rows])) for k in ['lpips','ssim','psnr']};row['seed_lpips']={str(s):x['quality']['lpips'] for s,x in zip(SEEDS,rows)}
  table.append(dict(label=name,generation_seconds=generation,**row))
 original=next(x for x in table if x['label']=='original');full=next(x for x in table if x['label']=='full')
 for row in table:row['speedup_vs_original']=original['generation_seconds']/row['generation_seconds'];row['speedup_vs_full']=full['generation_seconds']/row['generation_seconds'];row['lpips_gain_vs_original']=original['lpips']-row['lpips']
 write(OUT/'comparison.json',dict(table=table,unique_policies=summary['unique_policies'],videos=summary['videos'],selected=selected))
 fields=['label','generation_seconds','speedup_vs_original','speedup_vs_full','lpips','ssim','psnr','lpips_gain_vs_original']
 with (OUT/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in table)
 with (OUT/'ranking.csv').open('w') as f:
  fields=['rank','name','layers','heads','steps','mean_lpips','lpips_s701','lpips_s702','gain_vs_original'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for i,r in enumerate(summary['ranking'],1):w.writerow(dict(rank=i,name=r['name'],**{k:','.join(map(str,r['policy'][k])) for k in ['layers','heads','steps']},mean_lpips=r['lpips'],lpips_s701=r['seed_lpips']['701'],lpips_s702=r['seed_lpips']['702'],gain_vs_original=original['lpips']-r['lpips']))
 labels=dict(base='C＋Risk基础档',full='完整Full_Pset',original='原交集',best_mean='搜索：平均LPIPS最佳',best_balanced='搜索：两seed均不退化最佳')
 lines=['# 交集预算的两轮实际视频邻域搜索','','固定7层×3head×1noisy step=21个位置，预算21/1392=1.5086%，每个短视频评分10584次。没有增加修正预算或改变权重、候选数、gain阈值。只搜索三个集合的成员。', '', '## 搜索范围与判据', '', '- city_walk、seed701/702、9chunks/105帧、768×512、16FPS、history360、tile72；M16入口、固定编译形状、G2关闭。两seed平均Dense LPIPS为主目标。', '- 每轮完整评估154个单层替换（7×22）、27个单head替换（3×9）、3个step替换，共184个邻居。选实际视频平均LPIPS最小者进入下一轮，改进阈值1e-6；最多两轮，重复配置复用结果。', '- 除平均最优外，还从所有已测配置选出两个seed各自都不差于原交集（容差1e-6）的平均最优者。这个平衡候选是额外展示，不替代主搜索目标。', f"- 共测{summary['unique_policies']}个不同配置、{summary['videos']}份视频，另有初始逐位连续性检查和最后的计时重复。8卡各常驻一个worker，模型、VAE和编译缓存复用；每配置都从chunk0重建历史，未复用其他配置的生成轨迹。", '- Dense参考复用既有同prompt/seed的像素缓存；LPIPS(AlexNet)/SSIM在256×384计算，PSNR用原分辨率MSE；指标窗口[21,101)。', '- 两个seed同时用于搜索和报告，没有独立验证集；两轮搜索不证明全局最优。若第二轮仍有提升，只能称为有界搜索结束，不能称为邻域收敛。', '', '## 每轮结果', '', '|轮次|进入轮次LPIPS↓|邻域最佳LPIPS↓|本轮改善↑|已测不同配置数|','|---|---:|---:|---:|---:|']
 for r in summary['rounds']:lines.append(f"|{r['round']}|{r['center']['lpips']:.6f}|{r['winner']['lpips']:.6f}|{r['center']['lpips']-r['winner']['lpips']:.6f}|{r['new_unique_total']}|")
 lines+=['', '## 最终配置', '', '|方案|层|head|noisy step|','|---|---|---|---|']
 for name,row in selected.items():p=row['policy'];lines.append(f"|{labels[name]}|{p['layers']}|{p['heads']}|{p['steps']}|")
 if selected['best_mean']['name']==selected['best_balanced']['name']:lines+=['','平均最优和“两seed均不退化”最优是同一配置，下表为展示两个判据保留同一行数据；没有把重复项计为新配置。']
 lines+=['', '## 同口径速度与质量', '', '|方案|生成/s↓|相对原交集速度↑|相对完整档速度↑|LPIPS↓|SSIM↑|PSNR/dB↑|','|---|---:|---:|---:|---:|---:|---:|']
 for r in table:lines.append(f"|{labels[r['label']]}|{r['generation_seconds']:.4f}|{r['speedup_vs_original']:.4f}×|{r['speedup_vs_full']:.3f}×|{r['lpips']:.6f}|{r['ssim']:.6f}|{r['psnr']:.3f}|")
 lines+=['','seed701全部最终方案在GPU6测量，seed702在GPU7；A100-SXM4-40GB、CPU线程2，完整预热一次，3次生成取中位数，再对两seed等权平均。排除模型加载、编译、VAE、IO。搜索worker的单次生成时间只用于吞吐日志，不作正式加速比。基础/完整档质量引用上一轮相同固定形状控制，当前重新计时；原交集和搜索结果重新解码，指标与搜索记录对照。', '', '|方案|seed701 LPIPS↓|对原交集改善↑|seed702 LPIPS↓|对原交集改善↑|','|---|---:|---:|---:|---:|']
 for name,r in selected.items():a=r['seed_lpips'];o=selected['original']['seed_lpips'];lines.append(f"|{labels[name]}|{a['701']:.6f}|{o['701']-a['701']:+.6f}|{a['702']:.6f}|{o['702']-a['702']:+.6f}|")
 lines+=['', '![双seed质量分布](search_frontier.png)', '', '虚线标出原交集的两个seed指标；左下区域表示两个seed都改善。所有点来自相同21个位置预算的实际视频。', '', '## 执行与复现', '', '每个worker先对原交集的两个seed生成latent，并与上一轮逐位比较；新鲜解码的控制LPIPS也必须与旧记录相差小于1e-6。每个候选检查10584个评分事件、无非有限分数，固定形状避免已发现的任务顺序影响。所有候选保存latent和逐帧指标；最终候选另保存视频，不在搜索阶段编码数百个重复用途视频。', '', '没有更改全局默认。启用搜索结果：在当前stream命令添加`--full-pset --pset-policy artifacts/video_runtime_v1/intersection_search_v1/presets/best_mean.json`；希望明确使用双seed均不退化判据时改为`best_balanced.json`。文件SHA进入identity，preset强制static_route=True。', '', '实现：[intersection_search.py](../../../local_experiments/video_runtime_v1/intersection_search.py)负责固定预算邻居生成、SQLite领任务、常驻模型、两轮选择和完整排名；[intersection_finalize.py](../../../local_experiments/video_runtime_v1/intersection_finalize.py)负责最终同卡计时、解码和报告。范围执行复用`runtime.py::set_pset_scope`与`pipeline_graph.py`的selected-head路径，没有新增kernel或算法近似。', '', '在仓库根用项目venv执行`python -m local_experiments.video_runtime_v1.intersection_search`。结束后分别执行`intersection_finalize --mode timing --seed 701/702`、`--mode decode --seed 701/702`和`--mode report`。既有结果根保留，重做实验需新根目录，避免覆盖本轮证据。', '', '原始记录：[protocol.json](protocol.json)、[全部排名CSV](ranking.csv)、[结果CSV](comparison.csv)、[两轮及全部配置](summary.json)、[可部署preset](presets/)。每个候选`results/<policy>/s<seed>/result.json`保存逐帧指标；`round1/2_manifest.json`记录邻居，`round1/2_result.json`记录当轮全部分数。', '', '四列并排视频依次为Dense、原交集、平均最优、双seed均不退化最优：[seed701](comparison_s701.mp4)、[seed702](comparison_s702.mp4)。']
 (OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 first={policy_id(x['policy']) for x in json.loads((OUT/'round1_manifest.json').read_text())['neighbors']}
 fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
 for is_first,color,label in [(True,'#708da9','Round 1'),(False,'#dca16b','New in round 2')]:
  rr=[r for r in summary['ranking'] if (r['name'] in first)==is_first and r['name']!=selected['original']['name']]
  ax.scatter([r['seed_lpips']['701'] for r in rr],[r['seed_lpips']['702'] for r in rr],s=18,alpha=.65,c=color,label=label)
 for name,color,marker in [('original','#222222','X'),('best_mean','#c34740','*'),('best_balanced','#258866','*')]:
  rr=selected[name];ax.scatter(rr['seed_lpips']['701'],rr['seed_lpips']['702'],s=160,c=color,marker=marker,edgecolors='white',linewidths=.7,label=name,zorder=5)
 ax.axvline(original['seed_lpips']['701'],color='#777777',ls='--',lw=.8);ax.axhline(original['seed_lpips']['702'],color='#777777',ls='--',lw=.8)
 ax.set_xlabel('LPIPS to Dense: seed 701 (lower is better)');ax.set_ylabel('LPIPS to Dense: seed 702 (lower is better)');ax.set_title(f"Fixed 21-cell budget: {summary['unique_policies']} measured policies");ax.grid(alpha=.15);ax.legend(fontsize=9)
 fig.savefig(OUT/'search_frontier.png',dpi=160);fig.savefig(OUT/'search_frontier.svg');plt.close(fig)
 print(json.dumps(table,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['timing','decode','report'],required=True);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args();timing(a) if a.mode=='timing' else decode_final(a) if a.mode=='decode' else report()
