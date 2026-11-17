"""Single-axis Full_Pset ablation and three top-quarter selection rules, two fixed seeds."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from .weights import ROOT
OUT=ROOT/'artifacts/video_runtime_v1/pset_scope_v1'
SEEDS=[701,702]
def write(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(path)
def configure_scope(rt,spec):
 rt.full_pset=spec['name']!='base';rt.pset_steps='all'
 if 'schedule' in spec:rt.set_pset_schedule(spec['schedule'])
 else:rt.set_pset_scope(spec.get('layers'),spec.get('heads'),spec.get('steps'))
@torch.no_grad()
def worker(a):
 from .deployment import current,configure,load_weights,make_runtime,identity
 from ..broad_common import Engine
 from ..rollout_suite import generate_chunk
 from ..stream_eval_common import native_vae,decode,save_video
 from ..high_sparse_experiment import prefix_quality
 import lpips
 torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=512;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=current(static_route=True,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg);model,reader,phi,B,weights=load_weights(cfg)
 engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 vae=native_vae();os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache');metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 jobs=json.loads((OUT/(a.phase+'_manifest.json')).read_text())[a.index::a.workers]
 source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
 write(OUT/'workers'/f'{a.phase}_{a.index}_identity.json',dict(deployment=identity(cfg),weights=weights,source=source,gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
 def generate(seed):
  rt.reset();zs=[generate_chunk(engine,emb,seed,c) for c in range(9)];torch.cuda.synchronize();return torch.cat(zs,1)
 try:
  for job in jobs:
   name=job['name'];seed=job['seed'];folder=OUT/a.phase/name/f's{seed}';folder.mkdir(parents=True,exist_ok=True)
   if (folder/'result.json').exists():continue
   write(OUT/'workers'/f'{a.phase}_{a.index}_status.json',dict(state='running',job=job))
   configure_scope(rt,job)
   t=time.perf_counter();generate(seed);cold=time.perf_counter()-t
   torch.cuda.synchronize();t=time.perf_counter();z=generate(seed);generation=time.perf_counter()-t
   stats=rt.finish();assert stats['counters'][2]==0;assert torch.isfinite(z).all()
   torch.save(z.cpu(),folder/'latents.pt');t=time.perf_counter();pixels=decode(vae,z,105);torch.cuda.synchronize();decode_s=time.perf_counter()-t
   dense=ROOT/f'artifacts/video_runtime_v1/quality_batch_9d4d33d_v1/videos/city_walk_s{seed}/chunks9/dense/reference_pixels.npy'
   quality=prefix_quality(pixels,[np.load(dense,mmap_mode='r')],metric)
   save_video(folder/'video.mp4',pixels)
   # Decoded pixels are only retained for the controls and final combined policies.
   if name in ('base','full') or a.phase=='selected':np.save(folder/'pixels.npy',np.concatenate(pixels))
   write(folder/'result.json',dict(job=job,quality=quality,generation_seconds=generation,cold_generation_seconds=cold,decode_seconds=decode_s,runtime=stats,worker=a.index,source=source,weights=weights,scope='one full-video warmup then one timed generation, excludes IO; two-seed quality ranking, not multi-GPU aggregate speed'))
   print(name,seed,generation,quality['lpips'],stats['counters'],flush=True)
  write(OUT/'workers'/f'{a.phase}_{a.index}_status.json',dict(state='done',jobs=len(jobs)))
 finally:rt.close()
def initial():
 specs=[dict(name='base'),dict(name='full')]
 specs += [dict(name=f'layer_{l:02}',layers=[l]) for l in range(1,30)]
 specs += [dict(name=f'head_{h:02}',heads=[h]) for h in range(12)]
 specs += [dict(name=f'step_{s}',steps=[s]) for s in range(4)]
 write(OUT/'individual_manifest.json',[dict(spec,seed=seed) for spec in specs for seed in SEEDS])
def select():
 read=lambda n,s:json.loads((OUT/'individual'/n/f's{s}/result.json').read_text())
 base={s:read('base',s)['quality']['lpips'] for s in SEEDS};ranks={}
 for axis,indices in [('layer',range(1,30)),('head',range(12)),('step',range(4))]:
  rows=[]
  for i in indices:
   name=f'{axis}_{i:02}' if axis!='step' else f'step_{i}'
   records=[read(name,s) for s in SEEDS];gains=[base[s]-v['quality']['lpips'] for s,v in zip(SEEDS,records)]
   rows.append(dict(index=i,gain=float(np.mean(gains)),seed_gains=dict(zip(SEEDS,gains)),mean_lpips=float(np.mean([v['quality']['lpips'] for v in records])),generation_seconds=float(np.mean([v['generation_seconds'] for v in records]))))
  ranks[axis]=sorted(rows,key=lambda r:(-r['gain'],r['index']))
 chosen={a:[r['index'] for r in ranks[a][:len(ranks[a])//4]] for a in ranks}
 specs=[dict(name='top_layers',layers=chosen['layer']),dict(name='top_heads',heads=chosen['head']),dict(name='top_steps',steps=chosen['step']),dict(name='intersection',layers=chosen['layer'],heads=chosen['head'],steps=chosen['step'])]
 g={a:{r['index']:r['gain'] for r in ranks[a]} for a in ranks}
 cells=sorted([(g['layer'][l]+g['head'][h]+g['step'][s],l,h,s) for l in range(1,30) for h in range(12) for s in range(4)],key=lambda x:(-x[0],x[1],x[2],x[3]))
 schedule={}
 for gain,l,h,s in cells[:len(cells)//4]:schedule.setdefault(f'{l}:{s}',[]).append(h)
 for heads in schedule.values():heads.sort()
 specs.append(dict(name='additive_quarter',schedule=schedule))
 write(OUT/'selection.json',dict(base_lpips=base,rankings=ranks,selected=chosen,additive_selected_cells=cells[:len(cells)//4],selected_cells=348,total_cells=1392,rank_rule='mean over seed701/702 of LPIPS(base,Dense)-LPIPS(single-axis,Dense); ties ascending index',rounding='floor: 7/29 layers, 3/12 heads, 1/4 noisy steps; layer0 and clean step4 are fixed Dense/no correction',note='additive scores combine effects of axes with different coverage; per user independent-gain sum, not a calibrated per-cell prediction'))
 write(OUT/'selected_manifest.json',[dict(spec,seed=seed) for spec in specs for seed in SEEDS])
def run(a):
 OUT.mkdir(parents=True,exist_ok=True)
 write(OUT/(a.phase+'_status.json'),dict(state='running'))
 if a.phase=='individual':initial()
 else:select()
 procs=[]
 for i,gpu in enumerate(a.gpus):
  log=(OUT/f'{a.phase}_gpu{gpu}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
  p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.scope_experiment','--worker','--phase',a.phase,'--index',str(i),'--workers',str(len(a.gpus))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
 codes=[p.wait() for p,_ in procs]
 for _,log in procs:log.close()
 write(OUT/(a.phase+'_status.json'),dict(state='done' if not any(codes) else 'failed',exit_codes=codes))
 if any(codes):raise RuntimeError(codes)
 if a.phase=='individual':select()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--phase',choices=['individual','selected'],default='individual');p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=8);p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));a=p.parse_args();worker(a) if a.worker else run(a)
