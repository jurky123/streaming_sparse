"""Two-round measured one-swap search of 7 layers x 3 heads x 1 noisy step."""
import argparse,hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
from .weights import ROOT
from .scope_experiment import configure_scope,write
OUT=ROOT/'artifacts/video_runtime_v1/intersection_search_v1'
OLD=ROOT/'artifacts/video_runtime_v1/pset_scope_v1'
SEEDS=[701,702]

def normalize(p):return {k:sorted(p[k]) for k in ('layers','heads','steps')}
def policy_id(p):return 'p_'+hashlib.sha256(json.dumps(normalize(p),sort_keys=True).encode()).hexdigest()[:12]
def neighbors(p):
 p=normalize(p);out=[]
 for axis,universe in [('layers',range(1,30)),('heads',range(12)),('steps',range(4))]:
  for remove in p[axis]:
   for add in universe:
    if add in p[axis]:continue
    q=dict(p);q[axis]=sorted((set(p[axis])-{remove})|{add});out.append(dict(policy=q,axis=axis,remove=remove,add=add))
 assert len(out)==184 and len({policy_id(x['policy']) for x in out})==184
 return out

def database():
 OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'jobs.sqlite',timeout=60,isolation_level=None)
 db.execute('CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, spec TEXT NOT NULL, ordinal INTEGER NOT NULL, status TEXT NOT NULL DEFAULT "pending", worker INTEGER, error TEXT)')
 db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)');return db

def enqueue(db,policies,round_id):
 for i,p in enumerate(policies):
  spec=dict(name=policy_id(p),**normalize(p))
  write(OUT/'policies'/(spec['name']+'.json'),dict(spec,static_route=True))
  for j,seed in enumerate(SEEDS):db.execute('INSERT OR IGNORE INTO jobs(key,spec,ordinal) VALUES (?,?,?)',(spec['name']+f'_s{seed}',json.dumps(dict(spec,seed=seed),sort_keys=True),round_id*10000+i*2+j))

def read_scores(p):
 name=policy_id(p);rows=[json.loads((OUT/'results'/name/f's{s}/result.json').read_text()) for s in SEEDS]
 return dict(name=name,policy=normalize(p),lpips=sum(v['quality']['lpips'] for v in rows)/2,ssim=sum(v['quality']['ssim'] for v in rows)/2,psnr=sum(v['quality']['psnr'] for v in rows)/2,seed_lpips={str(s):v['quality']['lpips'] for s,v in zip(SEEDS,rows)})

def worker(a):
 import numpy as np
 import torch
 from .deployment import current,configure,load_weights,make_runtime,identity
 from ..broad_common import Engine
 from ..rollout_suite import generate_chunk
 from ..stream_eval_common import native_vae,decode
 from ..high_sparse_experiment import prefix_quality
 import lpips
 torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=512;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=current(static_route=True,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg);model,reader,phi,B,weights=load_weights(cfg)
 engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg);vae=native_vae()
 os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache');metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 initial=normalize(json.loads((OLD/'presets/intersection.json').read_text()));db=database()
 @torch.no_grad()
 def generate(seed):
  rt.reset();zs=[generate_chunk(engine,emb,seed,c) for c in range(9)];torch.cuda.synchronize();return torch.cat(zs,1)
 try:
  configure_scope(rt,dict(name='initial',**initial));checks={}
  for seed in SEEDS:
   z=generate(seed);ref=torch.load(OLD/f'selected/intersection/s{seed}/latents.pt',weights_only=True).cuda();checks[seed]=torch.equal(z,ref);assert checks[seed],('initial continuity',a.gpu,seed)
  write(OUT/'workers'/f'gpu{a.gpu}_identity.json',dict(deployment=identity(cfg),weights=weights,initial_latent_equal=checks,source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
  references={s:np.load(ROOT/f'artifacts/video_runtime_v1/quality_batch_9d4d33d_v1/videos/city_walk_s{s}/chunks9/dense/reference_pixels.npy',mmap_mode='r') for s in SEEDS}
  while True:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT key,spec FROM jobs WHERE status="pending" ORDER BY ordinal,key LIMIT 1').fetchone()
   if row:db.execute('UPDATE jobs SET status="running",worker=? WHERE key=?',(a.gpu,row[0]))
   db.execute('COMMIT')
   if row is None:
    stop=db.execute('SELECT value FROM meta WHERE key="stop"').fetchone()
    if stop and stop[0]=='true':break
    time.sleep(1);continue
   key,spec=row;job=json.loads(spec);seed=job['seed'];folder=OUT/'results'/job['name']/f's{seed}'
   write(OUT/'workers'/f'gpu{a.gpu}_status.json',dict(state='running',key=key,job=job))
   try:
    configure_scope(rt,job);t=time.perf_counter();z=generate(seed);seconds=time.perf_counter()-t
    counters=rt.counters.cpu().tolist();assert counters[1]==10584 and counters[2]==0,counters;assert torch.isfinite(z).all()
    folder.mkdir(parents=True,exist_ok=True);torch.save(z.cpu(),folder/'latents.pt')
    with torch.no_grad():pixels=decode(vae,z,105);quality=prefix_quality(pixels,[references[seed]],metric)
    write(folder/'result.json',dict(job=job,quality=quality,counters=counters,generation_seconds=seconds,gpu=a.gpu,static_route=True,weights=weights,timing_scope='search throughput only; resident worker and warmed graph shapes, formal paired timing follows selection'))
    db.execute('UPDATE jobs SET status="done",error=NULL WHERE key=?',(key,));print(key,quality['lpips'],seconds,flush=True)
   except BaseException as exc:
    db.execute('UPDATE jobs SET status="failed",error=? WHERE key=?',(repr(exc),key));raise
  write(OUT/'workers'/f'gpu{a.gpu}_status.json',dict(state='done'))
 finally:rt.close();db.close()

def run(a):
 initial=normalize(json.loads((OLD/'presets/intersection.json').read_text()));db=database()
 assert not list(OUT.glob('summary.json')),'Completed search exists; retain its evidence'
 db.execute('UPDATE jobs SET status="pending" WHERE status IN ("running","failed")');db.execute('INSERT OR REPLACE INTO meta VALUES ("stop","false")')
 write(OUT/'protocol.json',dict(initial=initial,seeds=SEEDS,case='city_walk',chunks=9,frames=105,resolution=[512,768],history_pages=360,tile_batch=72,static_route=True,round_limit=2,neighbors_per_round=184,objective='mean two-seed LPIPS to cached Dense',budget_cells=21,total_cells=1392,acceptance='strict mean improvement > 1e-6; no seed-specific filtering for primary winner; also retain best candidate non-worse on each seed',scope='bounded single-exchange neighborhoods; no global-optimum claim'))
 source=OUT/'source';source.mkdir(exist_ok=True)
 for name in ['intersection_search.py','runtime.py','pipeline_graph.py','deployment.py','weights.py']:(source/name).write_bytes((Path(__file__).parent/name).read_bytes())
 enqueue(db,[initial],0);procs=[]
 for gpu in a.gpus:
  log=(OUT/f'gpu{gpu}.log').open('a');p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.intersection_search','--worker','--gpu',str(gpu)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
 def wait_policies(policies):
  keys=[policy_id(p)+f'_s{s}' for p in policies for s in SEEDS]
  while True:
   states=dict(db.execute('SELECT key,status FROM jobs'));counts={v:list(states.values()).count(v) for v in set(states.values())};write(OUT/'progress.json',dict(state='running',counts=counts))
   if any(p.poll() not in (None,0) for p,_ in procs):raise RuntimeError('Worker failed; see GPU logs')
   if any(states[k]=='failed' for k in keys):raise RuntimeError('Search job failed')
   if all(states[k]=='done' for k in keys):return
   time.sleep(5)
 try:
  wait_policies([initial]);original=read_scores(initial)
  for seed in SEEDS:
   old=json.loads((OLD/f'selected/intersection/s{seed}/result.json').read_text())['quality']['lpips'];assert abs(original['seed_lpips'][str(seed)]-old)<1e-6
  center=initial;rounds=[];seen={policy_id(initial):initial}
  for round_id in [1,2]:
   candidates=neighbors(center);policies=[x['policy'] for x in candidates]
   write(OUT/f'round{round_id}_manifest.json',dict(center=center,neighbors=candidates));enqueue(db,policies,round_id);wait_policies(policies)
   rows=[read_scores(p) for p in policies];before=read_scores(center);best=min([before]+rows,key=lambda v:(v['lpips'],v['name']));improved=best['lpips']<before['lpips']-1e-6
   for p in policies:seen[policy_id(p)]=p
   rounds.append(dict(round=round_id,center=before,winner=best,improved=improved,new_unique_total=len(seen)));write(OUT/f'round{round_id}_result.json',dict(**rounds[-1],ranked=sorted(rows,key=lambda v:(v['lpips'],v['name']))));print('round',round_id,best,flush=True)
   if not improved:break
   center=best['policy']
  ranking=sorted([read_scores(p) for p in seen.values()],key=lambda v:(v['lpips'],v['name']));balanced=[r for r in ranking if all(r['seed_lpips'][str(s)]<=original['seed_lpips'][str(s)]+1e-6 for s in SEEDS)]
  best=ranking[0];best_balanced=balanced[0]
  selected=dict(original=original,best_mean=best,best_balanced=best_balanced)
  for name,row in selected.items():write(OUT/'presets'/(name+'.json'),dict(name=name,**row['policy'],static_route=True))
  write(OUT/'summary.json',dict(state='complete',selected=selected,rounds=rounds,unique_policies=len(seen),videos=len(seen)*2,ranking=ranking,independent_validation=False));write(OUT/'progress.json',dict(state='complete',policies=len(seen),videos=len(seen)*2))
 finally:
  db.execute('INSERT OR REPLACE INTO meta VALUES ("stop","true")')
  for p,log in procs:p.wait();log.close()
  db.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--gpu',type=int);p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));a=p.parse_args();worker(a) if a.worker else run(a)
