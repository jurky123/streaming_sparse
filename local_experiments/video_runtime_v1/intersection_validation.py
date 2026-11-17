"""Frozen 5-prompt x 4-new-seed validation of intersection search and earlier scopes."""
import argparse,hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .weights import ROOT
from .scope_experiment import write,configure_scope
SEARCH=ROOT/'artifacts/video_runtime_v1/intersection_search_v1'
OLD=ROOT/'artifacts/video_runtime_v1/pset_scope_v1'
OUT=ROOT/'artifacts/video_runtime_v1/intersection_validation_v1'
CASES=['kitchen_cook','forest_run','portrait_turn','reef_fish','tennis_match']
SEEDS=[703,704,705,706]
METHODS=['original','search_mean','search_balanced','top_layers','top_heads','top_steps']

def policies():
 files=dict(original=OLD/'presets/intersection.json',search_mean=SEARCH/'presets/best_mean.json',search_balanced=SEARCH/'presets/best_balanced.json',**{n:OLD/'presets'/(n+'.json') for n in ['top_layers','top_heads','top_steps']})
 return {n:dict(json.loads(p.read_text()),name=n) for n,p in files.items()}, {n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
def database():
 OUT.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(OUT/'groups.sqlite',timeout=60,isolation_level=None)
 db.execute('CREATE TABLE IF NOT EXISTS groups (id INTEGER PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL DEFAULT "pending", worker INTEGER, error TEXT)');return db
@torch.no_grad()
def worker(a):
 from .deployment import current,configure,load_weights,make_runtime,identity
 from ..broad_common import Engine
 from ..rollout_suite import generate_chunk
 from ..stream_eval_common import native_vae,decode,save_video
 from ..high_sparse_experiment import prefix_quality
 import lpips
 torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=512;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=current(static_route=True,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg);model,reader,phi,B,weights=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 vae=native_vae();os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache');metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
 frozen=json.loads((OUT/'protocol.json').read_text());specs=frozen['policies'];db=database()
 def generate(emb,seed):
  rt.reset();zs=[generate_chunk(engine,emb,seed,c) for c in range(9)];torch.cuda.synchronize();return torch.cat(zs,1)
 try:
  emb=torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{CASES[0]}.pt',weights_only=True).cuda()
  # Warm every route shape used by the frozen methods: base, 3 heads, all heads.
  for spec in [specs['original'],dict(name='full')]:
   rt.sparse=True;configure_scope(rt,spec);generate(emb,SEEDS[0])
  write(OUT/'workers'/f'gpu{a.gpu}_identity.json',dict(weights=weights,deployment=identity(cfg),source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
  while True:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT id,spec FROM groups WHERE status="pending" ORDER BY id LIMIT 1').fetchone()
   if row:db.execute('UPDATE groups SET status="running",worker=? WHERE id=?',(a.gpu,row[0]))
   db.execute('COMMIT')
   if row is None:break
   group_id,text=row;group=json.loads(text);case=group['case'];seed=group['seed'];folder=OUT/'videos'/f'{case}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
   emb=torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{case}.pt',weights_only=True).cuda()
   order=METHODS[group_id%len(METHODS):]+METHODS[:group_id%len(METHODS)]
   write(OUT/'workers'/f'gpu{a.gpu}_status.json',dict(state='running',group=group,order=['dense']+order))
   try:
    reference=None
    for name in ['dense']+order:
     out=folder/name;out.mkdir(exist_ok=True)
     if (out/'result.json').exists():
      if name=='dense':reference=np.load(out/'pixels.npy',mmap_mode='r')
      continue
     rt.sparse=name!='dense';configure_scope(rt,dict(name='base') if name=='dense' else specs[name])
     torch.cuda.synchronize();start=time.perf_counter();z=generate(emb,seed);generation=time.perf_counter()-start
     assert torch.isfinite(z).all();counters=rt.counters.cpu().tolist();assert counters[2]==0
     if name!='dense':
      spec=specs[name];cells=len(spec.get('layers',range(1,30)))*len(spec.get('heads',range(12)))*len(spec.get('steps',range(4)));assert counters[1]==cells*504,(name,counters)
     torch.save(z.cpu(),out/'latents.pt');start=time.perf_counter();frames=decode(vae,z,105);torch.cuda.synchronize();decode_seconds=time.perf_counter()-start
     quality={}
     if name=='dense':reference=np.concatenate(frames);np.save(out/'pixels.npy',reference)
     else:quality=prefix_quality(frames,[reference],metric)
     save_video(out/'video.mp4',frames)
     write(out/'result.json',dict(case=case,seed=seed,method=name,quality=quality,generation_seconds=generation,decode_seconds=decode_seconds,counters=counters,policy=specs.get(name),gpu=a.gpu,order=['dense']+order,static_route=True,timing_scope='one generation per sample after full route-shape warmup; excludes loading/compilation/VAE/IO; all methods of each sample on same GPU; method order rotated across samples'))
     print(case,seed,name,generation,quality.get('lpips'),flush=True)
    db.execute('UPDATE groups SET status="done",error=NULL WHERE id=?',(group_id,))
   except BaseException as exc:db.execute('UPDATE groups SET status="failed",error=? WHERE id=?',(repr(exc),group_id));raise
  write(OUT/'workers'/f'gpu{a.gpu}_status.json',dict(state='done'))
 finally:rt.close();db.close()
def run(a):
 assert (SEARCH/'summary.json').exists();specs,digests=policies();db=database()
 protocol=dict(cases=CASES,seeds=SEEDS,methods=['dense']+METHODS,policies=specs,policy_source_sha256=digests,chunks=9,frames=105,resolution=[512,768],fps=16,history_pages=360,tile_batch=72,static_route=True,search_prompt='city_walk',search_seeds=[701,702],frozen_before_validation=True,note='prompts and seeds did not participate in this policy search; not a claim of being unseen by model training; no policy retuning after validation')
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
 else:write(OUT/'protocol.json',protocol)
 for i,(case,seed) in enumerate((c,s) for c in CASES for s in SEEDS):db.execute('INSERT OR IGNORE INTO groups(id,spec) VALUES (?,?)',(i,json.dumps(dict(case=case,seed=seed))))
 db.execute('UPDATE groups SET status="pending" WHERE status IN ("running","failed")')
 procs=[]
 for gpu in a.gpus:
  log=(OUT/f'gpu{gpu}.log').open('a');p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.intersection_validation','--worker','--gpu',str(gpu)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
 codes=[p.wait() for p,_ in procs]
 for _,f in procs:f.close()
 write(OUT/'status.json',dict(state='complete' if not any(codes) else 'failed',exit_codes=codes,groups=[dict(status=s,count=n) for s,n in db.execute('SELECT status,count(*) FROM groups GROUP BY status')]))
 if any(codes):raise RuntimeError(codes)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--gpu',type=int);p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));a=p.parse_args();worker(a) if a.worker else run(a)
