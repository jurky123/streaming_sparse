"""Separate timing and event-based phase/module profiling of frozen Tile20."""
import json,time,statistics,collections,argparse
from pathlib import Path
import torch
from .setup import ROOT,PARENT
from ..common import atomic_json,paired_seed,update

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--optimized',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from ..opt20.network import load,TileGraph
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 out=ROOT/('profile_optimized' if a.optimized else 'profile_baseline');out.mkdir(exist_ok=True);update(out/'status.json',status='loading')
 net,m=load();ds=ReplayDataset('val',require_complete=True);pid=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text())['prompt_ids'][0]
 sample=ds.load(next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 if a.optimized:
  from .network import configure
  from .rollout import trajectory
  configure(net)
 def run(name):net.select(name);return torch.cat(trajectory(net,embeds,pid,0,name!='Dense')[0],1)
 with sdpa_backend():
  references={};timings=collections.defaultdict(list)
  for name in ['Dense','Tile20_after']:
   update(out/'status.json',status='warmup',variant=name);references[name]=run(name)
  for rep in range(4):
   for name in (['Dense','Tile20_after'] if rep%2==0 else ['Tile20_after','Dense']):
    torch.cuda.synchronize();t=time.perf_counter();y=run(name);torch.cuda.synchronize();timings[name].append(time.perf_counter()-t);assert torch.equal(y,references[name])
  atomic_json(out/'timings.json',dict(samples=dict(timings),median={n:statistics.median(v) for n,v in timings.items()}))
  events=[];active=[False];phase=[''];original=net.forward_chunk
  def wrap(name,fn):
   def f(*args,**kw):
    if not active[0]:return fn(*args,**kw)
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();t=time.perf_counter();value=fn(*args,**kw);cpu=time.perf_counter()-t;end.record();events.append((phase[0],name,begin,end,cpu));return value
   return f
  def forward(*args,**kw):
   from streaming_sparse.v08.common import protocol
   old=phase[0];phase[0]='history' if float(args[2])==protocol()['context_noise'] else 'denoise'
   value=wrap('forward_chunk',original)(*args,**kw);phase[0]=old;return value
  net.forward_chunk=forward
  for block in net.model.blocks:
   for name,module in [('q',block.self_attn.q),('k',block.self_attn.k),('v',block.self_attn.v),('output_projection',block.self_attn.o),('cross_attention',block.cross_attn),('ffn',block.ffn)]:module.forward=wrap(name,module.forward)
  TileGraph.refresh_history=wrap('sparse_history_prepare',TileGraph.refresh_history)
  TileGraph.__call__=wrap('sparse_pipeline',TileGraph.__call__)
  for name in ['Dense','Tile20_after']:
   net.select(name);events.clear();active[0]=True;update(out/'status.json',status='profiling',variant=name)
   chunks,_=trajectory(net,embeds,pid,0,name!='Dense');torch.cuda.synchronize();active[0]=False
   assert torch.equal(torch.cat(chunks,1),references[name])
   sums=collections.defaultdict(lambda:dict(calls=0,cuda_ms=0.,cpu_ms=0.))
   for ph,label,b,e,cpu in events:
    row=sums[ph+'/'+label];row['calls']+=1;row['cuda_ms']+=b.elapsed_time(e);row['cpu_ms']+=cpu*1000
   atomic_json(out/(name+'.json'),dict(modules=dict(sums),note='CUDA event elapsed intervals include CPU launch gaps; nested modules overlap, do not sum indiscriminately.',bit_identical=True))
   print(name,json.dumps(dict(sums)),flush=True)
 update(out/'status.json',status='completed')
if __name__=='__main__':main()
