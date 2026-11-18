"""Real full-window block contracts and alternating warmed eager/graph timings."""
import argparse,json,statistics,time
from pathlib import Path
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .block_graph import BlockGraphs
from .weights import ROOT
from ..broad_common import Engine
from ..rollout_suite import generate_chunk

@torch.no_grad()
def run(a):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 cfg=config(a.tier,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 z=torch.load(ROOT/f'artifacts/video_runtime_v1/group_softmax_v1/video/{a.tier}_new/latents.pt',weights_only=True).cuda()
 for c in range(8):engine.forward(z[:,c*3:c*3+3],emb,c,4)
 bg=BlockGraphs(engine,rt,layers=[5]);rows=[];mixed=[]
 def checked(x,**kwargs):
  if engine.current['step']==4:return bg.forward(5,x,kwargs)
  ref=bg.original[5](x,**kwargs).clone();bg.forward(5,x,kwargs)
  counts=rt.counters.clone();out=bg.forward(5,x,kwargs).clone();graph_counts=rt.counters-counts
  counts=rt.counters.clone();eager=bg.original[5](x,**kwargs).clone();eager_counts=rt.counters-counts
  row=dict(step=engine.current['step'],bitwise=torch.equal(ref,out),repeat_bitwise=torch.equal(eager,out),max_abs=(ref-out).abs().max().item(),counters_bitwise=torch.equal(graph_counts,eager_counts))
  if engine.current['step']<2:
   funcs=[lambda:bg.original[5](x,**kwargs),lambda:bg.forward(5,x,kwargs)];times=[[],[]]
   for fn in funcs:
    for _ in range(5):fn()
   for r in range(20):
    for i in ([0,1] if r%2==0 else [1,0]):
     torch.cuda.synchronize();start=time.perf_counter()
     for _ in range(5):funcs[i]()
     torch.cuda.synchronize();times[i].append((time.perf_counter()-start)*200)
   row.update(eager_ms=statistics.median(times[0]),graph_ms=statistics.median(times[1]),samples_ms=times)
   row['speedup']=row['eager_ms']/row['graph_ms']
  if a.mixed_scope and engine.current['step']==0:
   saved=(rt.full_pset,rt.pset_head_ids)
   try:
    for name,enabled,head_ids in [('base',False,None),('subset',True,[4,3,9]),('all',True,None),('other_subset',True,[1,2,7]),('base_again',False,None)]:
     rt.full_pset=enabled;rt.pset_head_ids=None if head_ids is None else torch.tensor(head_ids,device=x.device)
     expected=bg.original[5](x,**kwargs).clone();actual=bg.forward(5,x,kwargs).clone()
     mixed.append(dict(name=name,bitwise=torch.equal(expected,actual),max_abs=(expected-actual).abs().max().item()))
   finally:rt.full_pset,rt.pset_head_ids=saved
   cache=kwargs['crossattn_cache'];saved_cross=(cache['k'],cache['v'])
   try:
    for name,change in [('cross_reallocated',False),('cross_changed',True)]:
     cache['k']=saved_cross[0].clone();cache['v']=saved_cross[1].clone()
     if change:cache['v'].add_(.01)
     expected=bg.original[5](x,**kwargs).clone();actual=bg.forward(5,x,kwargs).clone()
     mixed.append(dict(name=name,bitwise=torch.equal(expected,actual),max_abs=(expected-actual).abs().max().item()))
   finally:cache['k'],cache['v']=saved_cross
  rows.append(row);print(json.dumps(row),flush=True)
  return out
 engine.wrapper.model.blocks[5].forward=checked
 try:
  generate_chunk(engine,emb,701,8)
  result=dict(tier=a.tier,layer=5,chunk=8,pages=432,rows=rows,mixed_scope=mixed,graphs=len(bg.entries),passed=all(r['bitwise'] for r in mixed) and all(r['bitwise'] and r['repeat_bitwise'] and r['counters_bitwise'] for r in rows))
  Path(a.output).write_text(json.dumps(result,indent=2))
 finally:bg.close();rt.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mixed-scope',action='store_true');p.add_argument('--tier',required=True);p.add_argument('--output',required=True);run(p.parse_args())
