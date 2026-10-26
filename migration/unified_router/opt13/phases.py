"""Separate instrumented pass: CUDA-event spans over the complete trajectory.
Nested component spans are not additive with top-level forward spans.
These spans may include host launch gaps; kernel-only attribution is separate.
"""
import os,json,time
from collections import defaultdict
from pathlib import Path
import torch
from ..common import atomic_json,paired_seed
from .. import fast_network
from .network import load
from .setup import ROOT

@torch.no_grad()
def main():
 paired_seed(0);variant=os.environ.get('SLIMAV_OPT13','Reference');net,m=load(ROOT);root=ROOT;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 ds=ReplayDataset('val',require_complete=True);pid=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text())['prompt_ids'][0];sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 with sdpa_backend():
  net.set_mode('K32-Set');chunks,_=trajectory(net,embeds,pid,0,True);expected=torch.cat(chunks,1);del chunks
  events=[];current={}
  def wrap(name,fn):
   def run(*args,**kwargs):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();result=fn(*args,**kwargs);b.record();events.append((name,dict(current),a,b));return result
   return run
  for name in ['make_part','make_build','prepare','graph_route']:setattr(net,name,wrap(name,getattr(net,name)))
  fast_network.sparse_attention=wrap('sparse_attention',fast_network.sparse_attention)
  forward=net.forward_chunk
  def measured(*args,**kwargs):
   current.update(start=args[4],step=kwargs.get('step',0),sparse=kwargs.get('sparse',True))
   name='history_rebuild'if not current['sparse']else('first_chunk_dense'if current['start']==0 else'later_chunks_forward')
   return wrap(name,forward)(*args,**kwargs)
  net.forward_chunk=measured;net.set_mode('K32-Set');torch.cuda.synchronize();start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,True);torch.cuda.synchronize();elapsed=time.perf_counter()-start;assert torch.equal(torch.cat(chunks,1),expected)
 rows=[dict(name=n,**tag,milliseconds=a.elapsed_time(b))for n,tag,a,b in events];groups=defaultdict(list)
 for row in rows:groups[row['name']].append(row['milliseconds'])
 result=dict(variant=variant,prompt=pid,wall_seconds=elapsed,spans={n:dict(calls=len(v),milliseconds=sum(v))for n,v in groups.items()},rows=rows,repeat_identical=True,scope='Instrumented CUDA event spans; nested route/prototype/attention spans are inside later_chunks_forward; spans can include CPU launch gaps; not kernel-only time; primary benchmark remains uninstrumented')
 atomic_json(root/f'phases_{variant}.json',result);print(result['spans'],flush=True)
if __name__=='__main__':main()
