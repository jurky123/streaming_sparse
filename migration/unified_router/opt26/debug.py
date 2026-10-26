"""Locate first selected-page drift on an eager closed-loop trajectory."""
import json,types,torch,triton as tr
from pathlib import Path
from .setup import ROOT
from . import pipeline as runtime
from . import route as new
from ..opt25.rollout import trajectory
from ..opt25.runtime import RuntimeNetwork
from ..opt25.network import FusedNetwork
from ..opt25.pipeline import PipelineState
from ..opt22.inference import load
from ..opt23.route import scores
from ..opt19.tile import select
from ..opt8.projection import project
from ..opt17.route import _landmark,_coarse,_top
from ..common import atomic_json,paired_seed
@torch.no_grad()
def main():
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 net,m=load();net.__class__=FusedNetwork;net.history_chunks=3;PipelineState.sparge=True;runtime.install()
 ds=ReplayDataset('val',require_complete=True);pid=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text())['prompt_ids'][4]
 sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 original_block=net._block;context={}
 def block(*args,**kw):
  context.update(layer=args[1],start=args[9],step=args[11]);return original_block(*args,**kw)
 net._block=block
 def eager(self,latent,embeds,timestep,cache,start,**kw):
  if self.needs_condition:self.condition(embeds)
  self.runtime_timestep=None;self.runtime_time_features=self.time_features(timestep);self.runtime_full_step=True
  return RuntimeNetwork.forward_chunk(self,latent,embeds,timestep,cache,start,**kw)
 net.forward_chunk=types.MethodType(eager,net)
 original=new.optimized
 def observed(q,state,seg):
  ids=original(q,state,seg);f,c=scores(q,state,.35,8,seg);b=state['key'].shape[1];old=select(f,c,b,32,.2)
  if not torch.equal(ids,old):
   qq=project(q,state['basis']);cs=new.coarse(qq,state,8,128 if b>1024 else 64);nc=new.radix_top(cs,c.shape[-1],8,4) if b>512 else None
   h,n,_=q.shape;g=tr.cdiv(n,32)
   if nc is None:
    nc=torch.empty_like(c);_top[(h*g,)](cs,nc,h*g,b,c.shape[-1],tr.next_power_of_2(b),num_warps=4)
   lm=torch.empty((h,g,8,64),device=q.device,dtype=torch.float16);_landmark[(g,h)](qq,lm,n,g,8,num_warps=4);co=torch.empty_like(cs);off=0
   for kk,lc in state['parts']:
    np,p=lc.shape[1:];gp=128//p;_coarse[(g,tr.cdiv(np,gp),h)](lm,kk,lc,co,n,b,g,off,np,p,gp,8,num_warps=4,num_stages=1);off+=np
   nf=new.refine(qq,state,c,seg,8,2,False);nf4=new.refine(qq,state,c,seg,8,4,False)
   info=dict(context,changed_ids=int((ids!=old).sum()),coarse_exact=torch.equal(cs,co),coarse_max_abs=float((cs-co).abs().max()),candidates_exact=torch.equal(c,nc),fine2_exact=torch.equal(f,nf),fine4_exact=torch.equal(f,nf4),fine2_max_abs=float((f-nf).abs().max()),fine4_max_abs=float((f-nf4).abs().max()))
   atomic_json(ROOT/'first_drift.json',info);torch.save(dict(q=q.cpu(),state={k:([tuple(x.cpu()for x in t)for t in v]if k=='parts'else v.cpu())for k,v in state.items()},seg=seg,info=info),ROOT/'first_drift.pt');print(info,flush=True);raise RuntimeError('First drift captured')
  return ids
 runtime.optimized=observed;net.select('Tile20_after')
 with sdpa_backend():trajectory(net,embeds,pid,0,True,height=720,width=1280)
 print('No drift in eager checks',flush=True)
if __name__=='__main__':main()
