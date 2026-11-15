import argparse,torch
from pathlib import Path
from ..broad_common import Engine,ROOT as DATA
from ..review_v1.model import Model
from ..review_v1.ops import SummaryOps
from ..rollout_suite import generate_chunk
from ..group_response.core import Runtime
from ..group_response.run import BASE,PRE
from ..learned_cache.io import atomic
ROOT=Path('/tmp/e11_results')
class Capture:
 def __init__(self,engine,model,case):self.engine=engine;self.model=model;self.case=case
 def __call__(self,l,q,kh,vh,store,start):
  c=self.engine.current['chunk'];step=self.engine.current['step']
  if c not in (2,5) or step!=0 or l not in (1,15,29):return
  t=(17*l+13*c+101)%72;h=(l+c)%12;p=store.n
  qs=q[0].transpose(0,1)[:,t*64:(t+1)*64];k=kh[0].transpose(0,1).reshape(12,-1,64,128);v=vh[0].transpose(0,1).reshape_as(k)
  old=torch.load(Path('/tmp/group_hierarchy_results/qkv')/f'{self.case}_s101'/f'c{c}_l{l}_s0.pt',weights_only=True)
  assert torch.equal(qs.cpu(),old['q'][:,:64]) and torch.equal(k.cpu(),old['k']) and torch.equal(v.cpu(),old['v'])
  cur=SummaryOps.build(self.model[l],k[:,p:],v[:,p:],cheap=True);hist=store.summary.slice(0,p)
  summary={n:torch.cat((getattr(hist,n),getattr(cur,n)),1).cpu() for n in ('keys','mean','detail','stats')}
  module=self.engine.wrapper.model.blocks[l].self_attn.o
  data=dict(q=qs.cpu(),k=k.cpu(),v=v.cpu(),summary=summary,basis=self.model[l].basis.detach().cpu(),theta=self.model[l].theta.detach().cpu(),ids=store.token_ids[:,:p].cpu(),own=p+t,head=h,tile=t,case=self.case,seed=101,chunk=c,layer=l,step=step,history_pages=p,output_weight=module.weight.detach().cpu(),output_bias=None if module.bias is None else module.bias.detach().cpu(),raw_matches_previous=True,summary_source='actual BF16 PageStore history plus original cheap current build')
  atomic(ROOT/'snapshots'/f'{self.case}_s101_c{c}_l{l}_s0.pt',data,True);print('CAPTURE',self.case,c,l,flush=True)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 engine=Engine();engine.wrapper.requires_grad_(False);model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
 rt=Runtime(engine,model,'G-KVR','L3',Capture(engine,model,a.case));emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda();zs=[]
 for c in range(6):zs.append(generate_chunk(engine,emb,101,c).cpu());print('CHUNK',c,flush=True)
 z=torch.cat(zs,1);ref=torch.load(Path('/tmp/group_response_results/video')/f'{a.case}_s101/G-KVR/L3/latents.pt',weights_only=True)
 assert torch.equal(z,ref[:,:z.shape[1]])
 atomic(ROOT/'capture_checks'/f'{a.case}.json',dict(prefix_latents_bitwise_equal=True,events=6,raw_qkv_equal=True,actual_cached_summary=True));rt.close()
if __name__=='__main__':main()
