import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from .projected_route import ProjectedRoute
from .network import DualMeanRoute
from .projection import project
from ..opt5.probe import capture,timing
from ..learning import ActionHead
from ..common import atomic_json
from ..opt7.split_attention import attention
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=data['q'][:,:512];sk=data['sketch'];prepared=data['prepared'];step=q.new_tensor(3.,dtype=torch.float32)
 head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);router=SimpleNamespace(kind='K32-Set',beta=data['beta'],heads={'24_K32-Set':head})
 ref=DualMeanRoute();new=ProjectedRoute();rs=ref.begin(prepared,sk);ns=new.begin(prepared,sk)
 rmask=ref.run(24,q,sk,rs,step,router);nmask=new.run(24,q,sk,ns,step,router)
 r=attention(q,data['k'],data['v'],rmask,splits=2,stages=2);n=attention(q,data['k'],data['v'],nmask,splits=2,stages=2)
 rg,_=capture(lambda:ref.run(24,q,sk,rs,step,router));ng,_=capture(lambda:new.run(24,q,sk,ns,step,router));timing(rg,100);timing(ng,100)
 times=[]
 for i in range(6):
  if i%2:nt=timing(ng,100);rt=timing(rg,100)
  else:rt=timing(rg,100);nt=timing(ng,100)
  times.append((rt,nt))
 rp=torch.bmm(q.float(),prepared[0]).half();np=project(q,prepared[0]);nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
 result=dict(reference_ms=sum(x[0]for x in times)/6,new_ms=sum(x[1]for x in times)/6,changed_queries=int((rmask!=nmask).any(-1).sum()),head_queries=q.shape[0]*q.shape[1],projection_nmse=nmse(np,rp),attention_nmse=nmse(n,r));print(result);atomic_json(root/'optimization8/projection_probe.json',result)
if __name__=='__main__':main()
