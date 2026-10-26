import argparse,statistics,itertools
from pathlib import Path
from types import SimpleNamespace
import torch
from ..opt9.network import PackedRoute
from .diag_route import DiagRoute
from ..opt9.select import fused_select
from ..opt5.probe import capture,timing
from ..learning import ActionHead
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=d['q'][:,:512];sk=d['sketch'];ref=PackedRoute(batch=512);state=ref.begin(d['prepared'],sk)
 head=ActionHead(d['head']).cuda();head.load_state_dict(d['head']);head.eval();router=SimpleNamespace(kind='K32-Set',beta=d['beta'],heads={'24_K32-Set':head});step=q.new_tensor(3.,dtype=torch.float32)
 rs,rr=ref.score(q,state);mask,rem,add=fused_select(rs,sk.origins,d['beta'],'packed32');rst,rcv=ref.stats(q,sk,rs,mask,rem,add,rr,state);ry=ref.run(24,q,sk,state,step,router)
 rg,_=capture(lambda:ref.run(24,q,sk,state,step,router));timing(rg,500);rows=[]
 for qm,gp in list(itertools.product([32,64,128],[1,2,4,8]))[a.rank::a.world]:
  try:
   route=DiagRoute(batch=512);route.qm=qm;route.page_group=gp;state['norms']=route.norms(state['padded_value']);ns,nr=route.score(q,state);st,cv=route.stats(q,sk,ns,mask,rem,add,nr,state);y=route.run(24,q,sk,state,step,router)
   ng,_=capture(lambda:route.run(24,q,sk,state,step,router));timing(ng,500);pairs=[]
   for i in range(6):
    if i%2:nt=timing(ng,100);rt=timing(rg,100)
    else:rt=timing(rg,100);nt=timing(ng,100)
    pairs.append((rt,nt))
   comps={}
   for name,old,new in [('score',lambda:ref.score(q,state),lambda:route.score(q,state)),('stats',lambda:ref.stats(q,sk,rs,mask,rem,add,rr,state),lambda:route.stats(q,sk,ns,mask,rem,add,nr,state))]:
    g0,_=capture(old);g1,_=capture(new);comps[name]=dict(old=timing(g0,200),new=timing(g1,200))
   nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
   row=dict(qm=qm,gp=gp,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),score_exact=torch.equal(ns,rs),weights_exact=torch.equal(nr[0],rr[0]),energy_exact=torch.equal(nr[1],rr[1]),stats_nmse=nmse(st[:,:73],rst[:,:73]),cv_nmse=nmse(cv,rcv),changed_queries=int((y!=ry).any(-1).sum()),components=comps,samples=pairs)
  except Exception as e:row=dict(qm=qm,gp=gp,error=str(e))
  rows.append(row);atomic_json(root/f'optimization10/diag_probe{a.rank}.json',rows);print(row,flush=True)
if __name__=='__main__':main()
