import torch,json
from pathlib import Path
from ..review_v1.model import Model
from ..review_v1.runtime import GroupingOps,PageStore
from ..review_v1.ops import SummaryOps,Config
from ..cache_grouping import reorder
from .core import grouping,regroup,responses,GROUPS
from .local import dense
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m.load_state_dict(torch.load('artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt',map_location='cuda',weights_only=False)['model']);m.requires_grad_(False);m=m[15]
 torch.manual_seed(17);k=torch.randn(1,4608,12,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 ref=GroupingOps.group(m,k,v,Config());test=grouping(m,k[0].transpose(0,1),v[0].transpose(0,1),'G-KVR');assert all(torch.equal(a,b) for a,b in zip(ref,test))
 key,vr=responses(m,k[0].transpose(0,1).float()[:,None],v[0].transpose(0,1).float()[:,None]);assert torch.equal(key+.1*m.group_gate.tanh()[:,None,None]*vr,m.group_features(k[0].transpose(0,1).float()[:,None],v[0].transpose(0,1).float()[:,None]))
 cache=dict(k=torch.empty(1,410*64,12,128,device='cuda',dtype=torch.bfloat16));cache['v']=torch.empty_like(cache['k']);store=PageStore(cache,Config());rows=[]
 for c in range(8):
  kg,vg,order=ref;store.commit(kg,vg,SummaryOps.build(m,kg,vg),order,c*4608)
  if c+1 not in [2,5,8]:continue
  p=store.n;kh=cache['k'][0,:p*64].transpose(0,1).reshape(12,p,64,128);vh=cache['v'][0,:p*64].transpose(0,1).reshape_as(kh);q=torch.randn(12,128,128,device='cuda',dtype=torch.bfloat16);target,z=dense(q,kh,vh)
  for group in GROUPS:
   kk,vv,ids,check=regroup(m,kh,vh,store.token_ids[:,:p],group);o,zz=dense(q,kk,vv);err=float((o-target).square().sum()/target.square().sum());assert err<2e-10
   rows.append(dict(chunk=c+1,group=group,dense_nmse=err,**check))
 out=Path('/tmp/group_response_results');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(dict(passed=True,original_kvr_bitwise=True,pool_features_unmodified=True,rows=rows),indent=2));print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
