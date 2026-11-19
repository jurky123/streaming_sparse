import torch,json
from pathlib import Path
from ..review_v1.model import Model
from ..review_v1.ops import Config
from ..response_hard import balanced
from ..cache_grouping import reorder
from ..group_response.core import grouping as old_grouping
from .core import responses,grouping,staged_order,GROUPS
from .run import BASE,PRE,ROOT
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 m=Model(BASE);m.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);m.requires_grad_(False);m=m[15]
 torch.manual_seed(17);k=torch.randn(12,4608,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 key,vr=responses(m,k.float()[:,None],v.float()[:,None]);key=key.squeeze(1);vr=vr.squeeze(1)
 assert torch.equal(staged_order(key,vr,('K',)*5),balanced(key,[3,3,2,2,2]))
 assert torch.equal(staged_order(key,vr,('V',)*5),balanced(vr,[3,3,2,2,2]))
 mixed=staged_order(key,vr);prefix=balanced(key,[3,3,2]);suffix=balanced(reorder(vr,prefix).reshape(12*18,256,32),[2,2]);reference=prefix.reshape(12*18,256).gather(1,suffix).reshape(12,4608);assert torch.equal(mixed,reference)
 assert torch.equal(mixed.reshape(12,18,256).sort(-1).values,prefix.reshape(12,18,256).sort(-1).values)
 assert torch.equal(key+.1*m.group_gate.tanh()[:,None]*vr,m.group_features(k.float()[:,None],v.float()[:,None]).squeeze(1))
 for g in GROUPS:
  a=grouping(m,k,v,g);assert torch.equal(a[2].sort(-1).values,torch.arange(4608,device='cuda')[None].expand(12,-1))
  if g in ('G-ORDER','G-K','G-KVR'):assert all(torch.equal(x,y) for x,y in zip(a,old_grouping(m,k,v,g)))
 for a,b in [('G-K','G-K-joint'),('G-K3-V2','G-K3-V2-Kinner')]:assert torch.equal(grouping(m,k,v,a)[2].reshape(12,72,64).sort(-1).values,grouping(m,k,v,b)[2].reshape(12,72,64).sort(-1).values)
 ROOT.mkdir(exist_ok=True);(ROOT/'preflight.json').write_text(json.dumps(dict(passed=True,homogeneous_stages_match_original=True,first_three_K_parent_membership_preserved=True,mixed_matches_composed_original_balanced=True,inner_variant_preserves_page_membership=True,inner_feature_matches_original_joint=True,old_three_groups_bitwise_identical=True,token_bijection=True),indent=2));print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
