"""Check ablation controls and bitwise full-branch rollout identity."""
import torch,json
from pathlib import Path
from .review_v1.ablation import AblationRuntime,change_summary,mass_scores
from .review_v1.ops import Summary
from .review_v1.model import Model
from .broad_common import Engine,ROOT as DATA
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,digest
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    basis=torch.zeros(2,128,8,device='cuda');basis[:,:8]=torch.eye(8,device='cuda')
    s=Summary(torch.randn(2,5,4,128,device='cuda').bfloat16(),torch.randn(2,5,128,device='cuda').bfloat16(),torch.randn(2,5,3,8,device='cuda').bfloat16(),torch.zeros(2,5,4,12,device='cuda').bfloat16());before=s.detail.clone()
    changed=change_summary(s,basis,torch.zeros(2,128,device='cuda'),'no_detail')
    assert not changed.detail.any() and torch.equal(s.detail,before) and changed.keys is s.keys
    changed=change_summary(s,basis,torch.zeros(2,128,device='cuda'),'projected_mean')
    torch.testing.assert_close(changed.mean[:,:,:8],s.mean[:,:,:8].float(),rtol=0,atol=0);assert not changed.mean[:,:,8:].any()
    scores=mass_scores(torch.zeros(2,32,128,device='cuda'),s,2)
    torch.testing.assert_close(scores,torch.full_like(scores,1/5))
    engine=Engine();base='artifacts/four_gaps_20261031/shared/epoch1.pt';model=Model(base)
    model.load_state_dict(torch.load('artifacts/review_v1_20261102/route_expanded/model.pt',map_location='cuda',weights_only=False)['model'])
    saved=torch.load(base,map_location='cpu',weights_only=False)['model'];centers={l:saved[f'layers.{l}.center'].cuda() for l in range(1,30)}
    rt=AblationRuntime(engine,model,'full',centers);emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda();rows=[]
    try:
        for c in range(3):
            out=generate_chunk(engine,emb,101,c).cpu();ref=torch.load(f'artifacts/review_v1_20261102/video/horse_field_s101/candidate/chunk{c:02d}.pt',weights_only=True)
            assert torch.equal(out,ref),float((out.float()-ref.float()).abs().max())
            rows.append(dict(chunk=c,bitwise_equal=True))
        # The no-grouping branch must still be a bijection and retain paired KV.
        rt.variant='no_grouping';k=torch.randn(1,4608,12,128,device='cuda').bfloat16();v=-k
        kg,vg,order=rt.group(1,k,v)
        torch.testing.assert_close(order.sort(-1).values,torch.arange(4608,device='cuda')[None].expand(12,-1),rtol=0,atol=0)
        torch.testing.assert_close(kg,-vg,rtol=0,atol=0)
    finally:rt.close()
    atomic(Path('artifacts/review_v1_20261102/ablation/smoke.json'),dict(passed=True,rows=rows,code=digest('local_experiments/review_v1/ablation.py')))
    print('Ablation controls and bitwise rollout identity passed',flush=True)
if __name__=='__main__':main()
