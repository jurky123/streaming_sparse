"""Numerical, budget and cache-group permutation checks on real operands."""
import json
from pathlib import Path
import torch
from .cache_sparse_prototype import sparse_forward
from .cache_grouping import history_order,reorder,page_stats
from .screen_cache_research import evaluate


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/cache_research_20261029')
    folder=root/'capture/sailboat/qkv'
    hist=torch.load(folder/'c20_l14_history.pt',weights_only=True)
    curr=torch.load(folder/'c20_l14_s2.pt',weights_only=True)
    kh,vh,kc,vc,q=[x.cuda() for x in [hist['k'],hist['v'],curr['k'],curr['v'],curr['q']]]
    def blhd(x):return x.transpose(0,1).unsqueeze(0).contiguous()
    kb=blhd(torch.cat((kh,kc),1));vb=blhd(torch.cat((vh,vc),1));qb=blhd(q)
    order=history_order(kh,vh,1.)
    full,budget=sparse_forward(qb,kb,vb,order,False,1.)
    dense=torch.nn.functional.scaled_dot_product_attention(q,torch.cat((kh,kc),1),torch.cat((vh,vc),1))
    full_nmse=float((full[0].transpose(0,1).float()-dense.float()).square().sum()/dense.float().square().sum())
    assert full_nmse<1e-4,full_nmse
    records=[dict(check='full_budget_and_permutation',nmse=full_nmse)]
    ids=torch.linspace(0,q.shape[1]-1,64,device='cuda').long()
    route=q.float().reshape(3,-1,64,128).mean(2)[:,ids//64]
    k=reorder(kh,order);v=reorder(vh,order)
    for mode in ['mass','var_tail']:
        actual,budget=sparse_forward(qb,kb,vb,order,mode=='var_tail')
        _,mask,expected,ref,_=evaluate(q.float()[:,ids],route,k,v,kc,vc,page_stats(k,v),mode)
        actual=actual[0].transpose(0,1)[:,ids].float()
        nmse=float((actual-expected).square().sum()/expected.square().sum())
        assert nmse<1e-4,(mode,nmse)
        assert budget['exact_history_tokens']+budget['summary_equivalent_tokens']<=budget['history_budget_tokens']
        records.append(dict(check=mode+'_matches_fp32_formula',nmse=nmse,**budget))
    # Regression: selected Gaussian estimates and dropped exact logits must
    # never set the stabilizer for terms that are not actually in the sum.
    sk=torch.zeros((1,256,128),device='cuda');sv=torch.arange(1,5,device='cuda').float().repeat_interleave(64)[None,:,None].expand(1,256,128)
    sq=torch.zeros((1,8,128),device='cuda');sq[:,:,0]=128**.5
    sk[0,:64:2,0]=40;sk[0,1:64:2,0]=-40
    sc=torch.zeros((1,64,128),device='cuda')
    _,mask,pred,_,_=evaluate(sq,sq,sk,sv,sc,sc,page_stats(sk,sv),'var_tail',.8)
    assert torch.isfinite(pred).all() and float((pred-1).abs().max())<1e-5
    records.append(dict(check='selected_extreme_gaussian_does_not_underflow',max_error=float((pred-1).abs().max())))
    sk.zero_();sk[:,0:64,0]=400
    _,mask,pred,_,_=evaluate(sq,-sq,sk,sv,sc,sc,page_stats(sk,sv),'mass',.8)
    logits=(sq@torch.cat((sk,sc),1).transpose(1,2))/128**.5
    valid=torch.cat((mask.repeat_interleave(64,-1),torch.ones((1,8,64),device='cuda',dtype=torch.bool)),-1)
    expected=logits.masked_fill(~valid,-torch.inf).softmax(-1)@torch.cat((sv,sc),1)
    assert torch.isfinite(pred).all() and float((pred-expected).abs().max())<1e-5
    records.append(dict(check='dropped_extreme_logit_does_not_underflow',max_error=float((pred-expected).abs().max())))
    (root/'prototype_checks.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
