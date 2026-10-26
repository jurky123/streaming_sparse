import torch
from .grouping import Tree,labels,permutation,gather,restore,fit_tree
from .route import prepare,route,_top
from ..common import atomic_json
from .setup import ROOT,CAPTURE
from ..fast_route import score_cached
from ..opt9.segmented_attention import attention

@torch.no_grad()
def main():
    torch.manual_seed(1601);torch.set_num_threads(4);rows=[]
    for n in [1,31,32,33,127,129,4680]:
        for c in [8,64,128]:
            for constant in [False,True]:
                lab=torch.zeros((2,n),device='cuda',dtype=torch.int32) if constant else torch.randint(c,(2,n),device='cuda',dtype=torch.int32)
                ix=permutation(lab,c)
                expected=lab.argsort(dim=-1,stable=True)
                assert torch.equal(ix.long(),expected),(n,c,constant)
                x=torch.randn(2,n,128,device='cuda');assert torch.equal(restore(gather(x,ix),ix),x)
                rows.append(dict(test='stable_permutation_roundtrip',n=n,buckets=c,constant=constant))
    x=torch.randn(2,129,128,device='cuda');op=torch.randn(2,257,128,device='cuda')
    tree=fit_tree(x,op,depth=3)
    lab=labels(x,tree);assert (lab>=0).all() and (lab<8).all()
    h=torch.arange(2,device='cuda')[:,None];node=torch.zeros((2,129),device='cuda',dtype=torch.long)
    for _ in range(3):node=node*2+1+((x*tree.weight[h,node]).sum(-1)>tree.threshold[h,node])
    # Calibrated median ties may round differently between reductions; verify away from ties with fresh data.
    fresh=torch.randn_like(x);lab=labels(fresh,tree);node.zero_()
    for _ in range(3):node=node*2+1+((fresh*tree.weight[h,node]).sum(-1)>tree.threshold[h,node])
    assert torch.equal(lab.long(),node-7)
    rows.append(dict(test='tree_traversal_reference',passed=True))
    import triton as tr
    for b in [23,74,148,296]:
        for kind in ['constant','negative','ties']:
            s=torch.zeros((5,b),device='cuda') if kind=='constant' else -torch.rand((5,b),device='cuda') if kind=='negative' else torch.arange(b,device='cuda').float().remainder(7)[None].expand(5,-1).contiguous()
            keep=max(1,b//6);ix=torch.empty((5,keep),device='cuda',dtype=torch.int32)
            _top[(5,)](s,ix,5,b,keep,tr.next_power_of_2(b),num_warps=4)
            expected=s.argsort(dim=-1,descending=True,stable=True)[:,:keep].sort(-1).values
            assert torch.equal(ix.long(),expected),(b,kind)
            rows.append(dict(test='topk_ties_negative_and_padding',pages=b,kind=kind))
    data=torch.load(CAPTURE/'layer24_chunk1.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:65];state=prepare(data['prepared'])
    for fraction in [.15,.5,1.]:
        mask,aux=route(q,state,.15,fraction,True)
        assert (mask.sum(-1)==aux['keep_pages']).all()
        ids=aux['candidates'];s=aux['scores'];assert torch.isfinite(s).all()
        # Independent FP32 logsumexp reference on quantized projected keys/queries.
        from ..opt8.projection import project
        qq=project(q,state['basis']).float();kk=state['key'].float()
        all_scores=(torch.einsum('hnd,hbpd->hnbp',qq,kk)/128**.5+state['logcount'][:,None]).logsumexp(-1)
        pos=torch.arange(q.shape[1],device=q.device)//32
        selected=all_scores.gather(-1,ids[:,pos].long())
        assert (selected-s).abs().max()<.025,float((selected-s).abs().max())
        picked=ids[:,pos].long().gather(-1,s.topk(aux['keep_pages'],-1).indices)
        expected=torch.zeros_like(mask).scatter_(-1,picked,True)
        assert torch.equal(mask,expected)
        rows.append(dict(test='route_score_and_independent_topk',fraction=fraction,max_abs=float((selected-s).abs().max())))
    # Attention correctness with arbitrary permutations, split softmax, and tail rows/pages.
    k,v=data['k'],data['v'];nh=data['sketch'].history_tokens
    ix=torch.randperm(q.shape[1],device=q.device)[None].expand(q.shape[0],-1)
    qq=gather(q,ix);mask=route(qq,state,.15,.5)
    y=attention(qq,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],mask,pack_v=True,fuse_pack=True)
    costs=data['sketch'].costs.long();tm=mask.repeat_interleave(costs,-1)
    ref=torch.nn.functional.scaled_dot_product_attention(qq[None],k[None],v[None],attn_mask=tm[None])[0]
    nmse=float((y.float()-ref.float()).square().sum()/ref.float().square().sum())
    assert nmse<.003,nmse
    rows.append(dict(test='int8_attention_vs_bf16_same_mask',nmse=nmse))
    indices=[]
    for start in range(0,k.shape[1],4680):
        end=min(start+4680,k.shape[1]);indices.append(torch.rand((k.shape[0],end-start),device=k.device).argsort(-1)+start)
    ki=torch.cat(indices,1);kk=gather(k,ki);vv=gather(v,ki)
    y=attention(qq,kk[:,:nh],vv[:,:nh],kk[:,nh:],vv[:,nh:],mask,pack_v=True,fuse_pack=True)
    ref=torch.nn.functional.scaled_dot_product_attention(qq[None],kk[None],vv[None],attn_mask=tm[None])[0]
    nmse=float((y.float()-ref.float()).square().sum()/ref.float().square().sum())
    assert nmse<.003,nmse
    rows.append(dict(test='bilateral_permutation_same_mask_attention',nmse=nmse))
    atomic_json(ROOT/'validation.json',dict(total=len(rows),cases=rows));print('PASSED',len(rows),flush=True)

if __name__=='__main__':main()
