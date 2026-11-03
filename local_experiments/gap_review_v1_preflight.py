"""Small GPU contracts before any real-model training; no timing acceptance."""
import json,torch
from pathlib import Path
from .review_v1.ops import SummaryOps,AttentionOps,RouteOps,Config
from .review_v1.runtime import SelectedAttention,PageStore,GroupingOps
from .review_v1.model import Model
from .learned_cache.io import atomic,digest

def main():
    torch.set_num_threads(4);torch.manual_seed(192);torch.backends.cuda.matmul.allow_tf32=False
    model=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m=model[1];m.freeze_stage('summary')
    k=torch.randn(12,5,64,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)+2;q=torch.randn(12,64,128,device='cuda',dtype=torch.bfloat16)
    s=SummaryOps.build(m,k,v);mask=torch.zeros(12,64,5,device='cuda',dtype=torch.bool);mask[:,:,1]=True;mask[:,:,3]=True
    reference,_=AttentionOps.evaluate(q,k,v,s,m.basis,mask)
    selection=dict(mask=mask[:,:1],ids=torch.tensor([1,3,-123456],device='cuda').expand(12,1,3),counts=torch.full((12,1),2,device='cuda',dtype=torch.int32))
    actual=SelectedAttention.run(q,k,v,s,m.basis,selection)
    batched=SelectedAttention.run(q,k,v,s,m.basis,selection|dict(fixed_count=2))
    torch.testing.assert_close(actual,batched,atol=.016,rtol=.004)
    error=float((actual.float()-reference).square().sum()/reference.square().sum())
    assert error<1e-5,error
    target,z=AttentionOps.evaluate(q,k,v,s,m.basis,torch.ones_like(mask))
    dense=torch.nn.functional.scaled_dot_product_attention(q.float(),k.flatten(1,2).float(),v.flatten(1,2).float())
    torch.testing.assert_close(target,dense,atol=5e-5,rtol=5e-5)
    zero,_=AttentionOps.evaluate(q*0,k,v,s,m.basis,torch.zeros_like(mask))
    torch.testing.assert_close(zero,s.mean.float().mean(1)[:,None].expand_as(zero),atol=3e-5,rtol=3e-5)
    loss=(reference-target.detach()).square().mean();loss.backward()
    norms={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None}
    assert len(norms)==7 and all(x>0 for x in norms.values())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    # Ring correctness is independent of attention: check token identity and raw KV
    # through warmup, full capacity, and eviction. Also exercise drop==0.
    with torch.no_grad():
        for cap in (338,360):
            cfg=Config(history_pages=cap);cache={n:torch.empty(1,(cap+72)*64,12,128,device='cuda',dtype=torch.bfloat16) for n in ('k','v')};store=PageStore(cache,cfg)
            for chunk in range(7):
                kk=torch.randn(1,4608,12,128,device='cuda',dtype=torch.bfloat16);vv=torch.randn_like(kk)
                kg,vg,order=GroupingOps.group(m,kk,vv,cfg);summ=SummaryOps.build(m,kg,vg)
                write=store.write;store.commit(kg,vg,summ,order,chunk*4608)
                slots=(torch.arange(72,device='cuda')+write)%cap
                torch.testing.assert_close(store.token_ids.index_select(1,slots),order.reshape(12,72,64)+chunk*4608,rtol=0,atol=0)
                tokens=(slots[:,None]*64+torch.arange(64,device='cuda')).flatten()
                torch.testing.assert_close(cache['k'][0].index_select(0,tokens),kg.flatten(1,2).transpose(0,1),rtol=0,atol=0)
                torch.testing.assert_close(cache['v'][0].index_select(0,tokens),vg.flatten(1,2).transpose(0,1),rtol=0,atol=0)
            assert store.n==cap and store.commits==7
            del store,cache
    result=dict(passed=True,selected_reference_nmse=error,gradient_norms=norms,ring_caps=[338,360],ops_sha=digest('local_experiments/review_v1/ops.py'),runtime_sha=digest('local_experiments/review_v1/runtime.py'),scope='GPU mathematical-reference contracts; not single-TF32 production validation or end-to-end quality',speed_valid=False)
    atomic(Path('artifacts/review_v1_20261102/gpu_preflight.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
