import json,types,torch
from .gap_model import GapLayer
from .gap_route import route
from .conditional_v_kernels import reconstruct
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    def rand(*shape):return torch.randn(*shape,device='cuda')
    rt=types.SimpleNamespace(bank={1:rand(12,32,128)},weights={1:rand(12,4,128)*.01},basis={1:torch.linalg.qr(rand(12,128,8))[0].bfloat16()},centers={1:rand(12,128)*.1},sw={1:rand(12,14)*.01})
    m=GapLayer(rt,1);k=rand(12,24,64,128).bfloat16();v=rand(12,24,64,128).bfloat16();rows=[]
    for cheap in [False,True]:
        pk,base,detail=m.pool(k,v,cheap);err=float((pk.mean(2)-k.float().mean(2)).abs().max());assert err<1e-6,err
        expected=torch.einsum('hpd,hdk->hpk',v.float().mean(2)-m.center[:,None],m.basis.bfloat16().float()).bfloat16();assert torch.equal(base,expected)
        rec=reconstruct(base,detail);assert torch.allclose(rec.mean(2),base.float(),atol=1e-6,rtol=1e-6)
        modified=k.clone();modified[:,0]+=1;other=m.pool(modified,v,cheap)
        assert torch.equal(pk[:,1:],other[0][:,1:]) and torch.equal(base[:,1:],other[1][:,1:]) and torch.equal(detail[:,1:],other[2][:,1:])
        cov=m.covariance(k,v);q=rand(1,128,12,128).bfloat16();offsets=torch.linspace(0,63,8,device='cuda').round().long();vv=v.float().var(2,unbiased=False).mean(-1)
        a=route(m,q,pk,base,detail,cov,vv,offsets,False)[0];b=route(m,q,pk,base,detail,cov,vv,offsets,True)[0];assert torch.equal(a,b)
        assert a.shape[-1]==2;own=(torch.arange(2,device='cuda')+22)[None,:,None];assert (a[0]==own).any(-1).all()
        rows.append(dict(cheap=cheap,page_k_mean_max_error=err,encoded_v_mean_exact=True,page_local=True,route_training_inference_exact=True))
    atomic('artifacts/four_gaps_20261031/checks.json',dict(passed=True,rows=rows));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
