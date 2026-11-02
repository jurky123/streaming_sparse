"""Fused aggregation numerical checks with ring eviction and unchanged budget."""
import json
import torch
from .high_sparse_learn import Layer,features,probe_mass,pick,swap_features,correct
from .high_sparse_learn_kernels import pool
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(191);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for scale in [0.,.03,.2]:
        old=dict(basis=torch.linalg.qr(torch.randn(12,128,8))[0],center=torch.randn(12,128)*.1)
        m=Layer(old).cuda();m.uk.normal_(0,scale);m.uv.normal_(0,scale)
        cache={k:torch.randn(1,(338+72)*64,12,128,device='cuda',dtype=torch.bfloat16) for k in ['k','v']}
        for ring in [False,True]:
            cap=338 if ring else 72;start=300 if ring else 0;offset=0 if ring else 338*64
            slots=(torch.arange(72,device='cuda')+start)%cap;tokens=slots[:,None]*64+torch.arange(64,device='cuda') if ring else offset+torch.arange(4608,device='cuda').reshape(72,64)
            k=cache['k'][0,tokens.flatten()].transpose(0,1).float().reshape(12,72,64,128);v=cache['v'][0,tokens.flatten()].transpose(0,1).float().reshape_as(k)
            refcode,_=m.value(k,v);out=torch.zeros(12,cap,8,device='cuda',dtype=torch.bfloat16);var=torch.zeros(12,cap,device='cuda',dtype=torch.bfloat16);pk=None if ring else torch.empty(12,72,4,128,device='cuda')
            pool(cache,m.basis.to(torch.bfloat16),m.center,m.uk,m.uv,out,var,start,offset,ring,True,pk)
            actual=out[:,slots].float();error=float((actual-refcode).square().sum()/refcode.square().sum().clamp_min(1e-8));assert error<1e-5,error
            refvar=(v-v.mean(2,keepdim=True)).square().mean((2,3))
            torch.testing.assert_close(var[:,slots].float(),refvar,rtol=.005,atol=1e-4)
            if pk is not None:torch.testing.assert_close(pk,k.reshape(12,72,4,16,128).mean(3),rtol=1e-5,atol=1e-5)
            rows.append(dict(scale=scale,ring=ring,code_nmse=error))
    q=torch.randn(12,4,4,128,device='cuda');pk=torch.randn(12,216,4,128,device='cuda');code=torch.randn(12,216,8,device='cuda')
    _,prob,mass=probe_mass(q,pk);f=features(prob,pk,torch.ones(12,216,device='cuda'));own=(torch.arange(4,device='cuda')+144)[None].expand(12,-1)
    mask=pick(mass,21,own);sf=swap_features(f,mass,code,mask);assert sf.shape==(12,4,216,14)
    s=torch.randn(12,4,216,device='cuda');new,change=correct(mask,s,own)
    assert (new.sum(-1)==21).all() and new.gather(-1,own[...,None]).all() and ((new^mask).sum(-1)<=2).all()
    result=dict(passed=True,rows=rows,extra_persistent_bytes=29*12*(2*128*4+(8+14)*4+338*2),total_persistent_bytes=5095292576+29*12*(2*128*4+(8+14)*4+338*2))
    assert result['total_persistent_bytes']<=5096079360
    atomic('artifacts/high_sparse_train_20261030/kernel_checks.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
