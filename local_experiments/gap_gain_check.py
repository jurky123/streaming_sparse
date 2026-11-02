import torch,triton,json
from .gap_gain_kernel import contribution,finish
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(871);rows=[]
    for p in [144,432]:
        mass=torch.rand(12,72,p,device='cuda');mass=mass/mass.sum(-1,keepdim=True)
        prob=torch.rand(12,72,16,p,device='cuda');qp=torch.randn(12,72,16,8,device='cuda')
        q=torch.einsum('htmp,htmd->htpd',prob,qp)
        cov=torch.randn(12,p,8,device='cuda').bfloat16();mask=torch.rand_like(mass)<.1;s=torch.randn_like(mass);gate=torch.randn(12,device='cuda')
        def native():
            qbar=q/(16*mass[...,None].clamp_min(1e-12));residual=qbar*cov.float()[:,None]
            c=mass[...,None]*residual;e=(c*(~mask)[...,None]).sum(2)
            energy=e.square().sum(-1,keepdim=True).clamp_min(1e-8)
            gain=(2*(e[:,:,None]*c).sum(-1)+(2*mask.float()-1)*c.square().sum(-1))/energy
            return c,e,s+gate.tanh()[:,None,None]*gain.clamp(-5,5)
        def fused():
            c,e=contribution(q,mass,cov,mask);return c,e,finish(c,e,mask,s,gate)
        ref=native();out=fused()
        row=dict(p=p,exact=[torch.equal(a,b) for a,b in zip(ref,out)],errors=[float((a-b).abs().max()) for a,b in zip(ref,out)],native_ms=triton.testing.do_bench_cudagraph(native,rep=200),fused_ms=triton.testing.do_bench_cudagraph(fused,rep=200))
        rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_pool_20261031/gain_checks.json',dict(rows=rows))
if __name__=='__main__':main()
