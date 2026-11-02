"""Page isolation, conservation, hard-forward and surrogate-gradient checks."""
import json
import math
from pathlib import Path
import subprocess
import sys
import torch
from .page_local import PageMixture,PageGate,hard_output
from .learned_pool import Bank
from .learned_cache.data import records
from .learned_cache.model import predict,initial_mask
from .learned_cache.io import atomic,restore
from migration.unified_router.core import part

ROOT=Path('artifacts/page_local_20261030/recovery')

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    ROOT.mkdir(exist_ok=True);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    model=PageMixture().cuda();e=Bank(4).get(records('train')[0]);h=e['h'];pk,pv,count=model(h)
    old=part(h['k'],h['v'],model.weight[h['layer'],h['heads']],model.bias[h['layer'],h['heads']])
    assert torch.allclose(pk,old.key,atol=2e-6,rtol=2e-5) and torch.allclose(pv,old.value,atol=2e-6,rtol=2e-5)
    assert torch.allclose(count,old.count,atol=2e-6,rtol=2e-5)
    errors={}
    for name,prototype,raw in [('k',pk,h['k']),('v',pv,h['v'])]:
        expected=raw.reshape(3,-1,64,128).sum(2);actual=(prototype*count[...,None]).sum(2)
        relative=(expected-actual).square().mean().sqrt()/expected.square().mean().sqrt()
        assert relative<1e-6;errors[name]=relative.item()
    assert torch.allclose(count.sum(-1),torch.full_like(count[...,0],64.),atol=2e-5,rtol=0)
    changed=dict(h,k=h['k'].clone(),v=h['v'].clone());changed['k'][:,:64]+=.1;changed['v'][:,:64]*=1.1
    after=model(changed)
    assert all(torch.equal(a[:,1:],b[:,1:]) for a,b in zip((pk,pv,count),after))
    assert not torch.equal(pk[:,:1],after[0][:,:1])
    one=part(h['k'],h['v'],torch.randn(3,1,128,device='cuda'),torch.zeros(3,1,device='cuda'))
    assert torch.allclose(one.key[:,:,0],h['k'].reshape(3,-1,64,128).mean(2),atol=1e-6,rtol=1e-5)
    z,_,_=predict(e['route'],(pk,pv,count),e['kc'],e['vc']);output=hard_output(z,e);mask=initial_mask(z)
    max_error=0.
    for head in range(3):
        for qi in [0,31,63]:
            ids=mask[head,qi,:-1].repeat_interleave(64)
            k=torch.cat((h['k'][head,ids],e['kc'][head]));v=torch.cat((h['v'][head,ids],e['vc'][head]))
            exact=(e['q'][head,qi]@k.T/math.sqrt(128)).softmax(-1)@v
            max_error=max(max_error,(exact-output[head,qi]).abs().max().item())
            assert torch.allclose(exact,output[head,qi],atol=3e-5,rtol=3e-4)
    output.square().mean().backward();assert torch.isfinite(model.weight.grad).all() and model.weight.grad.norm()>0
    # Validate the continuous gate Jacobian, NOT the derivative of top-k.
    zz=torch.randn(1,2,7,dtype=torch.float64);vv=torch.randn(1,2,7,5,dtype=torch.float64)
    hard=torch.tensor([[[1,0,1,0,1,0,1],[0,1,0,1,0,1,1]]],dtype=torch.bool)
    gate=hard.double().requires_grad_();up=torch.randn(1,2,5,dtype=torch.float64)
    analytic=torch.autograd.grad((PageGate.apply(gate,hard,zz,vv)*up).sum(),gate)[0]
    def continuous(g):
        mass=zz.exp()*g;out=(mass[...,None]*vv).sum(2)/mass.sum(-1)[...,None]
        return (out*up).sum()
    finite=torch.zeros_like(gate);epsilon=1e-6
    for i in range(gate.numel()):
        plus=gate.detach().clone();minus=gate.detach().clone();plus.flatten()[i]+=epsilon;minus.flatten()[i]-=epsilon
        finite.flatten()[i]=(continuous(plus)-continuous(minus))/(2*epsilon)
    assert torch.allclose(analytic,finite,atol=1e-8,rtol=1e-6)
    result=dict(core_part_equivalent=True,page_isolation_bit_exact=True,count_sum64=True,first_moment_relative_rms=errors,
        p1_is_mean=True,hard_forward_token_max_abs=max_error,hard_loss_gradient_finite_nonzero=True,
        continuous_gate_jacobian_max_abs=(analytic-finite).abs().max().item())
    def run(name,extra=()):
        with (ROOT/(name+'.log')).open('w') as f:
            subprocess.run([sys.executable,'-m','local_experiments.run_page_local','--root',str(ROOT/name),'--rep','mixture','--objective','hard','--steps','6',*extra],stdout=f,stderr=subprocess.STDOUT,check=True)
    run('continuous');run('split',['--max-updates','3'])
    assert json.loads((ROOT/'split/status.json').read_text())['status']=='interrupted'
    run('split');a,_=restore(ROOT/'continuous');b,_=restore(ROOT/'split')
    for key in ['model','optimizer','python_rng','torch_rng','cuda_rng','history','step']:
        result['resume_'+key]=equal(a[key],b[key]);assert result['resume_'+key],key
    atomic(ROOT/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
