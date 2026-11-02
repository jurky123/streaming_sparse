"""Gradient, conservation, swap budget/oracle and optimizer resume checks."""
import argparse,copy,time
from pathlib import Path
import torch
from .improve_models import Model,METHODS,objective,evaluate,swaps,scores
from .learned_cache.io import atomic,checkpoint,restore
ROOT=Path('artifacts/rollout_improve_20261030')
def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
    if a.wait_pid:
        while Path(f'/proc/{a.wait_pid}').exists():time.sleep(5)
    torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    k=torch.randn(12,338*64,128,device='cuda');v=torch.randn_like(k)
    z=torch.randn(12,32,339,device='cuda');values=torch.randn(12,32,339,128,device='cuda')
    target=(z.softmax(-1)[...,None]*values).sum(-2)
    mass=z.softmax(-1).reshape(12,8,4,339).mean(2)[...,:338]
    e=dict(h=dict(k=k,v=v,center=k.mean(1)),route=torch.randn(12,8,128,device='cuda'),var=torch.rand(12,8,128,device='cuda'),
        z=z,values=values,target=target,shared_mass=mass/mass.sum(-1,keepdim=True),kc=torch.randn(12,4608,128,device='cuda'),vc=torch.randn(12,4608,128,device='cuda'))
    checks={}
    for method in METHODS:
        m=Model(14,method).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        def step(model,optimizer):
            optimizer.zero_grad(set_to_none=True);loss=objective(model,e);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert sum(p.grad.abs().sum().item() for p in model.parameters())>0
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();model.project()
            return loss.item()
        loss=step(m,opt);saved=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
        checkpoint(ROOT/'resume_check'/method,saved);step(m,opt)
        resumed=Model(14,method).cuda();ropt=torch.optim.AdamW(resumed.parameters(),lr=.001)
        saved,_=restore(ROOT/'resume_check'/method);resumed.load_state_dict(saved['model']);ropt.load_state_dict(saved['optimizer']);step(resumed,ropt)
        equal(m.state_dict(),resumed.state_dict());equal(opt.state_dict(),ropt.state_dict())
        with torch.no_grad():
            proto=m(e['h']);pk,_,count=proto
            if method=='page_local_centered':
                torch.testing.assert_close(count.sum(-1),torch.full((12,338),64.,device='cuda'),atol=2e-5,rtol=1e-5)
                torch.testing.assert_close((pk*count[...,None]).sum(-2),k.reshape(12,338,64,128).sum(2),atol=5e-5,rtol=1e-4)
            if m.centered:
                c=torch.randn(12,128,device='cuda')*3
                before=scores(proto,e['route'],e['var'],m.alpha,e['h']['center']).softmax(-1)
                after=scores((pk+c[:,None,None],None,count),e['route'],e['var'],m.alpha,e['h']['center']+c).softmax(-1)
                torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-4)
            metric,s,mask=evaluate(m,e);oracle=swaps(e,s,mask,False);proxy=swaps(e,s,mask,True)
            assert all(x<=y+1e-5 for x,y in zip(oracle['nmse'],metric['nmse']))
            checks[method]=dict(loss=loss,finite_nonzero_gradient=True,optimizer_resume_exact=True,oracle_nonworsening=True,mask_budget_passed=True,
                nmse=sum(metric['nmse'])/12,oracle_nmse=sum(oracle['nmse'])/12,proxy_nmse=sum(proxy['nmse'])/12)
        print(method,'passed',flush=True)
    atomic(ROOT/'training_checks.json',dict(passed=True,checks=checks,note='Synthetic shape/gradient/identity checks; not model-quality evidence.'))
if __name__=='__main__':main()
