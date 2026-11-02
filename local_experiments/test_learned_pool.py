import json
from pathlib import Path
import subprocess
import sys
import torch
from .learned_pool import Pool, Bank, actions
from .learned_cache.data import records
from .learned_cache.model import predict, output_for_masks
from .learned_cache.io import restore, atomic
from .response_hard import grouping

ROOT=Path('artifacts/learned_pool_20261030/recovery')

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    ROOT.mkdir(exist_ok=True);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    results={};model=Pool().cuda();max_error=0.
    for count in [1,4]:
        bank=Bank(count)
        for path in [records('train')[i] for i in [0,4,8]]:
            e=bank.get(path);h=e['h'];protos=model(h)
            mean=model(h,False)
            assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(protos,mean))
            hp=path.with_name(path.name[:-6]+'_history.pt');data=torch.load(hp,weights_only=True)
            old=grouping(data['k'].cuda().float(),data['v'].cuda().float(),bank.dictionary[h['layer']],'response')[2 if count==1 else 3]
            assert all(torch.allclose(a,b,atol=2e-6,rtol=2e-5) for a,b in zip(mean,old))
            z,val,_=predict(e['route'],protos,e['kc'],e['vc'])
            for mode in ['boundary','residual']:
                masks,action,errors=actions(z,val,mode)
                direct=output_for_masks(z,val,masks)
                full=(z.softmax(-1)[:,:,:,None]*val).sum(2)
                ref=(direct-full[:,:,None]).square().mean(-1)
                max_error=max(max_error,(errors-ref).abs().max().item())
                assert torch.allclose(errors,ref,atol=2e-6,rtol=2e-4)
                assert (masks.sum(-1)==73).all()
            model.zero_grad();(protos[0].square().mean()+protos[1].square().mean()).backward()
            assert torch.isfinite(model.net[-1].weight.grad).all()
            assert (model.net[-1].weight.grad.norm(dim=-1)>0).all()
    results.update(initial_matches_hard_mean=True,gradient_both_channels=True,swap_budget=True,incremental_error_max_abs=max_error)
    # Release references before starting child GPU workers.
    del bank,model,e,protos,z,val,mean,old,data,errors,direct,full,ref,h
    torch.cuda.empty_cache()
    def run(name,extra=()):
        with (ROOT/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-m','local_experiments.run_learned_pool','--root',str(ROOT/name),'--prototypes','1','--steps','6',*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
    run('continuous');run('split',['--max-updates','3'])
    assert json.loads((ROOT/'split/status.json').read_text())['status']=='interrupted'
    run('split');a,_=restore(ROOT/'continuous');b,_=restore(ROOT/'split')
    for key in ['model','optimizer','python_rng','torch_rng','cuda_rng','history','step']:
        results['resume_'+key]=equal(a[key],b[key]);assert results['resume_'+key],key
    atomic(ROOT/'results.json',results);print(json.dumps(results,indent=2))

if __name__=='__main__':main()
