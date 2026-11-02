import json
from pathlib import Path
import subprocess
import sys
import torch
from .refine_pages import ROOT,make_model,prediction
from .learned_pool import Bank
from .learned_cache.data import records
from .learned_cache.io import atomic,restore,digest
from .test_page_local import equal

def main():
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    folder=ROOT/'recovery';folder.mkdir(exist_ok=True);e=Bank(4).get(records('train')[0]);model=make_model('calibration').cuda()
    z,v,out=prediction(model,e,'calibration');basez,basev,baseout=prediction(model.base,e,'random')
    assert torch.equal(z,basez) and torch.equal(v,basev) and torch.equal(out,baseout)
    assert not any(p.requires_grad for p in model.base.parameters())
    with torch.no_grad():
        model.net[-1].weight.normal_();model.net[-1].bias.fill_(.1)
        proto=model(e['h']);delta=model.correction(e['route'],e['h'],proto)
        assert delta.abs().max()<=.5
        h=dict(e['h'],k=e['h']['k'].clone());h.pop('variance8',None);h['k'][:,:64]+=.1
        other=model.correction(e['route'],h,model(h))
        assert torch.equal(delta[:,:,1:],other[:,:,1:])
    model.zero_grad();prediction(model,e,'calibration')[0].square().mean().backward()
    assert torch.isfinite(model.net[-1].weight.grad).all() and model.net[-1].weight.grad.norm()>0
    manifest=json.loads((ROOT/'initialization_manifest.json').read_text())
    for p,sha in manifest['inputs'].items():
        assert digest(p)==sha
        if p.endswith('_history.pt'):assert '/sailboat/' in p and ('/c05_' in p or '/c20_' in p)
    def run(name,extra=()):
        with (folder/(name+'.log')).open('w') as f:
            subprocess.run([sys.executable,'-m','local_experiments.run_refine_pages','--root',str(folder/name),'--rep','calibration','--objective','calibration','--steps','6',*extra],stdout=f,stderr=subprocess.STDOUT,check=True)
    run('continuous');run('split',['--max-updates','3']);assert json.loads((folder/'split/status.json').read_text())['status']=='interrupted';run('split')
    a,_=restore(folder/'continuous');b,_=restore(folder/'split')
    result=dict(zero_calibration_exact=True,base_frozen=True,page_local_correction=True,bounded_delta=True,train_only_initialization=True)
    for key in ['model','optimizer','python_rng','torch_rng','cuda_rng','history','step']:
        result['resume_'+key]=equal(a[key],b[key]);assert result['resume_'+key],key
    atomic(folder/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
