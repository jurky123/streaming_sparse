import json
from pathlib import Path
import shutil
import statistics
import torch
from .refine_pages import ROOT,make_model,prediction
from .evaluate_refine_pages import ARMS
from .learned_pool import Bank
from .learned_cache.data import records
from .learned_cache.io import atomic,digest,disk_guard

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    bank=Bank(4,max_history=1,max_events=1);results={};sources={}
    models=[('response_initial',None,'response')]+ARMS+[('calibration',ROOT/'calibration/best.pt','calibration')]
    for name,path,rep in models:
        torch.manual_seed(0);model=make_model(rep).cuda();info={}
        if path is not None:
            state=torch.load(path,weights_only=False,map_location='cpu');model.load_state_dict(state['model'])
            info.update(best_step=state['step'],validation_initial=state['history'][0]['val'],validation_best=state['best'])
            sources.update(state['identity'].get('code',{}))
        effective=[];explained=[];deltas=[];centered=[];shared=[];churn=[]
        for p in (records('test') if rep=='calibration' else records('test')[::4]):
            e=bank.get(p);h=e['h'];pk,pv,count=model(h);qbank=bank.dictionary[h['layer']]
            scores=torch.einsum('hprd,hmd->hprm',pk,qbank)
            mean=(scores*(count/64)[...,None]).sum(2)
            between=((scores-mean[:,:,None]).square()*(count/64)[...,None]).sum(2).mean(-1)
            raw=(h['k']@qbank.transpose(1,2)).reshape(3,-1,64,32)
            total=(raw-raw.mean(2,keepdim=True)).square().mean((2,3))
            explained.append((between/total.clamp_min(1e-8)).mean().item())
            effective.append((64**2/count.square().sum(-1)).mean().item())
            if rep=='calibration':
                delta=model.correction(e['route'],h,(pk,pv,count));deltas.append(delta.abs().mean().item())
                shared.append(delta.mean(-1).abs().mean().item());centered.append(delta.std(-1,unbiased=False).mean().item())
                z,_,_=prediction(model,e,'calibration');bz,_,_=prediction(model.base,e,'random')
                selected=torch.zeros_like(z[...,:-1],dtype=torch.bool).scatter_(-1,z[...,:-1].topk(72).indices,True)
                original=torch.zeros_like(selected).scatter_(-1,bz[...,:-1].topk(72).indices,True)
                churn.append(((selected&~original).sum(-1).float()/72).mean().item())
        info.update(mean_effective_prototypes=statistics.mean(effective),mean_response_between_fraction=statistics.mean(explained))
        if deltas:info.update(mean_abs_logz_correction=statistics.mean(deltas),mean_abs_shared_offset=statistics.mean(shared),
            mean_across_page_std=statistics.mean(centered),selected_page_replacement_fraction=statistics.mean(churn))
        results[name]=info
    snap=ROOT/'code_snapshot'
    for p,sha in sources.items():
        assert digest(p)==sha;dst=snap/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for name in ['evaluate_refine_pages.py','test_refine_pages.py','bench_refine_pages.py','audit_refine_pages.py']:
        shutil.copy2(Path('local_experiments')/name,snap/name)
    summary=json.loads((ROOT/'evaluation/summary.json').read_text())
    old=json.loads(Path('artifacts/page_local_20261030/evaluation/summary.json').read_text())
    assert abs(summary['baseline_none']['nmse']-old['mixture_reconstruction_none']['nmse'])<1e-8
    assert abs(summary['hard_mean_none']['nmse']-old['hard_mean_none']['nmse'])<1e-8
    atomic(ROOT/'audit.json',dict(models=results,prior_baselines_reproduced=True,storage=disk_guard(ROOT)))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
