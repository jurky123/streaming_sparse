import json
import statistics
from pathlib import Path
import shutil
import torch
from .page_local import make_model,prototypes
from .learned_pool import Bank,actions
from .learned_cache.model import predict,initial_mask
from .learned_cache.data import records
from .learned_cache.io import atomic,digest,disk_guard

ROOT=Path('artifacts/page_local_20261030')

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert json.loads((ROOT/'evaluation/status.json').read_text())['status']=='completed'
    rows=[]
    for p in (ROOT/'evaluation').glob('*_s?.json'):rows.extend(json.loads(p.read_text()))
    breakdown={}
    for field in ['scene','layer','head']:
        breakdown[field]={}
        for val in sorted({r[field] for r in rows}):
            rr=[r for r in rows if r[field]==val];breakdown[field][str(val)]={}
            for name,correction in sorted({(r['model'],r['correction']) for r in rr}):
                breakdown[field][str(val)][name+'_'+correction]=statistics.mean(r['nmse'] for r in rr if r['model']==name and r['correction']==correction)
    atomic(ROOT/'breakdown.json',breakdown)
    result={};bank=Bank(4,max_history=1,max_events=1)
    def timed(fn):
        for _ in range(5):fn()
        times=[]
        for _ in range(30):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();times.append(a.elapsed_time(b))
        return dict(median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times))
    manifest=None
    for name,rep in [('mixture_hard','mixture'),('mixture_reconstruction','mixture'),('pool_hard','pool')]:
        state=torch.load(ROOT/name/'best.pt',weights_only=False,map_location='cpu');model=make_model(rep).cuda();model.load_state_dict(state['model'])
        if manifest is None:
            manifest=state['identity']
            for path,sha in manifest['data'].items():assert digest(path)==sha,'Data changed: '+path
        item=dict(best_step=state['step'],validation_initial=state['history'][0]['val'],validation_best=state['best'],
                  parameters=sum(p.numel() for p in model.parameters()))
        e=bank.get(records('test')[0]);h=e['h']
        item['preprocess']=timed(lambda:prototypes(model,h,rep));proto=prototypes(model,h,rep)
        for mode in ['none','boundary','residual']:
            def route():
                z,v,_=predict(e['route'],proto,e['kc'],e['vc'])
                return initial_mask(z) if mode=='none' else actions(z,v,mode)
            item['routing_'+mode]=timed(route)
        if rep=='mixture':
            stats=[];moment=[];count_err=[];mincount=float('inf');maxcount=0.
            for path in records('test')[::4]:
                ev=bank.get(path);hh=ev['h'];pk,pv,count=model(hh);a=model.assignment(hh)
                mincount=min(mincount,count.min().item());maxcount=max(maxcount,count.max().item())
                stats.append(dict(effective_prototypes=(64**2/count.square().sum(-1)).mean().item(),
                    token_assignment_entropy=(-(a*a.clamp_min(1e-20).log()).sum(-1)).mean().item()))
                count_err.append((count.sum(-1)-64).abs().max().item())
                for proto,raw in [(pk,hh['k']),(pv,hh['v'])]:
                    expected=raw.reshape(3,-1,64,128).sum(2);actual=(proto*count[...,None]).sum(2)
                    moment.append(((actual-expected).square().mean().sqrt()/expected.square().mean().sqrt()).item())
            item.update(mean_effective_prototypes=statistics.mean(s['effective_prototypes'] for s in stats),
                assignment_entropy=statistics.mean(s['token_assignment_entropy'] for s in stats),minimum_count=mincount,maximum_count=maxcount,
                max_mass_abs=max(count_err),max_first_moment_relative_rms=max(moment))
        result[name]=item
    # Confirm reused baseline metrics match their original frozen-model results.
    current=json.loads((ROOT/'evaluation/summary.json').read_text())
    prior=json.loads(Path('artifacts/learned_pool_20261030/p4/evaluation/summary.json').read_text())
    for mode in ['none','boundary','residual','boundary_oracle','residual_oracle']:
        assert abs(current['pool_reconstruction_'+mode]['nmse']-prior['both_'+mode]['nmse'])<1e-8
        assert abs(current['hard_mean_'+mode]['nmse']-prior['mean_'+mode]['nmse'])<1e-8
    snap=ROOT/'code_snapshot'
    for source,sha in manifest['code'].items():
        assert digest(source)==sha;dst=snap/source;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    for file in ['audit_page_local.py','test_page_local.py']:
        shutil.copy2(Path('local_experiments')/file,snap/file)
    atomic(ROOT/'audit.json',dict(measurements=result,prior_baselines_reproduced=True,data_hashes_verified=True,
        scope='H200,3heads,64queries,5historychunks; cached grouped data; preprocessing excludes grouping/feature preparation; routing excludes actual sparse kernel',disk=disk_guard(ROOT)))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
