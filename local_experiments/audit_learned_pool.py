"""Stratification, frozen-model timing and nonuniform pooling diagnostics."""
import json
import statistics
import shutil
from pathlib import Path
import torch
from .learned_pool import Pool, Bank, actions
from .run_learned_pool import prediction
from .learned_cache.data import records
from .learned_cache.model import predict, initial_mask
from .learned_cache.io import atomic, digest, disk_guard

ROOT=Path('artifacts/learned_pool_20261030')

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    result={};summary={};breakdowns={}
    def timed(fn):
        for _ in range(5):fn()
        samples=[]
        for _ in range(30):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
        return dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples))
    for count in [1,4]:
        root=ROOT/f'p{count}';assert json.loads((root/'evaluation/status.json').read_text())['status']=='completed'
        saved=torch.load(root/'best.pt',weights_only=False,map_location='cpu');model=Pool().cuda();model.load_state_dict(saved['model'])
        bank=Bank(count,max_history=1,max_events=1);e=bank.get(records('test')[0]);h=e['h']
        item=dict(best_step=saved['step'],validation_initial=saved['history'][0]['val'],validation_best=saved['best'],parameters=sum(p.numel() for p in model.parameters()))
        item['pool_mean']=timed(lambda:model(h,False));item['pool_learned']=timed(lambda:model(h))
        for which in ['mean','both']:
            proto=model(h,which!='mean')
            for kind in ['none','boundary','residual']:
                def route():
                    z,v,_=predict(e['route'],proto,e['kc'],e['vc'])
                    return initial_mask(z) if kind=='none' else actions(z,v,kind)
                item[which+'_'+kind+'_route']=timed(route)
        rows=[]
        for p in (root/'evaluation').glob('*_s?.json'):rows.extend(json.loads(p.read_text()))
        splits={}
        for field in ['scene','layer','head']:
            splits[field]={}
            for value in sorted({r[field] for r in rows}):
                sub=[r for r in rows if r[field]==value]
                splits[field][str(value)]={}
                for pool,mode in sorted({(r['pool'],r['correction']) for r in sub}):
                    splits[field][str(value)][pool+'_'+mode]=statistics.mean(r['nmse'] for r in sub if r['pool']==pool and r['correction']==mode)
        breakdowns[str(count)]=splits;summary[str(count)]=json.loads((root/'evaluation/summary.json').read_text())
        # One history per scene/chunk/layer, all held-out histories; no model changes.
        weights=[];deltas=[];entropy=[]
        cache={}
        hook=model.net.register_forward_hook(lambda m,i,o:cache.update(logits=o))
        for p in records('test')[::4]:
            event=bank.get(p);hh=event['h'];learned=model(hh);mean=model(hh,False)
            w=(3*torch.tanh(cache['logits']/3)).softmax(3)
            weights.append((1/w.square().sum(3)).mean((0,1,2)).cpu())
            entropy.append((-(w*w.clamp_min(1e-20).log()).sum(3)).mean((0,1,2)).cpu())
            deltas.append(torch.stack([(learned[i]-mean[i]).square().mean().sqrt()/mean[i].square().mean().sqrt().clamp_min(1e-8) for i in [0,1]]).cpu())
        hook.remove();item['effective_tokens_k_v']=torch.stack(weights).mean(0).tolist();item['entropy_k_v']=torch.stack(entropy).mean(0).tolist()
        item['prototype_relative_rms_change_k_v']=torch.stack(deltas).mean(0).tolist();result[str(count)]=item
        snap=ROOT/'code_snapshot'
        for source,expected in saved['identity']['sources'].items():
            assert digest(source)==expected
            dst=snap/Path(source);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    for filename in ['audit_learned_pool.py','test_learned_pool.py']:
        shutil.copy2(Path('local_experiments')/filename,ROOT/'code_snapshot'/filename)
    atomic(ROOT/'summary.json',summary);atomic(ROOT/'breakdown.json',breakdowns)
    atomic(ROOT/'audit.json',dict(measurements=result,scope='H200,3heads,64query,5historychunks; routing includes summary prediction and mask/candidate generation, no sparse kernel; pooling excludes grouping and feature preparation',disk=disk_guard(ROOT)))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
