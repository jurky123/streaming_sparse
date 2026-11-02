import argparse
import json
from pathlib import Path
import signal
import torch
from .refine_pages import ROOT,make_model,prediction
from .learned_pool import Bank,Pool,actions
from .learned_cache.data import records
from .learned_cache.model import predict,initial_mask,output_for_masks
from .learned_cache.run import nmse
from .learned_cache.io import atomic,digest,disk_guard,RunLock

BASE=Path('artifacts/page_local_20261030/mixture_reconstruction/best.pt')
ARMS=[('baseline',BASE,'random'),('response_reconstruction',ROOT/'response_reconstruction/best.pt','response'),
      ('random_split',ROOT/'random_split/best.pt','random'),('response_split',ROOT/'response_split/best.pt','response')]


def select_base():
    candidates=[]
    for name,path,rep in ARMS:
        assert json.loads((path.parent/'status.json').read_text())['status']=='completed'
        state=torch.load(path,weights_only=False,map_location='cpu')
        candidates.append(dict(name=name,path=str(path),sha256=digest(path),validation=state['best'],rep=rep))
    best=min(candidates,key=lambda r:r['validation']);atomic(ROOT/'calibration_base.json',best)
    atomic(ROOT/'base_candidates.json',candidates);print(json.dumps(best,indent=2))


@torch.no_grad()
def evaluate():
    dest=ROOT/'evaluation';lock=RunLock(dest);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    models={};hashes={};data_identity=None
    for name,path,rep in ARMS+[('calibration',ROOT/'calibration/best.pt','calibration')]:
        assert json.loads((path.parent/'status.json').read_text())['status']=='completed'
        state=torch.load(path,weights_only=False,map_location='cpu');model=make_model(rep).cuda();model.load_state_dict(state['model'])
        models[name]=(model,rep);hashes[name]=digest(path)
        sources=state['identity'].get('code',state['identity'].get('sources',{}))
        for p,sha in sources.items():assert digest(p)==sha
        if data_identity is None:
            data_identity=state['identity']['data']
            for p,sha in data_identity.items():assert digest(p)==sha
    models['response_initial']=(make_model('response').cuda(),'response');models['hard_mean']=(Pool().cuda(),'mean')
    ident=dict(checkpoints=hashes,data=data_identity,code=digest(__file__),initial=digest(ROOT/'response_initial.pt'))
    if (dest/'identity.json').exists():assert json.loads((dest/'identity.json').read_text())==ident
    else:atomic(dest/'identity.json',ident)
    bank=Bank(4)
    try:
        for path in records('test'):
            shard=dest/(path.parent.parent.name+'_'+path.stem+'.json')
            if shard.exists():continue
            if stop[0]:break
            disk_guard(ROOT);e=bank.get(path);rows=[]
            for name,(model,rep) in models.items():
                if rep=='mean':z,val,_=predict(e['route'],model(e['h'],False),e['kc'],e['vc'])
                else:z,val,_=prediction(model,e,rep)
                masks,action,_=actions(z,val,'residual');outputs=output_for_masks(e['z'],e['values'],masks);errors=nmse(outputs,e['target'][:,:,None])
                choices={'none':torch.zeros_like(action),'residual':action,'oracle':errors.argmin(-1)}
                for mode,index in choices.items():
                    chosen=errors.gather(-1,index[:,:,None]).squeeze(-1)
                    for hi,head in enumerate(e['h']['heads']):
                        rows.append(dict(model=name,mode=mode,scene=e['scene'],layer=e['h']['layer'],head=int(head),chunk=e['chunk'],step=e['step'],
                            nmse=chosen[hi].mean().item(),swap=(index[hi]!=0).float().mean().item(),harm=(chosen[hi]>errors[hi,:,0]+1e-9).float().mean().item()))
            atomic(shard,rows);print(shard.name,flush=True)
        rows=[]
        for p in dest.glob('*_s?.json'):rows.extend(json.loads(p.read_text()))
        summary={};breakdown={}
        for name,mode in sorted({(r['model'],r['mode']) for r in rows}):
            rr=[r for r in rows if r['model']==name and r['mode']==mode]
            summary[name+'_'+mode]={k:sum(r[k] for r in rr)/len(rr) for k in ['nmse','swap','harm']}
        for field in ['scene','layer','head']:
            breakdown[field]={}
            for value in sorted({r[field] for r in rows}):
                rr=[r for r in rows if r[field]==value];breakdown[field][str(value)]={}
                for name,mode in sorted({(r['model'],r['mode']) for r in rr}):
                    sub=[r for r in rr if r['model']==name and r['mode']==mode]
                    breakdown[field][str(value)][name+'_'+mode]=sum(r['nmse'] for r in sub)/len(sub)
        atomic(dest/'summary.json',summary);atomic(dest/'breakdown.json',breakdown)
        atomic(dest/'status.json',dict(status='interrupted' if stop[0] else 'completed',events=len(rows)//63,disk=disk_guard(ROOT)))
        print(json.dumps(summary,indent=2))
    finally:lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--select-base',action='store_true');a=parser.parse_args()
    select_base() if a.select_base else evaluate()
