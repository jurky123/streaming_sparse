"""Read-only local error on actual trained closed-loop trajectories, no fitting."""
import json,gc
from pathlib import Path
import torch
from .high_sparse_learn import Layer,event,evaluate
from .high_sparse_learn_runtime import LearnedHighRuntime
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_train_20261030')
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    epoch=json.loads((ROOT/'selection.json').read_text())['selected_epoch'];path=ROOT/f'epoch{epoch}.pt';saved=torch.load(path,weights_only=True)['models'];old=torch.load('artifacts/compensation_probe_20261030/model.pt',weights_only=True)['models']
    models={};ob={};oc={}
    for l in [4,14,24]:
        m=Layer(saved[l]).cuda();m.load_state_dict(saved[l]);models[l]=m;ob[l]=old[l]['basis'].cuda().to(torch.bfloat16).float();oc[l]=old[l]['center'].cuda().float()
    engine=Engine()
    for case in ['basketball_court','butterfly_garden']:
        out=ROOT/'shift'/(case+'.json')
        if out.exists():continue
        rows=[];rt=LearnedHighRuntime(engine,path,'joint')
        class Observer:
            def __init__(self,base):self.base=base
            def __getitem__(self,grid):
                def call(*args,**kwargs):
                    c=engine.current;l=rt.active_layer
                    if l in models and c['chunk'] in [2,5,8] and c['step'] in [0,3]:
                        e=event(rt,args[0],args[1],args[2],args[5]);e['old_code']=((e['v'].mean(2)-oc[l][:,None])@ob[l]).to(torch.bfloat16).float()
                        row=evaluate(models[l],e,ob[l],oc[l]);row.update(layer=l,chunk=c['chunk'],step=c['step']);rows.append(row)
                    self.base[grid](*args,**kwargs)
                return call
        rt.kernel=Observer(rt.kernel)
        try:
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            for c in range(9):generate_chunk(engine,emb,1001,c)
            assert len(rows)==18
            means={m:sum(r['nmse'][m] for r in rows)/len(rows) for m in rows[0]['nmse']}
            atomic(out,dict(case=case,source='trained joint closed-loop; read-only local counterfactuals, not alternate videos',rows=rows,means=means));print(json.dumps(dict(case=case,means=means)),flush=True)
        finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
