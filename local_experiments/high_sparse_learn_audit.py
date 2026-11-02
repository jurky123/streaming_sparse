"""Actual closed-loop cache-page eager/fused agreement, read-only teacher audit."""
import json
import torch
from .high_sparse_learn import Layer
from .high_sparse_learn_runtime import LearnedHighRuntime
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic
ROOT='artifacts/high_sparse_train_20261030'
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    epoch=json.load(open(ROOT+'/selection.json'))['selected_epoch'];path=f'{ROOT}/epoch{epoch}.pt';state=torch.load(path,weights_only=True)['models']
    engine=Engine();rt=LearnedHighRuntime(engine,path,'joint');models={}
    for l in [4,14,24]:
        m=Layer(state[l]).cuda();m.load_state_dict(state[l]);models[l]=m
    rows=[];base=rt.route
    def route(l,q,p):
        lut=base(l,q,p);c=engine.current
        if l not in models or c['chunk'] not in [2,5,8] or c['step'] not in [0,3]:return lut
        cache=engine.cache[l];slots=torch.tensor([0,p-1,p,p+71],device='cuda');ids=(slots[:,None]*64+torch.arange(64,device='cuda')).flatten()
        k=cache['k'][0,ids].transpose(0,1).float().reshape(12,4,64,128);v=cache['v'][0,ids].transpose(0,1).float().reshape_as(k)
        code,_=models[l].value(k,v);actual=rt.active_codes[:,slots].float()
        err=float((code-actual).square().sum()/code.square().sum().clamp_min(1e-8));assert err<1e-5,(c,err)
        own=torch.arange(72,device='cuda')+p;assert (lut[0]==own[None,:,None]).any(-1).all()
        assert (lut.sort(-1).values.diff(dim=-1)>0).all()
        rows.append(dict(layer=l,chunk=c['chunk'],step=c['step'],code_nmse=err,selected=lut.shape[-1]))
        return lut
    rt.route=route
    try:
        emb=torch.load('artifacts/mass_variance_20261030/embeddings/basketball_court.pt',weights_only=True).cuda()
        for c in range(9):generate_chunk(engine,emb,1001,c)
        assert len(rows)==18
        atomic(ROOT+'/real_cache_checks.json',dict(passed=True,rows=rows,persistent_bytes=rt.persistent_bytes));print(json.dumps(dict(passed=True,checks=len(rows),max_nmse=max(r['code_nmse'] for r in rows))))
    finally:rt.close()
if __name__=='__main__':main()
