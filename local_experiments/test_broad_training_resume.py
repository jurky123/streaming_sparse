"""Compare continuous updates against actual interrupted trainer weights/Adam."""
import copy
import json
import torch
from .broad_common import ROOT,Engine,cases
from .broad_router import Events,LayerRouter,prediction
from .refine_pages import loss_terms
from .learned_cache.io import atomic

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def run():
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={m:LayerRouter(0).cuda() for m in ['reconstruction','split']}
    opts={m:torch.optim.AdamW(v.parameters(),lr=.001,weight_decay=.0001) for m,v in models.items()}
    before=torch.load(ROOT/'recovery_before.pt',map_location='cpu',weights_only=False)
    after=torch.load(ROOT/'recovery_after.pt',map_location='cpu',weights_only=False)
    assert before['cursor']['chunk']==5 and after['cursor']['chunk']==20
    assert before['updates'][0]==4 and after['updates'][0]==8
    case=next(c for c in cases('train') if c['id']=='train_robot_work_s0')
    engine=Engine();events=Events([0]);checks=[]
    def callback(current,q,k,v):
        if current['layer']!=0:return
        e=events.event(current,q,k,v)
        for m in models:
            with torch.enable_grad():
                opts[m].zero_grad(set_to_none=True)
                loss=loss_terms(*prediction(models[m],e),e,m);loss.backward()
                torch.nn.utils.clip_grad_norm_(models[m].parameters(),1.,error_if_nonfinite=True);opts[m].step()
            if current['step']==3:
                reference=before if current['chunk']==5 else after
                assert equal(models[m].state_dict(),reference['models'][m][0]), ('model',m,current)
                assert equal(opts[m].state_dict(),reference['optimizers'][m][0]), ('optimizer',m,current)
                checks.append(dict(method=m,chunk=current['chunk'],weights_exact=True,adam_exact=True))
    engine.replay(case,callback,stop_chunk=20)
    atomic(ROOT/'training_resume_checks.json',dict(passed=True,checks=checks,
        scope='layer0 both objectives; actual 15-layer trainer interrupted after 4 updates, resumed to8; compared against continuous8'))
    print(json.dumps(checks),flush=True)
if __name__=='__main__':run()
