"""Real teacher event + trained Adam state: two updates vs checkpoint/reload."""
import copy,json
from pathlib import Path
import torch
from .broad_common import Engine,cases,ROOT as DATA
from .improve_models import Model,Events,METHODS,objective
from .improve_train_checks import equal
from .learned_cache.io import atomic,checkpoint,restore
ROOT=Path('artifacts/rollout_improve_20261030')
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved,_=restore(ROOT/'pilot');assert saved['cursor']['phase']=='completed'
    engine=Engine();events=Events([14]);case=cases('train')[0];emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda();checks={}
    for chunk in range(6):
        data=torch.load(DATA/'trajectories'/case['id']/f'chunk{chunk:02d}.pt',weights_only=True)
        if chunk==5:
            def callback(current,q,k,v):
                if current['layer']!=14:return
                ids=torch.linspace(0,4607,data['queries'].shape[-2],device=q.device).long();assert torch.equal(q[0,ids].transpose(0,1).cpu(),data['queries'][0,14])
                e=events.event(current,q,k,v)
                for method in METHODS:
                    model=Model(14,method).cuda();model.load_state_dict(saved['models'][method][14])
                    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(saved['optimizers'][method][14])
                    def update(m,o):
                        with torch.enable_grad():
                            o.zero_grad(set_to_none=True);loss=objective(m,e);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();m.project()
                    update(model,opt);checkpoint(ROOT/'real_resume'/method,dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict())))
                    update(model,opt)
                    other=Model(14,method).cuda();oopt=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=.0001)
                    state,_=restore(ROOT/'real_resume'/method);other.load_state_dict(state['model']);oopt.load_state_dict(state['optimizer']);update(other,oopt)
                    equal(model.state_dict(),other.state_dict());equal(opt.state_dict(),oopt.state_dict());checks[method]=True
            engine.callback=callback;engine.forward(data['noisy'][0].cuda(),emb,5,0);engine.callback=None
        else:engine.forward(data['clean'].cuda(),emb,chunk,4)
    atomic(ROOT/'real_resume_checks.json',dict(passed=True,trained_adam_resume_exact=checks,teacher_replay_exact=True))
if __name__=='__main__':main()
