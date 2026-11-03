"""All-layer two self-conditioned steps; frozen backbone, hard routing.

Replays student state. Summary/group-feature families get actual downstream
loss gradients, theta is calibrated separately afterwards (no route STE).
"""
import argparse,gc,json,os,signal,time
from pathlib import Path
import torch
from .broad_common import Engine,noise,ROOT as DATA
from .rollout_suite import generate_chunk
from .review_v1.model import Model
from .review_v1.runtime import StreamingRuntime
from .all_sparse_step_context_v2 import install_step_context
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/review_v1_20261102/distill')

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--case',default='city_walk',choices=['city_walk','reef_fish']);args=ap.parse_args()
    root=ROOT/args.case;lock=RunLock(root);rt=None;engine=None;stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        identity=dict(initial_sha=digest(args.initial),code=digest(__file__),runtime=digest('local_experiments/review_v1/runtime.py'),case=args.case,seed=1,events=[2,5],step=0,objective='full-token two-step x0 NMSE on student history, all29 layers, backbone frozen; no new route STE')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/'complete.json').exists():return
        m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m.load_state_dict(torch.load(args.initial,map_location='cuda',weights_only=False)['model'])
        for l in range(1,30):m[l].freeze_stage('distill')
        params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=1e-5);done=[]
        if (root/'latest.pt').exists():
            saved=torch.load(root/'latest.pt',map_location='cuda',weights_only=False);m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);done=saved['done'];assert saved['identity']==identity
        if done==[2,5]:
            atomic(root/'complete.json',dict(status='two_step_training_complete_pending_fresh_calibration_and_video',identity=identity));return
        if done:m.load_state_dict(torch.load(args.initial,map_location='cuda',weights_only=False)['model'])
        engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False
        rt=StreamingRuntime(engine,m);install_step_context(rt);base_forward=engine.forward
        def forward(x,emb,chunk,step):
            rt.sparse=True;rt.training_event=False;hard=base_forward(x,emb,chunk,step)
            if chunk not in [2,5] or step!=0 or stop[0]:return hard
            if chunk in done:
                m.load_state_dict(torch.load(root/f'after_chunk{chunk}.pt',map_location='cuda',weights_only=False)['model']);rt.refresh_summaries();return hard
            disk_guard(ROOT,50,1);atomic(root/'status.json',dict(status='training',chunk=chunk,step=step,pid=os.getpid(),completed=done))
            eps=noise(1,chunk,step+1);t=torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)
            def next_input(clean):return engine.wrapper.scheduler.add_noise(clean.flatten(0,1),eps.flatten(0,1),t).unflatten(0,(1,3))
            rt.sparse=False;one=base_forward(x,emb,chunk,step);target=base_forward(next_input(one),emb,chunk,step+1);rt.sparse=True
            rt.training_event=True;opt.zero_grad(set_to_none=True);start=time.time()
            try:
                with torch.enable_grad():
                    first=Engine.forward.__wrapped__(engine,x,emb,chunk,step);first.retain_grad()
                    fidelity=float((first.detach().float()-hard.float()).square().sum()/hard.float().square().sum().clamp_min(1e-12))
                    assert fidelity<1e-6,fidelity
                    pred=Engine.forward.__wrapped__(engine,next_input(first),emb,chunk,step+1)
                    loss=(pred.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-12)
                    loss.backward()
                norms={name:float(p.grad.norm()) for name,p in m.named_parameters() if p.requires_grad and p.grad is not None}
                assert len(norms)==29*7 and all(torch.isfinite(p.grad).all() for p in params)
                assert first.grad is not None and torch.isfinite(first.grad).all() and first.grad.norm()>0
                assert all(p.grad is None for p in engine.wrapper.parameters())
                torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();rt.refresh_summaries()
                row=dict(loss=float(loss.detach()),hard_forward_nmse=fidelity,gradient_norms=norms,first_step_gradient_norm=float(first.grad.norm()),wall_seconds=time.time()-start,speed_valid=False)
                done.append(chunk);atomic(root/f'chunk{chunk}.json',row)
                atomic(root/f'after_chunk{chunk}.pt',dict(model={n:p.detach().cpu() for n,p in m.state_dict().items()}),True)
                atomic(root/'latest.pt',dict(model={n:p.detach().cpu() for n,p in m.state_dict().items()},optimizer=opt.state_dict(),done=done,identity=identity),True)
                print(json.dumps(dict(chunk=chunk,loss=row['loss'],gradient_parameters=len(norms))),flush=True)
            finally:rt.training_event=False
            return hard
        engine.forward=forward;emb=torch.load(DATA/'embeddings'/f'{args.case}.pt',weights_only=True).cuda()
        for chunk in range(6):
            if stop[0]:atomic(root/'status.json',dict(status='interrupted',done=done));return
            generate_chunk(engine,emb,1,chunk)
        assert done==[2,5],done
        atomic(root/'complete.json',dict(status='two_step_training_complete_pending_fresh_calibration_and_video',identity=identity))
    except BaseException as e:atomic(root/'status.json',dict(status='failed',error=repr(e)));raise
    finally:
        if rt is not None:rt.close()
        lock.close();gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
