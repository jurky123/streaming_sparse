"""Single-step native-flow teacher distillation; isolated side experiment."""
import argparse,gc,json,os,signal,time
from pathlib import Path
import torch
from .broad_common import Engine,ROOT as DATA
from .rollout_suite import generate_chunk
from .review_v1.model import Model
from .review_v1.runtime import StreamingRuntime
from .all_sparse_step_context_v2 import install_step_context
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/flow_single_20261104')

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--case',required=True);a=ap.parse_args()
    root=ROOT/'train'/a.case;lock=RunLock(root);rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ident=dict(initial=digest(a.initial),code=digest(__file__),runtime=digest('local_experiments/review_v1/runtime.py'),case=a.case,seed=1,chunks=[2,5],steps=[0,1,2,3],loss='native flow MSE, same x/t/text/history; dense stopgrad; frozen backbone and hard router')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==ident
        else:atomic(root/'identity.json',ident)
        if (root/'complete.json').exists():return
        m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m.load_state_dict(torch.load(a.initial,map_location='cuda',weights_only=False)['model'])
        for l in range(1,30):m[l].freeze_stage('distill')
        params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=1e-5)
        engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False
        rt=StreamingRuntime(engine,m);install_step_context(rt);normal=engine.forward
        def pair(x,emb,chunk,step):
            engine.current.update(chunk=chunk,step=step)
            t=0. if step==4 else engine.steps[step]
            return engine.wrapper(noisy_image_or_video=x,conditional_dict={'prompt_embeds':emb},timestep=torch.full((1,3),t,device='cuda',dtype=torch.float32),kv_cache=engine.cache,crossattn_cache=engine.cross,current_start=chunk*4608)
        def forward(x,emb,chunk,step):
            rt.sparse=True;rt.training_event=False
            if chunk not in [2,5] or step==4:return normal(x,emb,chunk,step)
            tag=f'c{chunk}_s{step}';path=root/f'{tag}.pt'
            # Replay each event with its historical weights before loading the update.
            zero={k:v.clone() for k,v in engine.cache[0].items() if torch.is_tensor(v)}
            ends=[{k:c[k].clone() for k in ('global_end_index','local_end_index')} for c in engine.cache]
            def restore():
                for k,v in zero.items():engine.cache[0][k].copy_(v)
                for c,e in zip(engine.cache,ends):
                    for k,v in e.items():c[k].copy_(v)
            hard_flow,hard=pair(x,emb,chunk,step)
            if path.exists():
                saved=torch.load(path,map_location='cuda',weights_only=False);assert saved['identity']==ident
                m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rt.refresh_summaries();return hard
            disk_guard(ROOT,50,2);atomic(root/'status.json',dict(status='training',event=tag,pid=os.getpid(),time=time.time()))
            restore();rt.sparse=False;target,_=pair(x,emb,chunk,step);rt.sparse=True;restore()
            rt.training_event=True;opt.zero_grad(set_to_none=True);start=time.time()
            try:
                with torch.enable_grad():
                    pred,_=pair(x,emb,chunk,step)
                    fidelity=float((pred.detach().float()-hard_flow.float()).square().sum()/hard_flow.float().square().sum().clamp_min(1e-12))
                    assert fidelity<1e-6, fidelity
                    loss=(pred.float()-target.float()).square().mean();loss.backward()
                norms={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None}
                assert len(norms)==203 and all(torch.isfinite(p.grad).all() for p in params)
                assert all(p.grad is None for p in engine.wrapper.parameters())
                assert all(m[l].logtheta.grad is None for l in range(1,30))
                grad=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True));opt.step();rt.refresh_summaries()
                row=dict(event=tag,flow_mse=float(loss),flow_nmse=float(loss/target.float().square().mean().clamp_min(1e-12)),forward_nmse=fidelity,grad_norm=grad,gradient_parameters=len(norms),seconds=time.time()-start)
                atomic(root/f'{tag}.json',row)
                atomic(path,dict(model={n:p.detach().cpu() for n,p in m.state_dict().items()},optimizer=opt.state_dict(),identity=ident),True)
                print(json.dumps(row),flush=True)
            finally:rt.training_event=False
            return hard
        engine.forward=forward;emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda()
        for chunk in range(6):generate_chunk(engine,emb,1,chunk)
        assert all((root/f'c{c}_s{s}.pt').exists() for c in [2,5] for s in range(4))
        atomic(root/'model.pt',dict(model={n:p.detach().cpu() for n,p in m.state_dict().items()},identity=ident),True)
        atomic(root/'complete.json',dict(status='complete',events=8,identity=ident))
    except BaseException as e:atomic(root/'status.json',dict(status='failed',error=repr(e)));raise
    finally:
        if rt is not None:rt.close()
        lock.close();gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
