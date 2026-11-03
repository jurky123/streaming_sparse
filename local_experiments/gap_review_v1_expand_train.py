"""Stage-1 fixed-mask summary pilot on real TRAIN trajectories; resumable events.

All new sparse families except theta train via pooling. No backbone update;
no route STE. This is attention-output pretraining, NOT downstream acceptance.
"""
import argparse,gc,json,os
from pathlib import Path
import torch
from .broad_common import Engine,ROOT as DATA
from .gap_gain_vector64_runtime import GapGainVector64Runtime
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .review_v1.model import Model
from .review_v1.ops import Summary,SummaryOps,AttentionOps
ROOT=Path('artifacts/review_v1_20261102/summary_expand')

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['city_walk','reef_fish'],required=True);ap.add_argument('--initial',required=True);args=ap.parse_args()
    root=ROOT/args.case;lock=RunLock(root)
    engine=None;rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        layers=list(range(1,30))
        identity=dict(ops=digest('local_experiments/review_v1/ops.py'),runner=digest(__file__),checkpoint=digest('artifacts/four_gaps_20261031/shared/epoch1.pt'),initial=digest(args.initial),layers=layers,case=args.case,seed=0,chunks=[2,5,8],steps=[0,2],updates_per_event=2,objective='fixed old masks, sampled same-state dense attention output; stage1 only')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/'complete.json').exists():return
        model=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');model.load_state_dict(torch.load(args.initial,weights_only=False,map_location='cuda')['model']);done=[];optimizers={}
        for l in layers:model[l].freeze_stage('summary');optimizers[l]=torch.optim.Adam([p for p in model[l].parameters() if p.requires_grad],lr=1e-4)
        if (root/'latest.pt').exists():
            saved=torch.load(root/'latest.pt',weights_only=False,map_location='cuda');assert saved['identity']==identity
            model.load_state_dict(saved['model']);done=saved['done']
            for l in layers:optimizers[l].load_state_dict(saved['optimizers'][l])
        engine=Engine();engine.wrapper.requires_grad_(False)
        rt=GapGainVector64Runtime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',weights_only=False,map_location='cpu')['model'])
        original=rt.decision;tiles=torch.tensor([0,36],device='cuda');offsets=torch.linspace(0,63,16,device='cuda').round().long()
        def decision(l,q,pk,base,detail,cov,vvar):
            result=original(l,q,pk,base,detail,cov,vvar);chunk=engine.current['chunk'];step=engine.current['step'];key=f'c{chunk}_l{l}_s{step}'
            if l not in layers or chunk not in [2,5,8] or step not in [0,2] or key in done:return result
            disk_guard(root,50,1);atomic(root/'status.json',dict(status='training',event=key,done=done,pid=os.getpid()))
            tiles=torch.tensor([(17*l+13*chunk+7*step)%72],device='cuda')
            p=pk.shape[1];history=p-72;m=model[l]
            k=engine.cache[l]['k'][0,:p*64].transpose(0,1).reshape(12,p,64,128).detach()
            v=engine.cache[l]['v'][0,:p*64].transpose(0,1).reshape_as(k).detach()
            qs=q[0].transpose(0,1).reshape(12,72,64,128).index_select(1,tiles).index_select(2,offsets).flatten(1,2).detach()
            ids=result[0][0].index_select(1,tiles).long();mask=torch.zeros(12,1,p,device='cuda',dtype=torch.bool).scatter(-1,ids,True)
            mask=mask[:,:,None].expand(-1,-1,16,-1).reshape(12,16,p)
            def summary():
                hs=SummaryOps.build(m,k[:,:history],v[:,:history]);cs=SummaryOps.build(m,k[:,history:],v[:,history:],cheap=True)
                return Summary(*(torch.cat((getattr(hs,n),getattr(cs,n)),1) for n in ('keys','mean','detail','stats')))
            initial=summary();target,_=AttentionOps.evaluate(qs,k,v,initial,m.basis,torch.ones_like(mask));den=target.square().mean().clamp_min(1e-8)
            losses=[];norms={}
            for _ in range(2):
                with torch.enable_grad():
                    s=summary();out,_=AttentionOps.evaluate(qs,k,v,s,m.basis,mask)
                    loss=(out-target).square().mean()/den
                    optimizers[l].zero_grad();loss.backward()
                    norms={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None}
                    assert len(norms)==7 and all(v>0 for v in norms.values()) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);optimizers[l].step()
                losses.append(float(loss))
            out,_=AttentionOps.evaluate(qs,k,v,summary(),m.basis,mask);post=float((out-target).square().mean()/den)
            done.append(key)
            atomic(root/(key+'.json'),dict(losses_before_update=losses,loss_after=post,grad_norms=norms,pages=p))
            atomic(root/'latest.pt',dict(model={n:t.detach().cpu() for n,t in model.state_dict().items()},optimizers={l:o.state_dict() for l,o in optimizers.items()},done=done,identity=identity),True)
            print(json.dumps(dict(event=key,losses=losses,post=post)),flush=True)
            return result
        rt.decision=decision;emb=torch.load(DATA/'embeddings'/f'{args.case}.pt',weights_only=True).cuda()
        for chunk in range(9):generate_chunk(engine,emb,0,chunk)
        assert len(done)==len(layers)*6
        atomic(root/'complete.json',dict(status='stage1_expanded_complete_pending_heldout_and_downstream',events=len(done),identity=identity))
    except BaseException as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if rt is not None:rt.close()
        lock.close();gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
