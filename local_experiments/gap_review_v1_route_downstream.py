"""Hard-route counterfactual pilot: frozen summaries/backbone, two-step x0 labels."""
import argparse,json,os,time
from pathlib import Path
import torch
from .broad_common import Engine,noise,ROOT as DATA
from .rollout_suite import generate_chunk
from .review_v1.model import Model
from .review_v1.downstream_route import DownstreamRouteRuntime
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/review_v1_20261102/route_downstream')
SOURCE='artifacts/review_v1_20261102/route_expanded/model.pt'
LAYERS=[1,8,15,22,28,29];HEADS=[0,5,11]

@torch.no_grad()
def collect(case):
    root=ROOT/'labels'/case;lock=RunLock(root);rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        identity=dict(source=digest(SOURCE),code=digest(__file__),runtime=digest('local_experiments/review_v1/downstream_route.py'),case=case,seed=1,layers=LAYERS,heads=HEADS,chunks=[2,5],objective='two-step x0 NMSE, hard one-page-per-tile swap in one head; frozen student history')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/'complete.json').exists():return
        engine=Engine();engine.wrapper.requires_grad_(False)
        m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m.load_state_dict(torch.load(SOURCE,map_location='cuda',weights_only=False)['model']);m.requires_grad_(False)
        rt=DownstreamRouteRuntime(engine,m,'full',{})
        emb=torch.load(DATA/'embeddings'/f'{case}.pt',weights_only=True).cuda()
        for chunk in range(6):
            if chunk in [2,5]:
                x=noise(1,chunk,0);eps=noise(1,chunk,1)
                # Layer0 uses the original cache writer; restore it before EVERY trial.
                zero={k:v.clone() for k,v in engine.cache[0].items() if torch.is_tensor(v)}
                ends=[{k:c[k].clone() for k in ('global_end_index','local_end_index')} for c in engine.cache]
                stores=[(s.n,s.write,s.commits) for s in rt.stores.values()]
                def restore():
                    for k,v in zero.items():engine.cache[0][k].copy_(v)
                    for c,e in zip(engine.cache,ends):
                        for k,v in e.items():c[k].copy_(v)
                    assert stores==[(s.n,s.write,s.commits) for s in rt.stores.values()]
                def two(dense=False,intervention=None):
                    restore();rt.sparse=not dense;rt.intervention=intervention;rt.observation=None
                    first=engine.forward(x,emb,chunk,0)
                    nxt=engine.wrapper.scheduler.add_noise(first.flatten(0,1),eps.flatten(0,1),torch.full((3,),engine.steps[1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
                    out=engine.forward(nxt,emb,chunk,1)
                    return out
                target=two(True);base=two();repeat=two(intervention=(LAYERS[0],HEADS[0],0,False))
                assert torch.equal(base,repeat),'Counterfactual replay is not bitwise deterministic'
                del repeat
                norm=target.double().square().mean().clamp_min(1e-30)
                loss=lambda y:float((y.double()-target.double()).square().mean()/norm)
                base_loss=loss(base)
                atomic(root/f'chunk{chunk}_control.json',dict(base_loss=base_loss,replay_bitwise_equal=True))
                for layer in LAYERS:
                    for head in HEADS:
                        path=root/f'c{chunk}_l{layer}_h{head}.json'
                        if path.exists():continue
                        disk_guard(ROOT,50,2)
                        atomic(root/'status.json',dict(pid=os.getpid(),chunk=chunk,layer=layer,head=head,updated=time.time()))
                        # Offset alternates across chunks to cover more than the nearest rejected page.
                        out=two(intervention=(layer,head,0 if chunk==2 else 2,True))
                        row=dict(case=case,chunk=chunk,layer=layer,head=head,base_loss=base_loss,swap_loss=loss(out),**rt.observation)
                        row['gain']=row['base_loss']-row['swap_loss'];assert all(torch.isfinite(torch.tensor(row[k],dtype=torch.float64)) for k in ['gain','base_loss','swap_loss'])
                        atomic(path,row);print(json.dumps(row),flush=True)
                restore();rt.sparse=True;rt.intervention=None
                del zero,target,base,out
            path=root/f'clean{chunk}.pt'
            if path.exists():engine.forward(torch.load(path,map_location='cuda',weights_only=True),emb,chunk,4)
            else:atomic(path,generate_chunk(engine,emb,1,chunk).cpu(),True)
        atomic(root/'complete.json',dict(events=36,identity=identity))
    finally:
        if rt is not None:rt.close()
        lock.close()

def fit():
    torch.set_num_threads(4);root=ROOT/'fit';lock=RunLock(root)
    try:
        rows=[]
        for case in ['city_walk','reef_fish']:
            assert (ROOT/'labels'/case/'complete.json').exists()
            rows += [json.loads(p.read_text()) for p in (ROOT/'labels'/case).glob('c*_l*_h*.json')]
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);state=saved['model'];before={k:v.clone() for k,v in state.items()};stats=[]
        for layer in LAYERS:
            key=f'layers.{layer}.logtheta'
            for head in HEADS:
                rr=[r for r in rows if r['layer']==layer and r['head']==head]
                x=torch.tensor([r['delta'] for r in rr],dtype=torch.float64);g=torch.tensor([r['gain'] for r in rr],dtype=torch.float64)
                old=torch.nn.functional.softplus(state[key][head].double());old=old/old.mean()
                scale=(x*old).abs().sum(-1).mean().clamp_min(1e-30)
                x=x/scale;weights=g.abs()/g.abs().max().clamp_min(1e-30)
                initial=old.expm1().log();raw=initial.clone().requires_grad_();opt=torch.optim.Adam([raw],lr=.02)
                for _ in range(250):
                    theta=torch.nn.functional.softplus(raw);theta=theta/theta.mean()
                    # Positive gain means the incoming page(s) deserve a higher score.
                    loss=(torch.nn.functional.softplus(-g.sign()*(x@theta))*weights).mean()+.05*(raw-initial).square().mean()
                    opt.zero_grad();loss.backward();opt.step()
                theta=torch.nn.functional.softplus(raw.detach());theta=theta/theta.mean()
                state[key][head]=theta.expm1().log().to(state[key].dtype)
                stats.append(dict(layer=layer,head=head,events=len(rr),train_pair_accuracy=float(((x@theta)*g>0).double().mean()),old_pair_accuracy=float(((x@old)*g>0).double().mean()),theta=theta.tolist()))
        assert all(torch.equal(v,before[k]) for k,v in state.items() if not k.endswith('.logtheta'))
        for layer in range(1,30):
            key=f'layers.{layer}.logtheta'
            for head in range(12):
                if layer not in LAYERS or head not in HEADS:assert torch.equal(state[key][head],before[key][head])
        atomic(root/'model.pt',dict(model=state,source=digest(SOURCE),scope='pilot six layers three heads; all other parameters unchanged'),True)
        atomic(root/'result.json',dict(events=len(rows),heads=stats,heldout='horse_field and traffic_crossing video; never used to fit',speed_valid=False))
    finally:lock.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['city_walk','reef_fish']);ap.add_argument('--fit',action='store_true');a=ap.parse_args()
    if a.fit:fit()
    else:assert a.case;collect(a.case)
