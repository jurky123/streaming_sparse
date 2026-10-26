import argparse,json,statistics,time
from pathlib import Path
import torch
from ..common import config,atomic_json
from ..opt18.setup import ROOT as PARENT
from ..opt18.measure import Runner,Opt14
from ..opt17.model import JointGroupedRouter
from ..opt17.data import Events
from ..opt17.route import route as native_route
from ..opt16.grouping import order,gather,restore
from ..opt9.segmented_attention import attention
from ..opt5.probe import capture,timing
from .cache import Cache
from . import tile

ROOT=Path('streaming_sparse/v10/experiments/opt19_tile_kernels_20261025').resolve()

def measure(fn):
    graph,_=capture(fn);timing(graph,80)
    return statistics.median(timing(graph,60) for _ in range(3))

def nmse(x,y):return float((x.float()-y.float()).square().sum()/y.float().square().sum().clamp_min(1e-20))

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',default='Fast35');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=256
    _,m=config(PARENT);events=Events(m,24)
    prompt=sorted({x['row']['prompt_id'] for x in events.val})[0]
    item=next(x for x in events.val if x['row']['prompt_id']==prompt and x['row']['history_tokens']==14040 and x['row']['step']==3)
    model=JointGroupedRouter(m,24,a.variant).cuda().eval()
    model.load_state_dict(torch.load(PARENT/f'training/layer24/{a.variant}/best.pt',map_location='cuda',weights_only=True)['model'])
    runner=Runner(model,events,item,Opt14(m,24))
    q,r,truth,targets=runner.q,runner.raw,runner.truth,runner.targets
    frac,reps=model.options['candidate_fraction'],model.options['representatives']
    ix=order(q,model.tree('q'));qq=gather(q,ix)
    expected=runner.warm(frac,reps)
    dest=ROOT/f'probe_{a.variant}.json';rows=[]
    def record(row):
        rows.append(row);print(row,flush=True);atomic_json(dest,dict(sample=item['row'],rows=rows))
    record(dict(kind='native',warm_ms=measure(lambda:runner.warm(frac,reps)),dense_ms=measure(runner.dense),nmse=nmse(expected[:,targets],truth)))
    for precision in ['reference','tf32x3','fp16']:
        try:
            cache=Cache(runner.pipeline,r,precision)
            kv=cache.current(r['kc'],r['vc'])
            def native():
                kv=cache.current(r['kc'],r['vc']);return runner.execute(kv,frac,reps)
            out=native();assert torch.isfinite(out).all()
            record(dict(kind='cache',precision=precision,prepare_ms=measure(lambda:cache.current(r['kc'],r['vc'])),warm_ms=measure(native),nmse_vs_native=nmse(out,expected),nmse=nmse(out[:,targets],truth)))
            if precision=='reference':continue
            for g in [16,32]:
                for budget in [.15,.20,.25]:
                    for mode in ['shared','cap']:
                        def run():
                            kv=cache.current(r['kc'],r['vc']);ix=order(q,model.tree('q'));qq=gather(q,ix)
                            ids,fine,candidates=tile.route(qq,kv.state,frac,reps,g,budget,False,True)
                            mask=tile.expand(ids,q.shape[1],kv.state['key'].shape[1],g)
                            if mode=='cap':
                                native_mask=native_route(qq,kv.state,.15,frac,representatives=reps)
                                inter=mask&native_mask
                                mask=torch.where(inter.any(-1,keepdim=True),inter,mask)
                            n=kv.history_tokens
                            y=attention(qq,kv.k[:,:n],kv.v[:,:n],kv.k[:,n:],kv.v[:,n:],mask,group=g,pack_v=True,fuse_pack=True)
                            return restore(y,ix),mask
                        out,mask=run();assert torch.isfinite(out).all()
                        b=mask.shape[-1];cost=torch.full((b,),64.,device=q.device);cost[73::74]=8.
                        keep=float((mask*cost).sum()/(q.shape[0]*q.shape[1]*(r['kh'].shape[1]+r['kc'].shape[1])))
                        exec_ratio=float(torch.nn.functional.pad(mask,(0,0,0,(-q.shape[1])%g)).reshape(12,-1,g,b).any(2).sum()*g*64/(12*q.shape[1]*(r['kh'].shape[1]+r['kc'].shape[1])))
                        record(dict(kind='tile',precision=precision,group=g,budget=budget,mode=mode,keep=keep,executed=exec_ratio,warm_ms=measure(run),nmse=nmse(out[:,targets],truth),nmse_vs_native=nmse(out,expected)))
        except Exception as e:
            import traceback;traceback.print_exc();record(dict(kind='error',precision=precision,error=str(e)))

if __name__=='__main__':main()
