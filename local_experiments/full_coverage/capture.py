"""Resident single-GPU capture worker. No test data; one model load per worker."""
import argparse, collections, json, os, time
import torch
from .common import *
from ..broad_common import Engine
from ..review_v1.model import Model
from ..review_v1.cached_runtime import CachedLatestRuntime
from ..review_v1.ops import Config, Summary
from ..rollout_suite import generate_chunk
from ..learned_cache.io import RunLock

class CaptureRuntime(CachedLatestRuntime):
    def __init__(self,*args,**kwargs):
        self.collector=None
        super().__init__(*args,**kwargs)
    def selected_attention(self,q,k,v,s,basis,selection):
        state=self.engine.current
        if self.collector and state['chunk'] in (2,5,8) and state['step'] in (0,2):
            self.collector(state['layer'],q,k,v,s)
        return super().selected_attention(q,k,v,s,basis,selection)

class Collector:
    def __init__(self,case,runtime): self.case=case;self.rt=runtime;self.files=[];self.counts=collections.Counter()
    @torch.no_grad()
    def __call__(self,l,q,k,v,s):
        case=self.case;state=self.rt.engine.current;c=state['chunk'];step=state['step'];hp=self.rt.stores[l].n
        folder=OUT/'data'/case['id'];folder.mkdir(parents=True,exist_ok=True)
        t0=(17*l+13*c+7*step+case['seed'])%72
        # Compute canonical teacher one head/tile at a time; bound GPU lifetime.
        for h in HEADS:
            tag=f'c{c}_l{l}_s{step}_h{h}';kvpath=folder/(tag+'_kv.pt')
            ss={n:getattr(s,n)[h:h+1].cpu() for n in NAMES}
            if not kvpath.exists():
                ids=torch.cat((self.rt.stores[l].token_ids[h,:hp],torch.arange(c*4608,(c+1)*4608,device=q.device).reshape(72,64)))
                atomic(kvpath,dict(k=k[h].cpu(),v=v[h].cpu(),ids=ids.cpu(),summary=ss,history_pages=hp),True)
            for tile in (t0,(t0+36)%72):
                path=folder/(tag+f'_t{tile}.pt')
                if not path.exists():
                    query=q[h,tile*64:(tile+1)*64];qd=query.double()
                    logits=torch.einsum('qd,ptd->qpt',qd,k[h].double())/128**.5
                    lz=logits.logsumexp(-1)
                    op=torch.einsum('qpt,ptd->qpd',logits.softmax(-1),v[h].double())
                    ls=(torch.einsum('qd,prd->qpr',qd,s.keys[h].double())/128**.5).logsumexp(-1)+__import__('math').log(16)
                    scale=torch.maximum(lz.amax(1),ls.amax(1))-300
                    Z=(lz-scale[:,None]).exp();N=op*Z[...,None]
                    dense=(lz.softmax(1)[...,None]*op).sum(1)
                    assert torch.isfinite(N).all() and torch.isfinite(dense).all()
                    atomic(path,dict(q=query.cpu(),N=N.cpu(),Z=Z.cpu(),dense=dense.cpu(),
                        page_logz=lz.cpu(),page_output=op.cpu(),log_scale=scale.cpu(),
                        kv_file=kvpath.name,own=hp+tile,trajectory=case['id'],split=case['split'],
                        layer=l,head=h,chunk=c,step=step,tile=tile,scale_version='page_log_center_v3'),True)
                self.files.append(str(path.relative_to(OUT)));self.counts[f'l{l}h{h}']+=1
        atomic(OUT/'trajectories'/case['id']/'progress.json',dict(events=len(self.files),layer=l,chunk=c,step=step))

@torch.no_grad()
def run(worker=0,workers=1):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=protocol(); lock=RunLock(OUT/f'workers/capture{worker}');rt=None
    try:
        todo=[c for i,c in enumerate(p['cases']) if i%workers==worker and not (OUT/'trajectories'/c['id']/'complete.json').exists()]
        if not todo:return
        atomic(OUT/f'workers/capture{worker}/status.json',dict(status='loading',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        engine=Engine();engine.wrapper.requires_grad_(False)
        model=Model(str(BASE));model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.eval().requires_grad_(False)
        centers={l:torch.zeros(12,128,device='cuda') for l in LAYERS}
        rt=CaptureRuntime(engine,model,'no_downstream',centers,config=Config())
        for case in todo:
            with_lock=RunLock(OUT/'trajectories'/case['id'])
            try:
                rt.reset();collector=Collector(case,rt);rt.collector=collector
                emb=torch.load(E24/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                begin=time.perf_counter()
                for chunk in range(9):
                    generate_chunk(engine,emb,case['seed'],chunk)
                    atomic(OUT/f'workers/capture{worker}/status.json',dict(status='capturing',case=case['id'],chunk=chunk,events=len(collector.files),seconds=time.perf_counter()-begin,pid=os.getpid()))
                    print(case['id'],chunk,len(collector.files),round(time.perf_counter()-begin,2),flush=True)
                assert len(collector.files)==4176 and len(collector.counts)==348 and set(collector.counts.values())=={12}
                atomic(OUT/'trajectories'/case['id']/'complete.json',dict(case=case,events=len(collector.files),files=collector.files,coverage=dict(collector.counts),seconds=time.perf_counter()-begin,protocol_sha=digest(OUT/'protocol.json')))
            finally:with_lock.close()
        atomic(OUT/f'workers/capture{worker}/status.json',dict(status='complete',pid=os.getpid()))
    except BaseException as exc:
        atomic(OUT/f'workers/capture{worker}/status.json',dict(status='failed',error=repr(exc),pid=os.getpid()));raise
    finally:
        if rt is not None:rt.close()
        lock.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--worker',type=int,default=0);a.add_argument('--workers',type=int,default=1);v=a.parse_args();run(v.worker,v.workers)
