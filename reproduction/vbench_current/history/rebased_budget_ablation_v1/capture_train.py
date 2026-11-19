import json,math,time
from pathlib import Path
import torch
from scan import ROOT
from phase import set_phase
from anchor_policy import install
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1 import legacy_components
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.full_coverage.training_execution import load_cpu,to_device

@torch.no_grad()
def run():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'train_pilot';folder.mkdir(exist_ok=True)
    start=time.monotonic()
    cfg=config('intersection',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg);install(rt,'frame_fixed15')
    emb=torch.load(Path('artifacts/e24_joint_representatives_v1/embeddings/forest_run.pt'),weights_only=True).cuda()
    times=[]
    def collect(q,k,v,s,basis,ids,workspace):
        c,l,st=[engine.current[x] for x in ('chunk','layer','step')]
        if c in (2,6,20,40) and l==5 and st==0:
            t=time.monotonic();h=4;tile=(c*7+13)%72;p=s.mean.shape[1];hp=p-72
            kr=k[0,:,h].reshape(p,64,128).contiguous();vr=v[0,:,h].reshape_as(kr).contiguous()
            query=q[0,64*tile:64*(tile+1),h].contiguous()
            z=torch.einsum('qd,ptd->qpt',query.double(),kr.double())/math.sqrt(128)
            lz=z.logsumexp(-1);val=torch.einsum('qpt,ptd->qpd',z.softmax(-1),vr.double())
            rep=(query.double()@s.keys[h].flatten(0,1).double().T).reshape(64,p,4)/math.sqrt(128)
            shift=torch.maximum(lz.amax(-1),rep.logsumexp(-1).amax(-1)+math.log(16))-300
            mass=(lz-shift[:,None]).exp()
            ss={n:getattr(s,n)[h:h+1].cpu() for n in ('keys','stats','mean','detail')}
            token_ids=torch.cat([rt.stores[l].token_ids[h,:hp],torch.arange(c*4608,(c+1)*4608,device=q.device).reshape(72,64)])
            assert torch.equal(token_ids[:24].sort().values.flatten().sort().values,torch.arange(1536,device=q.device))
            kv=f'c{c}_kv.pt';ev=folder/f'c{c}_teacher.pt'
            torch.save(dict(k=kr.cpu(),v=vr.cpu(),summary=ss,history_pages=hp,ids=token_ids.cpu()),folder/kv)
            torch.save(dict(split='train',trajectory='forest_run_113',seed=113,chunk=c,step=st,head=h,layer=l,tile=tile,own=hp+tile,kv_file=kv,
                            q=query.cpu(),N=(val*mass[...,None]).cpu(),Z=mass.cpu(),dense=(lz.softmax(-1)[...,None]*val).sum(1).cpu(),
                            page_logz=lz.cpu(),log_scale=shift.cpu(),anchor_offset=max(0,c*3-15)),ev)
            d=to_device(load_cpu(ev));torch.testing.assert_close(d['N'].sum(1)/d['Z'].sum(1)[:,None],d['dense'],atol=1e-9,rtol=1e-9)
            times.append(time.monotonic()-t)
        return engine.dense(q,k,v)
    legacy_components.attention=collect
    rt.reset()
    for c in range(41):
        if c>=6:
            if c==6:canonical=torch.cat([cache['k'][:,:1536] for cache in engine.cache])
            set_phase(rt,canonical,3*c-15)
        generate_chunk(engine,emb,113,c)
    torch.cuda.synchronize()
    (folder/'status.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,records=len(times),capture_seconds=times)))
run()
