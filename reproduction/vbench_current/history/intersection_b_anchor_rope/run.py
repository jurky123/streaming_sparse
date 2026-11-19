import hashlib,json,sys,time
from pathlib import Path
import torch
from anchor_policy import install
from local_experiments.anchor_rebase import shift_temporal_
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.stream_eval_common import decode,native_vae,save_video

def set_phase(rt,canonical,offset):
    # Shift original frame-0 K from scratch, in global coordinates. With an
    # 18-frame window and 3-frame chunk the Q-to-anchor distances are 15..17.
    k=canonical.clone()
    shift_temporal_(k,1536,rt.engine.wrapper.model.freqs,offset)
    for l,cache in enumerate(rt.engine.cache):
        cache['k'][0,:1536].copy_(k[l])
        if l in rt.stores:
            def grouped(x):
                return x.transpose(0,1).reshape(12,24,64,128)
            summary=rt.summary_graph(rt.model[l],grouped(k[l]),grouped(cache['v'][0,:1536]))
            for field in ('keys','stats','detail','mean'):
                getattr(rt.stores[l].summary,field)[:,:24].copy_(getattr(summary,field))
            if l in rt.pset_layers:
                rt.write_encoding(l,rt.encode(l,summary),start=0)
@torch.no_grad()
def main(mode,i,seed):
    out=Path(__file__).parent
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('intersection',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    install(rt,'frame_fixed15')
    emb=torch.load(out.parent.parent/'embeddings'/f'long_{i}.pt',weights_only=True).cuda()
    for n in (7,41):
        rt.reset()
        torch.cuda.synchronize()
        t=time.monotonic()
        zs=[]
        for c in range(n):
            if mode=='rebased' and c>=6:
                if c==6:
                    canonical=torch.cat([cache['k'][:,:1536] for cache in engine.cache],0)
                set_phase(rt,canonical,3*c-15)
            zs.append(generate_chunk(engine,emb,seed,c))
        z=torch.cat(zs,1)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-t
    assert z.isfinite().all()
    def digest(x):
        return hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    stem=f'{mode}_{i}_{seed}'
    data=dict(seconds=elapsed,prefix=digest(z[:,:18]),first_rolled=digest(z[:,18:21]))
    (out/(stem+'.json')).write_text(json.dumps(data))
    print(stem,data,flush=True)
    save_video(out/(stem+'.mp4'),decode(native_vae(),z,480))
main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
