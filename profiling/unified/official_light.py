"""Historical upstream LightForcing pipeline, with benchmark anchor constraints."""
import sys,time,json,types
from pathlib import Path
import torch
here=Path(__file__).resolve().parent
repo=here.parents[1]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from light import build
from local_experiments.broad_common import noise

@torch.inference_mode()
def run(n):
    torch.set_num_threads(2)
    pipe=build(n,window=18)
    for block in pipe.generator.model.blocks:
        block.self_attn.long_video_gen=True
        block.self_attn.sink_size=1
    pipe.text_encoder.embeds=torch.load(repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt',weights_only=True).cuda()
    init=torch.cat([noise(701,c,0) for c in range(n)],1)
    phases=[noise(701,c,p).flatten(0,1) for c in range(n) for p in (1,2,3)]
    import wan.modules.causal_model as causal
    import wan.modules.kernel as kernel
    import wan.modules.sparse_attention as sa
    # Keep the upsteam scores and top-k budget; reserve 24 exact KV blocks.
    class AnchorTorch:
        def __getattr__(self,name):return getattr(torch,name)
        def topk(self,scores,k,*args,**kwargs):
            scores[...,:24]=float('inf')
            return torch.topk(scores,max(24,k),*args,**kwargs)
    for func in ('get_sm_80_120_block_map_1stage','get_sm_80_120_block_map_2stage'):
        old=getattr(kernel,func)
        scope=dict(old.__globals__,torch=AnchorTorch())
        patched=types.FunctionType(old.__code__,scope,old.__name__,old.__defaults__,old.__closure__)
        def mapping(*args,_fn=patched,**kw):
            mask,lut,_=_fn(*args,**kw)
            return mask,lut,lut.size(-1)
        setattr(kernel,func,mapping)
    calls=[0,0]
    original=causal.sparse_attention
    def dispatch(q,k,v,chunk_id,**kw):
        if chunk_id<2:
            calls[0]+=1
            return sa._dense_attention(q,k,v)
        calls[1]+=1
        return original(q,k,v,chunk_id=chunk_id,**kw)
    causal.sparse_attention=dispatch
    times=[]
    for i in range(3):
        calls[:]=[0,0]
        stream=iter(phases);rand=torch.randn_like
        torch.randn_like=lambda *a,**kw:next(stream)
        torch.cuda.synchronize();t=time.monotonic()
        try:_,z=pipe.inference(init,['cached'],return_latents=True,profile=False)
        finally:torch.randn_like=rand
        torch.cuda.synchronize();times.append(time.monotonic()-t)
        assert calls==[300,(n-2)*150] and next(stream,None) is None and z.isfinite().all()
        print(n,times[-1],calls,flush=True)
    torch.save(z.cpu(),here/f'lightforcing_{n}.pt')
    (here/f'lightforcing_{n}.json').write_text(json.dumps(dict(times=times,calls=calls)))
run(int(sys.argv[1]))
