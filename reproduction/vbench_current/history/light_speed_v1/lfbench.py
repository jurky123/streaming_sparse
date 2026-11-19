import sys,torch,json,time
import os
from pathlib import Path
from local_experiments.broad_common import noise
root=Path(__file__).resolve().parent
method,scene=sys.argv[1],int(sys.argv[2]); seed=701+scene
if method=='legacy':from legacy_light import build
else:from light import build
torch.set_num_threads(2)
@torch.inference_mode()
def run():
 chunks=7 if method=="short" else 41
 pipe=build(chunks) if method in ("legacy","light12","short") else build(chunks,window=18); layer=pipe.generator.model.blocks[0]; import wan.modules.sparse_attention as sa
 actual=sa._attention.apply;hits=[];fallback=[]
 def trace(*args):
  hits.append(int(args[5]) if isinstance(args[5],int) else None)
  return actual(*args)
 sa._attention.apply=trace
 dense=sa._dense_attention
 def trace_dense(*args,**kwargs):
  fallback.append(1);return dense(*args,**kwargs)
 sa._dense_attention=trace_dense
 emb=torch.load(root.parents[1]/'embeddings'/f'{"short" if chunks==7 else "long"}_{scene}.pt',weights_only=True).cuda()
 pipe.text_encoder.embeds=emb
 init=torch.cat([noise(seed,c,0) for c in range(chunks)],1)
 phases=[noise(seed,c,p).flatten(0,1) for c in range(chunks) for p in (1,2,3)]
 if method=='lfdense':
  import wan.modules.causal_model as causal
  causal.sparse_attention=lambda q,k,v,**kw:sa._dense_attention(q,k,v)
 times=[]
 for rep in range(4):
  calls=iter(phases); original=torch.randn_like
  torch.randn_like=lambda x,*a,**kw:next(calls)
  hits.clear();fallback.clear();torch.cuda.synchronize();t=time.monotonic()
  try:_,z=pipe.inference(init,['cached'],return_latents=True,profile=False)
  finally:torch.randn_like=original
  torch.cuda.synchronize();times.append(time.monotonic()-t)
  assert next(calls,None) is None and z.isfinite().all() and ((fallback and not hits) if method=="lfdense" else (hits and not fallback))
  data=dict(window=pipe.local_attn_size,times=times,sm=sa.DEVICE_SM,sparse_kernel_calls=len(hits),fallback=len(fallback),rope=layer.rope_kernel,scale_shift=layer.scale_shift_kernel,rms=layer.rmsnorm_kernel)
  (root/f'{method}_{scene}{os.getenv("REF_SUFFIX","")}.json' ).write_text(json.dumps(data))
run()
