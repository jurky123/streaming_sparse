"""Bounded causal rolling history with absolute positions; identical four-chunk prefix protocol."""
import torch,time
from streaming_sparse.v08.common import protocol
from streaming_sparse.v08.backbone import native_steps
from streaming_sparse.v08.cache import CacheIdentity,LayerKV,validate

@torch.no_grad()
def trajectory(network,embeds,pid,seed,sparse,height,width,num_chunks,progress=None):
 p=protocol();device=next(network.model.parameters()).device;dtype=next(network.model.parameters()).dtype;steps,_=native_steps(network.wrapper)
 shape=(p['latent_shape'][0],height//8,width//8);seg=3*(height//16)*(width//16);rng=torch.Generator(device=device).manual_seed(seed)
 noise=torch.randn((1,num_chunks*3,*shape),generator=rng,device=device,dtype=dtype)
 transitions=torch.stack([torch.stack([torch.randn((3,*shape),generator=rng,device=device,dtype=dtype).unflatten(0,(1,3))for _ in range(3)])for _ in range(num_chunks)])
 cache=();chunks=[];times=[]
 def identity(chunk,kind):return CacheIdentity(network.model_version,network.router_version,f'{pid}.seed{seed}',f'{kind}.chunk{chunk}',max(0,chunk-3)*seg,chunk*seg,p['context_noise'],owner='student'if sparse else'teacher')
 for chunk in range(num_chunks):
  torch.cuda.synchronize();start=time.perf_counter()
  if chunk:
   ci=identity(chunk-1,'context');cache=tuple(LayerKV(x.k,x.v,ci,x.layer)for x in cache)
   validate(cache,ci,min(chunk-1,3)*seg)
   result=network.forward_chunk(previous,embeds,p['context_noise'],cache,(chunk-1)*3,sparse=False,step=0,identity=ci,checkpoint=False,collect_aux=False,write_cache=True)
   cache=result['cache']
  di=identity(chunk,'denoise');cache=tuple(LayerKV(x.k,x.v,di,x.layer)for x in cache);validate(cache,di,min(chunk,3)*seg)
  current=noise[:,chunk*3:(chunk+1)*3]
  for step,ts in enumerate(steps):
   result=network.forward_chunk(current,embeds,ts,cache,chunk*3,sparse=sparse,step=step,identity=di,checkpoint=False,collect_aux=False,write_cache=False);x0=result['x0']
   if not torch.isfinite(x0).all():raise FloatingPointError('Nonfinite rolling video latent')
   if step<3:current=network.wrapper.scheduler.add_noise(x0.flatten(0,1),transitions[chunk,step].flatten(0,1),torch.full((3,),steps[step+1],device=device)).unflatten(0,(1,3))
  previous=x0;chunks.append(x0.cpu());torch.cuda.synchronize();elapsed=time.perf_counter()-start;times.append(elapsed)
  if progress:progress(chunk+1,num_chunks,elapsed)
 return torch.cat(chunks,1),times
