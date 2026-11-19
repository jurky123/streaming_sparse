"""Official square-kernel integration, with explicit query-padding overhead."""
import argparse,sys,torch,os
from pathlib import Path
from .broad_common import Engine
from .stream_eval_common import benchmark,REPO
from .learned_cache.io import digest
@torch.no_grad()
def main():
 from local_experiments.stream_eval_common import identify_gpu_process
 identify_gpu_process()
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['pisa','solattn','svgear'],required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--chunks',type=int,default=41);a=p.parse_args()
 if a.method!='svgear':sys.path.insert(0,'/tmp/streaming_sparse_baselines_deps')
 ext=Path('/home/shuchenweng/jzk/streaming_sparse/external_baselines')
 if a.method=='pisa':
  sys.path.insert(0,str(ext/'PISA'));from piecewise_attn import piecewise_sparse_attention as kernel
 elif a.method=='solattn':
  sys.path.insert(0,str(ext/'Sana/techniques/sparse_backends'));from sol_attn import sol_attn as kernel
 else:
  os.environ['TORCH_CUDA_ARCH_LIST']='8.0';os.environ['CUDA_HOME']='/usr/local/cuda-12.2';os.environ['MAX_JOBS']='2'
  from .svg_ear_stream import EAR
  kernel=EAR()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inputs={k:v.cuda() for k,v in torch.load(a.inputs,map_location='cpu',weights_only=True).items()}
 original_compile=torch.compile
 def compile_compat(fn,*args,**kwargs):
  if a.method in ('pisa','solattn') and getattr(fn,'__name__','')=='flex_attention':
   def unused_training_path(*args,**kwargs):
    raise RuntimeError('Training-only flex attention must not execute in streaming KV inference')
   return unused_training_path
  return original_compile(fn,*args,**kwargs)
 try:
  torch.compile=compile_compat
  engine=Engine()
 finally:torch.compile=original_compile
 import wan.modules.causal_model as causal
 counts={'sparse_calls':0,'max_kv_tokens':0}
 def attention(q,k,v,*args,**kwargs):
  if engine.current['step']==4 or engine.current['chunk']<2 or engine.current['layer']==0:return engine.dense(q,k,v,*args,**kwargs)
  counts['sparse_calls']+=1;counts['max_kv_tokens']=max(counts['max_kv_tokens'],k.shape[1])
  if a.method=='svgear':return kernel(q,k,v,engine.current['layer'])
  # No synthetic K/V: pad only queries, keeping current query positions at tail.
  n=q.shape[1];padded=torch.zeros_like(k);padded[:,-n:]=q
  if a.method=='pisa':out=kernel(padded.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),density=.1).transpose(1,2)
  else:out=kernel(padded.contiguous(),k.contiguous(),v.contiguous(),tau=1.,sink_start=k.shape[1]-n,sink_tokens=n)
  return out[:,-n:].contiguous()
 causal.attention=attention
 def generate():
  engine.reset();torch.manual_seed(0);zs=[];counts.update(sparse_calls=0,max_kv_tokens=0)
  if a.method=='svgear':kernel.reset()
  for chunk in range(a.chunks):
   x=inputs['initial_noise'][:,chunk*3:chunk*3+3]
   for step in range(4):
    clean=engine.forward(x,inputs['prompt_embeds'],chunk,step)
    if step<3:x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),inputs['transition_noise'][chunk,step].flatten(0,1),torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
   engine.forward(clean,inputs['prompt_embeds'],chunk,4);zs.append(clean)
  return torch.cat(zs,1)
 metadata=dict(method=a.method,checkpoint_sha256=digest(REPO/'checkpoints/self_forcing_dmd.pt'),inputs_sha256=digest(a.inputs),height=512,width=768,history_latent_frames=15,chunk_latent_frames=3,chunks=41,query_padding=a.method!='svgear',scope='Official kernel with padded extra queries discarded; padding and extra computation included in timing; no claim of optimized rectangular-kernel performance.',density=.1 if a.method=='pisa' else None,tau=1. if a.method=='solattn' else None)
 if a.method=='svgear':metadata.update(scope='Official SVG-EAR clustering, error-aware routing and FlashInfer fusion; rectangular KV views adapted; no query padding.',query_centroids=300,key_centroids=1000,top_p=.9,kmeans_iterations=[50,2])
 metadata['benchmark_script_sha256']=digest(__file__)
 def storage():
  raw=sum(c[n].numel()*c[n].element_size() for c in engine.cache for n in ('k','v'))
  extra=sum(x.numel()*x.element_size() for pair in kernel.centroids.values() for x in pair) if a.method=='svgear' else 0
  return dict(self_kv_storage_bytes=raw,auxiliary_bytes=extra,**counts)
 if a.smoke:
  from .learned_cache.io import atomic
  latent=generate();assert torch.isfinite(latent).all()
  atomic(Path(a.output)/'smoke.json',dict(status='passed',shape=list(latent.shape),storage=storage()))
 else:
  assert a.chunks==41
  benchmark(a.method,a.output,generate,metadata,storage=storage)
if __name__=='__main__':main()
