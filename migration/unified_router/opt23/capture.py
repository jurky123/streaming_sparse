"""Real frozen Dense closed-loop QKV capture at requested spatial/temporal sizes."""
import argparse,json,types,time
from pathlib import Path
import torch
from .setup import ROOT,TRAINING,tag
from ..common import atomic_json,update,paired_seed
from ..opt21.network import RuntimeNetwork
from streaming_sparse.v08.cache import CacheIdentity,LayerKV,validate

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--chunks',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=ROOT/'capture'/tag(a.height,a.width,a.chunks);out.mkdir(parents=True,exist_ok=True)
 def status(state,**kw):update(out/'status.json',status=state,**kw);print(state,kw,flush=True)
 status('loading')
 from ..opt22.inference import load
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.common import protocol
 from streaming_sparse.v08.backbone import native_steps
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from ..opt21.rope import rope
 net,m=load();net.select('Dense');net.inspect_callback=None
 ds=ReplayDataset('val',require_complete=True);pid=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text())['prompt_ids'][0]
 sample=ds.load(next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 n=3*(a.height//16)*(a.width//16);maximum=(a.chunks-1)*n
 buffers=[(torch.empty((1,maximum,12,128),device='cuda',dtype=torch.bfloat16),torch.empty((1,maximum,12,128),device='cuda',dtype=torch.bfloat16)) for _ in net.model.blocks]
 # Only clean history appends to bounded workspaces. Denoising never writes them.
 original=RuntimeNetwork.forward_chunk
 import inspect
 source=inspect.getsource(original)
 source=source.replace('    def forward_chunk','def forward_chunk',1)
 lines=source.splitlines();source='\n'.join([lines[0]]+[line[4:] if line.startswith('    ') else line for line in lines[1:]])
 source=source.replace("if write_cache:new_cache.append(LayerKV(torch.cat((kh,k),1),torch.cat((vh,v),1),identity,layer,k,v))",'''if write_cache:
            bk,bv=self.scale_buffers[layer];end=start*(s//frames)+s;offset=end-s
            bk[:,offset:end].copy_(k);bv[:,offset:end].copy_(v)
            new_cache.append(LayerKV(bk[:,:end],bv[:,:end],identity,layer))''')
 # The source uses x length instead of an undeclared local s/frames.
 source=source.replace('end=start*(s//frames)+s;offset=end-s','nt=k.shape[1];end=start*(nt//3)+nt;offset=end-nt')
 namespace=dict(original.__globals__);exec(source,namespace);forward=namespace['forward_chunk'];net.scale_buffers=buffers
 def eager(self,*args,**kw):
  if self.needs_condition:self.condition(args[1])
  self.runtime_timestep=None;self.runtime_time_features=self.time_features(args[2]);return forward(self,*args,**kw)
 net.forward_chunk=types.MethodType(eager,net)
 original_block=net._block;captured=[]
 def block(block,layer,x,e0,grid,freqs,context,hk,hv,start,sparse,step,*args,**kw):
  take=layer in (1,14,24,29) and sparse and start==(a.chunks-1)*3 and step==3
  if not take:return original_block(block,layer,x,e0,grid,freqs,context,hk,hv,start,False,step,*args,**kw)
  values={};att=block.self_attn
  hooks=[att.norm_q.register_forward_hook(lambda mod,ins,out:values.update(q=out)),att.norm_k.register_forward_hook(lambda mod,ins,out:values.update(k=out))]
  # QKV fusion bypasses att.v; collect its output via project instead.
  project=net.project
  def project_capture(l,at,mod):
   q,k,v=project(l,at,mod);values.update(q=q,k=k,v=v);return q,k,v
  net.project=project_capture
  try:result=original_block(block,layer,x,e0,grid,freqs,context,hk,hv,start,False,step,*args,**kw)
  finally:
   net.project=project
   for hook in hooks:hook.remove()
  q=rope(values['q'],grid,freqs,start)[0].transpose(0,1);k=rope(values['k'],grid,freqs,start)[0].transpose(0,1);v=values['v'][0].transpose(0,1)
  data=dict(layer=layer,height=a.height,width=a.width,chunks=a.chunks,chunk_tokens=n,history_tokens=hk.shape[1],prompt_id=pid,seed=0,step=step,source='actual frozen Dense closed-loop generation',q=q.cpu().contiguous(),kh=hk[0].transpose(0,1).cpu().contiguous(),vh=hv[0].transpose(0,1).cpu().contiguous(),kc=k.cpu().contiguous(),vc=v.cpu().contiguous())
  torch.save(data,out/f'layer{layer:02d}.pt');captured.append(layer);status('captured',layer=layer);return result
 net._block=block
 p0=protocol();steps,_=native_steps(net.wrapper);rng=torch.Generator(device='cuda').manual_seed(0);shape=(p0['latent_shape'][0],a.height//8,a.width//8)
 noise=torch.randn((1,a.chunks*3,*shape),generator=rng,device='cuda',dtype=torch.bfloat16)
 transitions=torch.stack([torch.stack([torch.randn((3,*shape),generator=rng,device='cuda',dtype=torch.bfloat16).unflatten(0,(1,3)) for _ in range(3)]) for _ in range(a.chunks)])
 prefix=();last=None;latents=[]
 with sdpa_backend():
  for chunk in range(a.chunks):
   identity=CacheIdentity(net.model_version,net.router_version,f'{pid}.seed0',f'rollout.chunk{chunk}',0,chunk*n,p0['context_noise'],owner='teacher')
   cache=tuple(LayerKV(x.k,x.v,identity,x.layer) for x in prefix)
   if chunk:
    result=net.forward_chunk(last,embeds,p0['context_noise'],cache,(chunk-1)*3,sparse=False,step=0,identity=identity,checkpoint=False,collect_aux=False,write_cache=True);cache=result['cache'];prefix=cache
   validate(cache,identity,chunk*n);current=noise[:,chunk*3:(chunk+1)*3]
   for step,timestep in enumerate(steps):
    status('generating',chunk=chunk,step=step)
    # sparse=True here is only a capture phase tag. The block wrapper forces Dense.
    result=net.forward_chunk(current,embeds,timestep,cache,chunk*3,sparse=True,step=step,identity=identity,checkpoint=False,collect_aux=False,write_cache=False);x0=result['x0']
    assert torch.isfinite(x0).all()
    if step<3:current=net.wrapper.scheduler.add_noise(x0.flatten(0,1),transitions[chunk,step].flatten(0,1),torch.full((3,),steps[step+1],device='cuda')).unflatten(0,(1,3))
   last=x0;latents.append(x0.cpu())
  if (a.height,a.width,a.chunks)==(480,832,4):
   expected=torch.load(TRAINING/f'e2e_final/{pid}.seed0/Dense.latent.pt',weights_only=True);assert torch.equal(torch.cat(latents,1),expected),'Capture adapter changes original Dense output'
 assert sorted(captured)==[1,14,24,29]
 atomic_json(out/'completed.json',dict(completed=True,source='actual Dense generated QKV',layers=captured,shape=[a.height,a.width],chunks=a.chunks,chunk_tokens=n,history_tokens=(a.chunks-1)*n,base_latents_exact=(a.height,a.width,a.chunks)==(480,832,4),peak_gb=torch.cuda.max_memory_allocated()/1e9));status('completed')
if __name__=='__main__':main()
