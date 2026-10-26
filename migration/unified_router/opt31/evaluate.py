import argparse,json,time,gc,math,statistics,tempfile
from pathlib import Path
import numpy as np
import torch
from .setup import ROOT,PARENT
from .inference import load
from .runtime import PipelineState
from .rollout import trajectory
from ..common import atomic_json,save,sha

def summarize(rows):
 mse=sum(x['mse_rgb01']for x in rows)/len(rows)
 return dict(frames=len(rows),mse_rgb01=mse,psnr_db=-10*math.log10(mse)if mse else None,lpips=sum(x['lpips']for x in rows)/len(rows))

@torch.no_grad()
def decode_compare(latents,folder,height,width,frames,status):
 import lpips,imageio.v2 as imageio
 from PIL import Image,ImageDraw
 from utils.wan_wrapper import WanVAEWrapper
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
 decoded={}
 with tempfile.TemporaryDirectory(prefix='opt31_rgb_')as tmp:
  reference=np.memmap(Path(tmp)/'dense.f32',dtype=np.float32,mode='w+',shape=(frames,3,height,width))
  for name,latent in latents.items():
   vae.model.clear_cache();position=0;seconds=0.;rows=[];writer=None
   if name!='Dense':writer=imageio.get_writer(folder/f'Dense_{name}.mp4',fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18'])
   try:
    for start in range(0,latent.shape[1],3):
     status('decoding',variant=name,latent_chunk=start//3)
     torch.cuda.synchronize();t0=time.perf_counter();rgb=vae.decode_to_pixel(latent[:,start:start+3].cuda().to(torch.bfloat16),use_cache=True)[0].cpu();torch.cuda.synchronize();seconds+=time.perf_counter()-t0
     rgb=rgb[:max(0,frames-position)]
     for frame in rgb:
      if name=='Dense':reference[position]=frame.numpy()
      else:
       dense=torch.from_numpy(np.array(reference[position],copy=True));mse=float(((dense.double()-frame.double())*.5).square().mean());lp=float(metric(dense[None].cuda(),frame[None].cuda()))
       assert math.isfinite(mse)and math.isfinite(lp)
       if position<9:assert mse==0,'Dense firstchunk RGB changed'
       rows.append(dict(frame=position,second=position/16,mse_rgb01=mse,psnr_db=-10*math.log10(mse)if mse else None,lpips=lp))
       canvas=Image.fromarray(np.concatenate([((x.permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()for x in [dense,frame]],axis=1));draw=ImageDraw.Draw(canvas)
       for i,label in enumerate(['Dense',name]):draw.text((i*width+10,10),label,fill='white',stroke_width=2,stroke_fill='black')
       writer.append_data(np.asarray(canvas))
      position+=1
     if position>=frames:break
   finally:
    if writer:writer.close()
   assert position==frames
   decoded[name]=dict(vae_seconds=seconds)
   if rows:
    decoded[name]['quality']=dict(full=summarize(rows),sparse_frames_only=summarize(rows[9:]),final5s=summarize(rows[-80:]),per_second=[dict(second=i//16,**summarize(rows[i:i+16]))for i in range(0,frames,16)],per_frame=rows)
   atomic_json(folder/'quality_partial.json',decoded)
  del reference
 vae.model.clear_cache();return decoded

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--case',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert json.loads((ROOT/f'preflight_{a.height}.json').read_text())['passed']
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for method in manifest['methods'].values():
  for c in method['checkpoints'].values():assert sha(c['path'])==c['sha256']
 frames=a.seconds*16;chunks=math.ceil((frames+3)/12);folder=ROOT/f'e2e/{a.seconds}s/{a.height}x{a.width}/case{a.case}';folder.mkdir(parents=True,exist_ok=True)
 def status(phase,**kw):atomic_json(folder/'status.json',dict(status=phase,**kw));print(phase,kw,flush=True)
 status('loading');net,m=load()
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 ds=ReplayDataset('val',require_complete=True);base=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());assert ds.fingerprint()==base['val_fingerprint'];pid=base['prompt_ids'][a.case];sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 methods=list(manifest['methods']);methods=methods[a.case%len(methods):]+methods[:a.case%len(methods)];latents={};results={}
 with sdpa_backend():
  for name in ['Dense']+methods+['Dense']:
   reference=None;samples=[];chunk_samples=[]
   if a.seconds>5:
    # Independent saved static-position seven-chunk run proves dynamic absolute
    # RoPE/steady-state graph reuse preserved output across window eviction.
    shortpath=ROOT/f'e2e/5s/{a.height}x{a.width}/case{a.case}'/(name+'.latent.pt')
    if shortpath.exists():
     net.select(name);status('verifying_bounded_graph',variant=name)
     short,_=trajectory(net,embeds,pid,0,name!='Dense',a.height,a.width,7)
     assert torch.equal(short,torch.load(shortpath,map_location='cpu',weights_only=True)),(name,'bounded graph changed seven-chunk output')
     del short
   for repeat in range(4):
    net.select(name);status('warmup'if repeat==0 else'timing',variant=name,repeat=repeat)
    torch.cuda.synchronize();t0=time.perf_counter();z,times=trajectory(net,embeds,pid,0,name!='Dense',a.height,a.width,chunks);torch.cuda.synchronize();seconds=time.perf_counter()-t0
    if reference is None:reference=z.clone()
    else:assert torch.equal(reference,z),(name,'repeat mismatch')
    if name!='Dense':
     assert torch.equal(z[:,:3],latents['Dense'][:,:3]);rr=net.routing_records;assert len(rr)==(chunks-1)*4*29
     assert max(r['history_tokens']for r in rr)==3*(3*(a.height//16)*(a.width//16))
    if repeat:samples.append(seconds);chunk_samples.append(times)
   if name in latents:assert torch.equal(z,latents[name])
   latents[name]=z;save(folder/(name+'.latent.pt'),z)
   if name=='Dense'and name in results:samples=results[name]['samples']+samples;chunk_samples=results[name]['chunk_samples']+chunk_samples
   row=dict(seconds=statistics.median(samples),samples=samples,chunk_samples=chunk_samples,first_chunk_dense_exact=True,repeats_exact=True,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
   if name!='Dense':
    den=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in rr);row.update(keep=torch.stack([r['retained_H']+r['retained_C']for r in rr]).sum().item()/den,execution=sum(r['executed_pairs']for r in rr)/den)
   results[name]=row;atomic_json(folder/'timing.json',results)
 net.step_graphs.clear();net.tile_graphs.clear();PipelineState.workspaces.clear();del net,embeds;gc.collect();torch.cuda.empty_cache()
 decoded=decode_compare(latents,folder,a.height,a.width,frames,status)
 for name,row in results.items():row.update(decoded[name]);row['generation_plus_vae_seconds']=row['seconds']+row['vae_seconds']
 for name in methods:
  results[name]['speedup_vs_dense']=results['Dense']['seconds']/results[name]['seconds'];results[name]['speedup_with_vae']=results['Dense']['generation_plus_vae_seconds']/results[name]['generation_plus_vae_seconds']
 atomic_json(folder/'result.json',dict(completed=True,case=a.case,prompt_id=pid,height=a.height,width=a.width,duration_seconds=a.seconds,frames=frames,generated_frames=chunks*12-3,num_chunks=chunks,history_chunks=3,variants=results,scope=manifest['timing'],frame_policy=manifest['frame_policy']));status('completed')
if __name__=='__main__':main()
