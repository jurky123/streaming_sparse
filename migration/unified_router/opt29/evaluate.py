"""Closed-loop video acceptance: frozen Dense/Q32 controls and all-layer finalists."""
import argparse,json,gc,time,statistics,gzip,traceback
from pathlib import Path
import torch
from .setup import ROOT
from .inference import load
from .runtime import PipelineState
from ..opt25.rollout import trajectory
from ..common import atomic_json,update,paired_seed,save

def preview(path,pixels,names):
 import imageio.v2 as imageio
 import numpy as np
 from PIL import Image,ImageDraw
 width=pixels[names[0]].shape[-1]
 with imageio.get_writer(path,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18'])as writer:
  for frame in range(45):
   arrays=[((pixels[n][frame].permute(1,2,0)+1)*127.5).clamp(0,255).to(torch.uint8).numpy()for n in names];canvas=Image.fromarray(np.concatenate(arrays,axis=1));draw=ImageDraw.Draw(canvas)
   for i,name in enumerate(names):draw.text((i*width+10,10),name,fill='white',stroke_width=2,stroke_fill='black')
   writer.append_data(np.asarray(canvas))
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--case',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;paired_seed(0)
 for name in ['preflight.json','sequence_check.json',f'graph_smoke_{a.height}.json']:assert json.loads((ROOT/name).read_text())['passed']
 state=json.loads((ROOT/'queue.json').read_text());finalists=state.get('viable',[])
 if a.smoke:finalists=[]
 folder=ROOT/('evaluation_smoke'if a.smoke else'e2e')/f'{a.height}x{a.width}'/f'case{a.case}';folder.mkdir(parents=True,exist_ok=True)
 def status(s,**kw):update(folder/'status.json',status=s,**kw);print(s,kw,flush=True)
 status('loading');net,m=load();versions=tuple(x._version for x in net.model.parameters())
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 ds=ReplayDataset('val',require_complete=True);manifest=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint'];pid=manifest['prompt_ids'][a.case]
 sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 names=['FixedQ32','FixedQ64']+finalists;names=names[a.case%len(names):]+names[:a.case%len(names)];names=['Dense']+names+['Dense'];latents={};results={};failures={}
 with sdpa_backend():
  for name in names:
   try:
    samples=[];reference=None;checks={}
    for repeat in range(4):
     net.select(name);torch.cuda.synchronize();status('warmup'if repeat==0 else'timing',variant=name,repeat=repeat);start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,name!='Dense',height=a.height,width=a.width);torch.cuda.synchronize();seconds=time.perf_counter()-start;z=torch.cat(chunks,1)
     assert torch.isfinite(z).all()
     if reference is None:reference=z.clone()
     else:assert torch.equal(z,reference),(name,'repeat mismatch')
     if name!='Dense':assert torch.equal(z[:,:3],latents['Dense'][:,:3]),'Dense firstchunk changed'
     if repeat:samples.append(seconds)
    if name in latents:assert torch.equal(z,latents[name])
    latents[name]=z;save(folder/(name+'.latent.pt'),z)
    oldpath=ROOT.parent/'opt25_720p_e2e_20261025'/f'{a.height}x{a.width}'/f'case{a.case}/latents.pt'
    if name in ['Dense','FixedQ32']and oldpath.exists():assert torch.equal(z,torch.load(oldpath,map_location='cpu',weights_only=True)['Dense'if name=='Dense'else'Sparge32']),'Frozen baseline reference changed'
    if name=='Dense'and name in results:samples=results[name]['samples']+samples
    row=dict(seconds=statistics.median(samples),samples=samples,first_chunk_dense_exact=True,all_repeats_exact=True,checkpoints=dict(net.tile_checksums),scope='Complete45-frame generation including clean-history encoding and router; excludesVAE,compile,capture,load,video encoding')
    if name!='Dense':
     rr=net.routing_records;assert len(rr)==348 and{r['layer']for r in rr}==set(range(1,30));den=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in rr)
     row.update(keep=torch.stack([r['retained_H']+r['retained_C']for r in rr]).sum().item()/den,execution=sum(r['executed_pairs']for r in rr)/den,sparse_layers=list(range(1,30)))
    results[name]=row;atomic_json(folder/'timing.json',dict(prompt_id=pid,variants=results,failures=failures))
    if a.case==0 and not a.smoke and name!='Dense':
     original=net.forward_chunk;profiles=[]
     def observed(*args,**kw):
      if args[4]==9 and kw.get('sparse',True)and kw.get('step',0)==3:
       torch.cuda.synchronize()
       with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
        result=original(*args,**kw);torch.cuda.synchronize()
       profiles.append(prof);return result
      return original(*args,**kw)
     try:
      net.forward_chunk=observed;net.select(name);z2=torch.cat(trajectory(net,embeds,pid,0,True,height=a.height,width=a.width)[0],1);assert torch.equal(z,z2)
     finally:net.forward_chunk=original
     path=folder/(name+'.trace.json');profiles[0].export_chrome_trace(str(path));data=json.loads(path.read_text());kernels={}
     for event in data['traceEvents']:
      if event.get('cat')=='kernel':
       rec=kernels.setdefault(event['name'],dict(name=event['name'],calls=0,ms=0.));rec['calls']+=1;rec['ms']+=event.get('dur',0)/1000
     with gzip.open(folder/(name+'.trace.json.gz'),'wb')as f:f.write(path.read_bytes())
     path.unlink();atomic_json(folder/(name+'.profile.json'),dict(scope='All30 layers, finalchunk step3, warmed CUDA kernel sums; not complete-video wall time',kernels=sorted(kernels.values(),key=lambda x:-x['ms'])));del profiles,data
   except Exception as error:
    failures[name]=dict(error=repr(error),traceback=traceback.format_exc());atomic_json(folder/'failures.json',failures);traceback.print_exc()
    if name in ['Dense','FixedQ32']or'illegal memory access'in str(error):raise
    net.step_graphs.clear();net.tile_graphs.clear();PipelineState.workspaces.clear();gc.collect();torch.cuda.empty_cache()
 assert versions==tuple(x._version for x in net.model.parameters());net.step_graphs.clear();net.tile_graphs.clear();PipelineState.workspaces.clear();del net,embeds;gc.collect();torch.cuda.empty_cache()
 status('decoding');from utils.wan_wrapper import WanVAEWrapper
 import lpips
 from migration.frozen_quality.worker import pixel_metrics
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False);pixels={}
 for name,z in latents.items():
  status('decoding',variant=name);torch.cuda.synchronize();start=time.perf_counter();pixels[name]=vae.decode_to_pixel(z.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();torch.cuda.synchronize();results[name]['vae_seconds']=time.perf_counter()-start;vae.model.clear_cache()
  assert tuple(pixels[name].shape)==(45,3,a.height,a.width)
  if name!='Dense':
   assert torch.equal(pixels[name][:9],pixels['Dense'][:9]);results[name]['quality']=pixel_metrics(pixels['Dense'],pixels[name],metric)
  atomic_json(folder/'quality_partial.json',results)
 for name in latents:
  if name=='Dense':continue
  status('video',variant=name);preview(folder/f'Dense_FixedQ32_{name}.mp4',pixels,['Dense','FixedQ32',name]);results[name]['speedup_vs_dense']=results['Dense']['seconds']/results[name]['seconds'];results[name]['speedup_vs_fixed_q32']=results['FixedQ32']['seconds']/results[name]['seconds']
 atomic_json(folder/'result.json',dict(completed=not failures,prompt_id=pid,case=a.case,height=a.height,width=a.width,frames=45,variants=results,failures=failures,scope='All29 sparse; layer0/firstchunk Dense; quality on uncompressed RGB; post-firstchunk metrics are sparse_frames_only'))
 atomic_json(folder/'completed.json',dict(completed=not failures,variants=list(results),failed_variants=list(failures)));status('completed'if not failures else'partial_failure')
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
