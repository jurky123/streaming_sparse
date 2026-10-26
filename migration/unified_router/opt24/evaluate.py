"""Paired full45-frame evaluation of frozen Opt22 versus same-route Sparge32."""
import argparse,gc,gzip,json,os,time,statistics
from pathlib import Path
import torch
from .setup import ROOT,TRAINING
from ..common import paired_seed,atomic_json,update
from ..opt23.bench import nmse
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert json.loads((ROOT/'preflight.json').read_text())['passed']
 from ..opt22.inference import load
 from ..opt21.rollout import trajectory
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from .pipeline import install
 out=ROOT/'e2e'/f'case{a.case}';out.mkdir(parents=True,exist_ok=True)
 def status(s,**kw):update(out/'status.json',status=s,**kw);print(s,kw,flush=True)
 status('loading');net,m=load();ds=ReplayDataset('val',require_complete=True);manifest=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint'];pid=manifest['prompt_ids'][a.case]
 sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 refs={n:torch.load(TRAINING/f'e2e_final/{pid}.seed0/{n}.latent.pt',weights_only=True)for n in ['Dense','Tile20_after']};latents={};results={};versions=tuple(p._version for p in net.model.parameters())
 with sdpa_backend():
  for label in ['Dense','Opt22','Sparge32']:
   if label=='Sparge32':
    net.step_graphs.clear();net.tile_graphs.clear();net.step_pool=torch.cuda.graph_pool_handle();net.tile_pool=torch.cuda.graph_pool_handle();gc.collect();torch.cuda.empty_cache();install()
   name='Dense' if label=='Dense' else 'Tile20_after';samples=[];first=None
   for repeat in range(4):
    status('warmup' if repeat==0 else 'timing',variant=label,repeat=repeat);net.select(name);torch.cuda.synchronize();start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,label!='Dense');torch.cuda.synchronize();seconds=time.perf_counter()-start;z=torch.cat(chunks,1)
    assert torch.isfinite(z).all() and torch.equal(z[:,:3],refs['Dense'][:,:3])
    if label!='Sparge32':assert torch.equal(z,refs[name]),(label,'baseline drift')
    if first is None:first=z.clone()
    else:assert torch.equal(z,first),(label,'replay nondeterministic')
    if repeat:samples.append(seconds)
   latents[label]=z.cpu();row=dict(seconds=statistics.median(samples),samples=samples,repeat_exact=True,first_chunk_exact=True,latent_nmse_vs_old=nmse(z,refs['Tile20_after']))
   if label!='Dense':
    rr=net.routing_records;assert len(rr)==348 and {r['layer']for r in rr}==set(range(1,30));den=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in rr)
    row['keep']=torch.stack([r['retained_H']+r['retained_C']for r in rr]).sum().item()/den;row['execution']=sum(r['executed_pairs']for r in rr)/den
   results[label]=row;atomic_json(out/'timing.json',dict(case=a.case,prompt_id=pid,results=results))
  torch.save(latents['Sparge32'],out/'Sparge32.latent.pt')
  if a.case==0:
   status('profiling');original=net.forward_chunk;profiles=[]
   def observed(*args,**kw):
    if args[4]==9 and kw.get('sparse',True) and kw.get('step',0)==3:
     torch.cuda.synchronize()
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
      r=original(*args,**kw);torch.cuda.synchronize()
     profiles.append(prof);return r
    return original(*args,**kw)
   net.forward_chunk=observed;net.select('Tile20_after');chunks,_=trajectory(net,embeds,pid,0,True);assert torch.equal(torch.cat(chunks,1),latents['Sparge32']);net.forward_chunk=original
   assert len(profiles)==1;path=out/'trace.json';profiles[0].export_chrome_trace(str(path));data=json.loads(path.read_text());kernels={}
   for e in data['traceEvents']:
    if e.get('cat')=='kernel':
     r=kernels.setdefault(e['name'],dict(name=e['name'],calls=0,ms=0.));r['calls']+=1;r['ms']+=e.get('dur',0)/1000
   with gzip.open(out/'trace.json.gz','wb')as f:f.write(path.read_bytes())
   path.unlink();atomic_json(out/'profile.json',dict(scope='All30 layers, final chunk step3, warmed separate pass',kernels=sorted(kernels.values(),key=lambda r:-r['ms']),kernel_sum_ms=sum(r['ms']for r in kernels.values())))
 assert versions==tuple(p._version for p in net.model.parameters())
 net.step_graphs.clear();net.tile_graphs.clear();del net;gc.collect();torch.cuda.empty_cache();status('decoding')
 from utils.wan_wrapper import WanVAEWrapper
 from migration.frozen_quality.worker import pixel_metrics,preview
 import lpips
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False);pixels={}
 for label,z in latents.items():pixels[label]=vae.decode_to_pixel(z.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache()
 status('quality');quality={label:pixel_metrics(pixels['Dense'],pixels[label],metric)for label in ['Opt22','Sparge32']};quality['Sparge32_vs_Opt22']=pixel_metrics(pixels['Opt22'],pixels['Sparge32'],metric)
 for label in ['Opt22','Sparge32']:assert torch.equal(pixels[label][:9],pixels['Dense'][:9])
 atomic_json(out/'quality.json',quality)
 if a.case in [0,1]:preview(out/'Dense_Opt22_Sparge32.mp4',pixels,['Dense','Opt22','Sparge32'])
 atomic_json(out/'completed.json',dict(completed=True,case=a.case,prompt_id=pid,frozen_weights=True,sparse_layers=list(range(1,30))));status('completed')
if __name__=='__main__':main()
