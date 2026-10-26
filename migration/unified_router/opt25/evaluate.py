"""Paired full-video 720p benchmark. One isolated worker/GPU per prompt."""
import argparse,gc,json,time,statistics
from pathlib import Path
import torch
from ..common import paired_seed,atomic_json,update
from ..opt24.setup import ROOT as PARENT,TRAINING
from .rollout import trajectory
ROOT=PARENT.parent/'opt25_720p_e2e_20261025'

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--height',type=int,default=720);p.add_argument('--width',type=int,default=1280);p.add_argument('--repeats',type=int,default=5);p.add_argument('--skip-decode',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from ..opt22.inference import load
 from .network import FusedNetwork
 from .pipeline import PipelineState
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 out=ROOT/f'{a.height}x{a.width}'/f'case{a.case}';out.mkdir(parents=True,exist_ok=True)
 def status(s,**kw):update(out/'status.json',status=s,**kw);print(s,kw,flush=True)
 status('loading');net,m=load();net.__class__=FusedNetwork;net.history_chunks=3
 ds=ReplayDataset('val',require_complete=True);manifest=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint'];pid=manifest['prompt_ids'][a.case]
 sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 versions=tuple(p._version for p in net.model.parameters());latents={};results={};base=a.height==480 and a.width==832
 # Rotate variant order across prompts to reduce order/thermal bias.
 labels=['Dense','Opt22','Sparge32'];labels=labels[a.case%3:]+labels[:a.case%3]
 with sdpa_backend():
  for label in labels:
   net.step_graphs.clear();net.tile_graphs.clear();net.step_pool=torch.cuda.graph_pool_handle();net.tile_pool=torch.cuda.graph_pool_handle();gc.collect();torch.cuda.empty_cache();PipelineState.workspaces.clear();PipelineState.sparge=label=='Sparge32'
   if PipelineState.sparge:
    from ..opt24.sparge import module
    module()
   name='Dense' if label=='Dense' else 'Tile20_after';samples=[];first=None
   torch.cuda.reset_peak_memory_stats()
   for repeat in range(a.repeats+1):
    status('warmup' if repeat==0 else 'timing',variant=label,repeat=repeat)
    net.select(name);torch.cuda.synchronize();start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,label!='Dense',height=a.height,width=a.width);torch.cuda.synchronize();seconds=time.perf_counter()-start;z=torch.cat(chunks,1)
    assert torch.isfinite(z).all()
    if first is None:first=z.clone()
    else:assert torch.equal(z,first),(label,'replay nondeterministic')
    if repeat:samples.append(seconds)
    else:warmup=seconds
   if base:
    refpath=TRAINING/f'e2e_final/{pid}.seed0/{name}.latent.pt' if label!='Sparge32' else PARENT/f'e2e/case{a.case}/Sparge32.latent.pt'
    assert torch.equal(z,torch.load(refpath,weights_only=True)),(label,'480p regression')
   latents[label]=z.cpu();row=dict(seconds=statistics.median(samples),samples=samples,warmup_seconds=warmup,repeat_exact=True,peak_gb=torch.cuda.max_memory_allocated()/1e9)
   if label!='Dense':
    rr=net.routing_records;assert len(rr)==348 and {r['layer']for r in rr}==set(range(1,30));den=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in rr)
    row['keep']=torch.stack([r['retained_H']+r['retained_C']for r in rr]).sum().item()/den;row['execution']=sum(r['executed_pairs']for r in rr)/den
   results[label]=row;atomic_json(out/'timing.json',dict(case=a.case,prompt_id=pid,height=a.height,width=a.width,frames=45,results=results))
 for label in ['Opt22','Sparge32']:assert torch.equal(latents[label][:,:3],latents['Dense'][:,:3])
 assert versions==tuple(p._version for p in net.model.parameters());net.step_graphs.clear();net.tile_graphs.clear();del net;gc.collect();torch.cuda.empty_cache()
 torch.save(latents,out/'latents.pt')
 if not a.skip_decode:
  status('decoding')
  from utils.wan_wrapper import WanVAEWrapper
  vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);decode={}
  for label in labels:
   zz=latents[label].cuda().to(torch.bfloat16);times=[]
   for rep in range(3):
    vae.model.clear_cache();torch.cuda.synchronize();t=time.perf_counter();pix=vae.decode_to_pixel(zz,use_cache=False);torch.cuda.synchronize();elapsed=time.perf_counter()-t
    assert pix.shape==(1,45,3,a.height,a.width) and torch.isfinite(pix).all()
    if rep:times.append(elapsed)
    del pix
   decode[label]=dict(seconds=statistics.median(times),samples=times)
   atomic_json(out/'decode.json',decode)
  del vae;gc.collect();torch.cuda.empty_cache()
 atomic_json(out/'completed.json',dict(completed=True,case=a.case,prompt_id=pid,height=a.height,width=a.width,frames=45,frozen_weights=True,first_chunk_exact=True,regression_480p=base,sparse_layers=list(range(1,30))));status('completed')
if __name__=='__main__':main()
