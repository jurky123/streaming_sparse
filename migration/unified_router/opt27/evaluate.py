"""Paired video speed and exact-output checks against frozen Opt25."""
import argparse,gc,json,time,statistics,gzip
from pathlib import Path
import torch
from .setup import ROOT
from .pipeline import install
from ..common import paired_seed,atomic_json,update
from ..opt25.rollout import trajectory
BASE=ROOT.parent/'opt25_720p_e2e_20261025'
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--height',type=int,default=720);p.add_argument('--width',type=int,default=1280);p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from ..opt22.inference import load
 from ..opt25.network import FusedNetwork
 from ..opt25.pipeline import PipelineState
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 out=ROOT/'e2e'/f'{a.height}x{a.width}'/f'case{a.case}';out.mkdir(parents=True,exist_ok=True)
 def status(s,**kw):update(out/'status.json',status=s,**kw);print(s,kw,flush=True)
 status('loading');net,m=load();net.__class__=FusedNetwork;net.history_chunks=3
 ds=ReplayDataset('val',require_complete=True);manifest=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint'];pid=manifest['prompt_ids'][a.case]
 sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 refs=torch.load(BASE/f'{a.height}x{a.width}'/f'case{a.case}/latents.pt',weights_only=True);versions=tuple(p._version for p in net.model.parameters());results={}
 labels=['Dense','Route26','Prep27'];labels=labels[a.case%3:]+labels[:a.case%3]
 with sdpa_backend():
  for label in labels:
   net.step_graphs.clear();net.tile_graphs.clear();net.step_pool=torch.cuda.graph_pool_handle();net.tile_pool=torch.cuda.graph_pool_handle();PipelineState.workspaces.clear();gc.collect();torch.cuda.empty_cache();PipelineState.sparge=label!='Dense';install(label=='Prep27')
   if PipelineState.sparge:
    from ..opt24.sparge import module
    module()
   name='Dense' if label=='Dense' else 'Tile20_after';reference=refs['Dense' if label=='Dense' else 'Sparge32'];samples=[];torch.cuda.reset_peak_memory_stats()
   for repeat in range(a.repeats+1):
    status('warmup' if repeat==0 else 'timing',variant=label,repeat=repeat);net.select(name);torch.cuda.synchronize();start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,label!='Dense',height=a.height,width=a.width);torch.cuda.synchronize();seconds=time.perf_counter()-start;z=torch.cat(chunks,1)
    assert torch.isfinite(z).all()
    if not torch.equal(z,reference):
     error=float((z.float()-reference.float()).square().sum()/reference.float().square().sum());atomic_json(out/'mismatch.json',dict(variant=label,repeat=repeat,nmse=error));raise AssertionError((label,'latent changed',error))
    if repeat:samples.append(seconds)
   row=dict(seconds=statistics.median(samples),samples=samples,full_video_bit_exact=True,peak_gb=torch.cuda.max_memory_allocated()/1e9)
   if label!='Dense':
    rr=net.routing_records;assert len(rr)==348 and {r['layer']for r in rr}==set(range(1,30));den=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in rr)
    row['keep']=torch.stack([r['retained_H']+r['retained_C']for r in rr]).sum().item()/den;row['execution']=sum(r['executed_pairs']for r in rr)/den
   results[label]=row;atomic_json(out/'timing.json',dict(case=a.case,prompt_id=pid,height=a.height,width=a.width,frames=45,results=results))
   if label=='Prep27' and a.case==0:
    status('profiling');original=net.forward_chunk;profiles=[]
    def observed(*args,**kw):
     if args[4]==9 and kw.get('sparse',True) and kw.get('step',0)==3:
      torch.cuda.synchronize()
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
       r=original(*args,**kw);torch.cuda.synchronize()
      profiles.append(prof);return r
     return original(*args,**kw)
    net.forward_chunk=observed;net.select(name);chunks,_=trajectory(net,embeds,pid,0,True,height=a.height,width=a.width);assert torch.equal(torch.cat(chunks,1),reference);net.forward_chunk=original
    path=out/'trace.json';profiles[0].export_chrome_trace(str(path));data=json.loads(path.read_text());kernels={}
    for e in data['traceEvents']:
     if e.get('cat')=='kernel':
      r=kernels.setdefault(e['name'],dict(name=e['name'],calls=0,ms=0.));r['calls']+=1;r['ms']+=e.get('dur',0)/1000
    with gzip.open(out/'trace.json.gz','wb')as f:f.write(path.read_bytes())
    path.unlink();atomic_json(out/'profile.json',dict(scope='30 layers, final chunk step3, warmed independent pass',kernels=sorted(kernels.values(),key=lambda r:-r['ms']),kernel_sum_ms=sum(r['ms']for r in kernels.values())))
 assert versions==tuple(p._version for p in net.model.parameters())
 atomic_json(out/'completed.json',dict(completed=True,case=a.case,prompt_id=pid,height=a.height,width=a.width,frames=45,frozen_weights=True,all_full_latents_bit_exact=True,sparse_layers=list(range(1,30))));status('completed')
if __name__=='__main__':main()
