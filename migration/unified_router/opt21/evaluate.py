"""Eight fresh paired cases: original and equally optimized Dense/Tile20."""
import argparse,collections,gzip,json,os,time,statistics,gc
from pathlib import Path
import torch
from .setup import ROOT,PARENT
from ..common import paired_seed,atomic_json,update,sha
from ..speed_profile import gpu_info

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from ..opt20.network import load
 from streaming_sparse.v08.rollout import trajectory as original
 from .rollout import trajectory as optimized
 from .step_graph import configure_graph
 from .rope import install
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 out=ROOT/'final'/f'case{a.case}';out.mkdir(parents=True,exist_ok=True)
 def status(stage,**kw):update(out/'status.json',status=stage,pid=os.getpid(),case=a.case,**kw);print(stage,kw,flush=True)
 status('loading');net,m=load();ds=ReplayDataset('val',require_complete=True);manifest=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint']
 pid=manifest['prompt_ids'][a.case];sample=ds.load(next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 refs={name:torch.load(PARENT/f'e2e_final/{pid}.seed0/{name}.latent.pt',weights_only=True) for name in ['Dense','Tile20_after']}
 versions=tuple(p._version for p in net.model.parameters());results={};initialization={}
 with sdpa_backend():
  for phase,fn in [('original',original),('optimized',optimized)]:
   if phase=='optimized':
    configure_graph(net,True);install();gc.collect();torch.cuda.empty_cache()
   for name in ['Dense','Tile20_after']:
    samples=[];records=None
    for repeat in range(4):
     status('warmup' if repeat==0 else 'timing',phase=phase,variant=name,repeat=repeat)
     net.select(name);torch.cuda.synchronize();started=time.perf_counter();chunks,_=fn(net,embeds,pid,0,name!='Dense');torch.cuda.synchronize();seconds=time.perf_counter()-started
     actual=torch.cat(chunks,1);assert torch.equal(actual,refs[name]),(a.case,phase,name,repeat,float((actual.float()-refs[name].float()).square().mean()))
     if repeat==0:initialization[phase+'_'+name]=seconds
     else:samples.append(seconds)
     if name!='Dense':
      records=net.routing_records;assert len(records)==348 and {r['layer'] for r in records}==set(range(1,30))
    row=dict(seconds=statistics.median(samples),samples=samples,latent_bit_identical_to_opt20=True,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
    if records:
     den=sum(r['queries']*(r['history_tokens']+r['current_tokens']) for r in records)
     row.update(keep=torch.stack([r['retained_H']+r['retained_C'] for r in records]).sum().item()/den,executed=sum(r['executed_pairs'] for r in records)/den)
    results[phase+'_'+name]=row;atomic_json(out/'result.json',dict(case=a.case,prompt_id=pid,results=results,warmup_seconds=initialization,gpu=gpu_info()))
  assert versions==tuple(p._version for p in net.model.parameters())
  # A separate pass profiles only a warmed last denoising step, excluding compilation.
  if a.case==0:
   status('profiling');original_forward=net.forward_chunk;profiles=[];phase_events=[]
   def observed(*args,**kw):
    start=args[4];sparse=kw.get('sparse',True);step=kw.get('step',0)
    category='history' if float(args[2])==0. else 'denoise'
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
    if sparse and start==9 and step==3:
     torch.cuda.synchronize()
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
      value=original_forward(*args,**kw);torch.cuda.synchronize()
     profiles.append(prof)
    else:value=original_forward(*args,**kw)
    end.record();phase_events.append((category,begin,end));return value
   net.forward_chunk=observed;net.select('Tile20_after');chunks,_=optimized(net,embeds,pid,0,True);torch.cuda.synchronize();assert torch.equal(torch.cat(chunks,1),refs['Tile20_after']);net.forward_chunk=original_forward
   assert len(profiles)==1;trace=out/'trace.json';profiles[0].export_chrome_trace(str(trace));data=json.loads(trace.read_text());kernels={}
   for event in data['traceEvents']:
    if event.get('cat')=='kernel':
     row=kernels.setdefault(event['name'],dict(name=event['name'],calls=0,ms=0.));row['calls']+=1;row['ms']+=event.get('dur',0)/1000
   with gzip.open(out/'trace.json.gz','wb') as stream:stream.write(trace.read_bytes())
   trace.unlink();atomic_json(out/'profile.json',dict(scope='all30 layers, chunk4 denoise step3, separate profiled pass',kernels=sorted(kernels.values(),key=lambda r:-r['ms']),kernel_sum_ms=sum(r['ms'] for r in kernels.values()),latent_bit_identical=True))
  # Decoding is measured separately and is identical because full latents match.
  status('vae_timing');net.step_graphs.clear();net.tile_graphs.clear();gc.collect();torch.cuda.empty_cache()
  from utils.wan_wrapper import WanVAEWrapper
  vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);decode={}
  for name in ['Dense','Tile20_after']:
   times=[]
   for repeat in range(3):
    torch.cuda.synchronize();started=time.perf_counter();pixels=vae.decode_to_pixel(refs[name].cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();torch.cuda.synchronize();seconds=time.perf_counter()-started
    assert pixels.shape==(45,3,480,832)
    if repeat:times.append(seconds)
   decode[name]=dict(seconds=statistics.median(times),samples=times)
  atomic_json(out/'decode.json',decode)
 atomic_json(out/'completed.json',dict(status='completed',case=a.case,prompt_id=pid,all_latents_bit_identical=True));status('completed')
if __name__=='__main__':main()
