import argparse,json,time,statistics,gc
from pathlib import Path
import torch
from .setup import ROOT,TRAINING
from ..common import paired_seed,atomic_json,update
from .network import configure

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',default='native_stats');p.add_argument('--qkv',action='store_true');p.add_argument('--case',type=int,default=0);p.add_argument('--dense',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from ..opt21.inference import load,trajectory
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 tag=a.mode+('_qkv' if a.qkv else '')+('_dense' if a.dense else '')
 out=ROOT/'bench'/tag/f'case{a.case}';out.mkdir(parents=True,exist_ok=True);update(out/'status.json',status='loading')
 net,m=load();ds=ReplayDataset('val',require_complete=True);pid=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text())['prompt_ids'][a.case]
 sample=ds.load(next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 name='Dense' if a.dense else 'Tile20_after';reference=torch.load(TRAINING/f'e2e_final/{pid}.seed0/{name}.latent.pt',weights_only=True);rows={}
 with sdpa_backend():
  for variant in ['Opt21','Opt22']:
   if variant=='Opt22':configure(net,a.mode,a.qkv)
   samples=[]
   for rep in range(4):
    update(out/'status.json',status='warmup' if rep==0 else 'timing',variant=variant,repeat=rep);net.select(name);torch.cuda.synchronize();start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,name!='Dense');torch.cuda.synchronize();seconds=time.perf_counter()-start
    y=torch.cat(chunks,1);error=float((y.float()-reference.float()).square().sum()/reference.float().square().sum());same=torch.equal(y,reference)
    if variant=='Opt21':assert same
    if rep==0:own=y.clone()
    else:assert torch.equal(y,own);samples.append(seconds)
    print(tag,variant,rep,seconds,error,same,flush=True)
   rows[variant]=dict(seconds=statistics.median(samples),samples=samples,nmse_vs_opt21=error,bit_identical=same)
   atomic_json(out/'result.json',dict(mode=a.mode,qkv=a.qkv,prompt_id=pid,rows=rows));torch.save(y,out/(variant+'.latent.pt'))
 update(out/'status.json',status='completed')
if __name__=='__main__':main()
