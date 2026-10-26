import json,time,statistics,argparse
from pathlib import Path
import torch
from .setup import ROOT,PARENT
from ..common import paired_seed,atomic_json,update

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--kernel',action='store_true');p.add_argument('--case',type=int,default=0);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from ..opt20.network import load
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory as original
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from .network import configure
 from .rollout import trajectory as optimized
 out=ROOT/('runtime_kernel' if a.kernel else 'runtime_exact')/f'case{a.case}';out.mkdir(parents=True,exist_ok=True)
 update(out/'status.json',status='loading');net,m=load();ds=ReplayDataset('val',require_complete=True)
 pid=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text())['prompt_ids'][a.case]
 sample=ds.load(next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 refs={};rows={};phases={};versions=tuple(p._version for p in net.model.parameters())
 def run(name,fn,**kw):
  net.select(name);return torch.cat(fn(net,embeds,pid,0,name!='Dense',**kw)[0],1)
 with sdpa_backend():
  for name in ['Dense','Tile20_after']:
   refs[name]=run(name,original);saved=torch.load(PARENT/f'e2e_final/{pid}.seed0/{name}.latent.pt',weights_only=True);assert torch.equal(saved,refs[name])
  for name in ['Dense','Tile20_after']:
   ts=[]
   for i in range(a.repeats):
    torch.cuda.synchronize();t=time.perf_counter();y=run(name,original);torch.cuda.synchronize();ts.append(time.perf_counter()-t);assert torch.equal(y,refs[name])
   rows['baseline_'+name]=dict(seconds=statistics.median(ts),samples=ts)
  configure(net,a.kernel)
  for label,opts in [('conditioning_only',dict(incremental=False,write_cache=True)),('no_denoise_kv',dict(incremental=False,write_cache=False)),('incremental_history',dict(incremental=True,write_cache=False))]:
   for name in ['Dense','Tile20_after']:
    update(out/'status.json',status='benchmark',optimization=label,variant=name)
    y=run(name,optimized,**opts);err=float((y.float()-refs[name].float()).square().sum()/refs[name].float().square().sum())
    if not a.kernel:assert torch.equal(y,refs[name]),(label,name,err)
    ts=[]
    for i in range(a.repeats):
     torch.cuda.synchronize();t=time.perf_counter();z=run(name,optimized,**opts);torch.cuda.synchronize();ts.append(time.perf_counter()-t);assert torch.equal(y,z)
    rows[label+'_'+name]=dict(seconds=statistics.median(ts),samples=ts,latent_nmse_vs_parent=err,bit_identical=torch.equal(y,refs[name]))
    atomic_json(out/'result.json',dict(prompt_id=pid,rows=rows));print(label,name,rows[label+'_'+name],flush=True)
  assert versions==tuple(p._version for p in net.model.parameters())
 update(out/'status.json',status='completed')
if __name__=='__main__':main()
