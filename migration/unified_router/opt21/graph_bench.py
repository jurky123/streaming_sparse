import argparse,json,time,statistics
from pathlib import Path
import torch
from .setup import ROOT,PARENT
from ..common import paired_seed,atomic_json,update

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--kernel',action='store_true');p.add_argument('--rope',action='store_true');p.add_argument('--cases',default='0,1');a=p.parse_args()
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from ..opt20.network import load
 from .step_graph import configure_graph
 from .rollout import trajectory
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 out=ROOT/('graph_rope' if a.rope else 'graph_kernel' if a.kernel else 'graph_exact');out.mkdir(exist_ok=True);update(out/'status.json',status='loading')
 if a.rope:
  from .rope import install
  install()
 net,m=load();configure_graph(net,a.kernel);ds=ReplayDataset('val',require_complete=True);pids=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text())['prompt_ids']
 results=[]
 with sdpa_backend():
  for case in map(int,a.cases.split(',')):
   pid=pids[case];sample=ds.load(next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
   for name in ['Dense','Tile20_after']:
    expected=torch.load(PARENT/f'e2e_final/{pid}.seed0/{name}.latent.pt',weights_only=True);ts=[];errors=[]
    for repeat in range(4):
     update(out/'status.json',status='warmup' if repeat==0 else 'timing',case=case,variant=name,repeat=repeat)
     net.select(name);torch.cuda.synchronize();t=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,name!='Dense');torch.cuda.synchronize();seconds=time.perf_counter()-t
     y=torch.cat(chunks,1);err=float((y.float()-expected.float()).square().sum()/expected.float().square().sum());print(case,name,repeat,seconds,err,torch.equal(y,expected),flush=True)
     if not a.kernel:assert torch.equal(y,expected),(case,name,err)
     if repeat==0:reference=y.clone()
     else:assert torch.equal(y,reference);ts.append(seconds)
     errors.append(err)
    results.append(dict(case=case,prompt_id=pid,variant=name,seconds=statistics.median(ts),samples=ts,nmse=max(errors),bit_identical=torch.equal(y,expected),peak_gb=torch.cuda.max_memory_allocated()/1e9))
    atomic_json(out/'result.json',results)
 update(out/'status.json',status='completed')
if __name__=='__main__':main()
