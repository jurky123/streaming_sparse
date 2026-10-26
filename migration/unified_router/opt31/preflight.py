import argparse,json,gc
from pathlib import Path
import torch
from .setup import ROOT,PREVIOUS
from .inference import load
from .runtime import PipelineState
from .rollout import trajectory
from ..common import atomic_json

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 net,m=load()
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.capture.capture_runner import sdpa_backend
 ds=ReplayDataset('val',require_complete=True);base=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());pid=base['prompt_ids'][0];s=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=s['prompt_embeds'].cuda().to(torch.bfloat16);rows=[]
 with sdpa_backend():
  for name in ['Dense','FastReference']:
   net.select(name);z,_=trajectory(net,embeds,pid,0,name!='Dense',a.height,a.width,4)
   path=PREVIOUS/f'e2e/{a.height}x{a.width}/case0'/('Dense.latent.pt'if name=='Dense'else'q64_mean8.latent.pt')
   if path.exists():
    assert torch.equal(z,torch.load(path,map_location='cpu',weights_only=True)),(name,'four-chunk compatibility')
    rows.append(dict(name=name,four_chunk_reference_exact=True))
   else:
    net.select(name);again,_=trajectory(net,embeds,pid,0,name!='Dense',a.height,a.width,4)
    assert torch.equal(z,again),(name,'new-resolution repeat')
    rows.append(dict(name=name,four_chunk_repeat_exact=True,saved_reference_available=False))
   print(rows[-1],flush=True)
  reference=None
  for repeat in range(2):
   net.select('Best');z,_=trajectory(net,embeds,pid,0,True,a.height,a.width,7)
   if reference is None:reference=z.clone()
   else:assert torch.equal(z,reference),'Rolling eviction repeat mismatch'
   assert len(net.routing_records)==6*4*29
   assert max(r['history_tokens']for r in net.routing_records)==9*(a.height//16)*(a.width//16)
  rows.append(dict(name='Best',seven_chunk_rollover_exact=True));print(rows[-1],flush=True)
  if a.height==512 and a.width==768:
   net.step_graphs.clear();net.step_pool=torch.cuda.graph_pool_handle();net.static_position_graphs=True
   net.select('Best');static,_=trajectory(net,embeds,pid,0,True,a.height,a.width,7)
   assert torch.equal(z,static),'Bounded versus per-position graph mismatch'
   rows.append(dict(name='Best',bounded_vs_per_position_graph_exact=True));print(rows[-1],flush=True)
 net.step_graphs.clear();net.tile_graphs.clear();PipelineState.workspaces.clear();del net,embeds;gc.collect();torch.cuda.empty_cache()
 from utils.wan_wrapper import WanVAEWrapper
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);short=z[:,:6].cuda();full=vae.decode_to_pixel(short,use_cache=False).cpu();vae.model.clear_cache();stream=torch.cat([vae.decode_to_pixel(short[:,i:i+3],use_cache=True).cpu()for i in [0,3]],1);vae.model.clear_cache();error=float((full-stream).abs().max());assert error<1e-5,error
 atomic_json(ROOT/f'preflight_{a.height}.json',dict(passed=True,height=a.height,width=a.width,rows=rows,streaming_vae_max_abs=error,history_window=3,source='Saved Opt29 references when available, otherwise deterministic repeats; seven-chunk rollover; 512x768 additionally compares bounded and per-position graphs; streaming versus original VAE'))
 print('passed',a.height,error,flush=True)
if __name__=='__main__':main()
