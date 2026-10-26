import argparse,json,gc,torch,time
from .setup import ROOT
from .inference import load
from ..opt25.rollout import trajectory
from ..common import atomic_json,paired_seed
from streaming_sparse.v08.replay import ReplayDataset
from streaming_sparse.capture.capture_runner import sdpa_backend
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,default=480);p.add_argument('--width',type=int,default=832);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;paired_seed(0)
 net,m=load();net.allow_untrained=True;ds=ReplayDataset('val',require_complete=True);manifest=json.loads((__import__('pathlib').Path(m['parent_experiment'])/'e2e_manifest.json').read_text());pid=manifest['prompt_ids'][0];sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 names=['Dense','FixedQ32','PrevQ32','PrevMean8','q64_qk_local8','q32_qk_local16'];rows=[];dense=None
 with sdpa_backend():
  for name in names:
   output=None
   for repeat in range(2):
    net.select(name);chunks,_=trajectory(net,embeds,pid,0,name!='Dense',height=a.height,width=a.width);z=torch.cat(chunks,1)
    if repeat==0:output=z.clone()
    else:assert torch.equal(z,output),(name,'cross-rollout state leak')
    if dense is not None:assert torch.equal(z[:,:3],dense[:,:3]),(name,'dense first chunk')
    if name!='Dense':assert len(net.routing_records)==348 and{r['layer']for r in net.routing_records}==set(range(1,30))
   if name=='Dense':dense=output
   if name in ['Dense','FixedQ32']:
    path=ROOT.parent/'opt25_720p_e2e_20261025'/f'{a.height}x{a.width}/case0/latents.pt'
    ref=torch.load(path,map_location='cpu',weights_only=True)['Dense'if name=='Dense'else'Sparge32'];assert torch.equal(output,ref),(name,'original reference changed')
   if name in ['PrevQ32','PrevMean8']:
    from .setup import PREVIOUS
    previous=PREVIOUS/f'e2e/{a.height}x{a.width}/case0'/('q32_full.latent.pt'if name=='PrevQ32'else'q64_mean8.latent.pt')
    assert torch.equal(output,torch.load(previous,map_location='cpu',weights_only=True)),'Opt29 baseline changed'
   rows.append(dict(variant=name,repeat_exact=True,first_chunk_dense_exact=True,latent_finite=True));print(rows[-1],flush=True);atomic_json(ROOT/f'graph_smoke_{a.height}.json',dict(passed=False,rows=rows))
 atomic_json(ROOT/f'graph_smoke_{a.height}.json',dict(passed=True,rows=rows,scope='Mixed/untrained candidates: correctness only, never a trained quality result'))
if __name__=='__main__':main()
