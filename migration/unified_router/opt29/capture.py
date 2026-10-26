"""Replay missing adjacent-step Q, preserving original trajectory/token ownership."""
import argparse,json,time
from pathlib import Path
import torch
from .setup import ROOT
from ..common import event_key,save,update,paired_seed
from streaming_sparse.v08.replay import ReplayDataset
from streaming_sparse.v08.backbone import FunctionalBackbone,load_teacher
from streaming_sparse.capture.capture_runner import sdpa_backend
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int);p.add_argument('--world',type=int,default=8);a=p.parse_args();m=json.loads((ROOT/'capture_manifest.json').read_text());rows=m['rows'][a.rank::a.world];status=ROOT/f'capture/rank{a.rank}.json';paired_seed(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;row=None
 datasets={s:ReplayDataset(s,require_complete=True)for s in ['train','val']}
 class Capture(FunctionalBackbone):
  def _block(self,*args,**kw):
   block,layer,x,e0,grid,freqs=args[:6];start=args[9]
   if layer in m['layers']and start==row['history_tokens']//1560:
    dest=ROOT/'full_queries'/event_key(row)/f'layer{layer:02d}.pt'
    if not dest.exists():
     from wan.modules.causal_model import causal_rope_apply
     frames=e0.shape[1];tokens=x.shape[1]//frames;e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2);mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2);att=block.self_attn
     q=att.norm_q(att.q(mod)).view(1,mod.shape[1],12,128);q=causal_rope_apply(q,grid,freqs,start_frame=start).to(x.dtype)[0].transpose(0,1)
     assert q.shape==(12,4680,128)and torch.isfinite(q).all()
     save(dest,dict(q=q.cpu().contiguous(),indices=list(range(4680)),sample_id=row['sample_id'],trajectory_id=row['trajectory_id'],chunk=row['chunk'],step=row['step'],split=row['split'],teacher_sha256=m['teacher_sha256'],payload_sha256=row['payload_sha256'],capture='fresh teacher replay of exact source step; original token positions'))
   return super()._block(*args,**kw)
 update(status,status='loading',completed=0,total=len(rows));net=Capture(load_teacher('cuda'),None,layers=m['layers']).eval();versions=tuple(p._version for p in net.model.parameters())
 with sdpa_backend():
  for i,row in enumerate(rows):
   if(ROOT/'STOP').exists():raise RuntimeError('STOP requested')
   if not all((ROOT/'full_queries'/event_key(row)/f'layer{l:02d}.pt').exists()for l in m['layers']):
    sample=datasets[row['split']].load(row['replay_index']);assert sample['sample_id']==row['sample_id'];net.replay(sample,sparse=False,checkpoint=False,collect_aux=False,recent_gradient=False)
   update(status,status='capturing',completed=i+1,total=len(rows),sample_id=row['sample_id'])
 assert versions==tuple(p._version for p in net.model.parameters());update(status,status='completed',completed=len(rows),total=len(rows))
if __name__=='__main__':main()
