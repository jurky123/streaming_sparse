"""Supplement group-aligned queries; reuse validated original history/current KV."""
import argparse,os,time
from pathlib import Path
import torch
from ..common import config,event_key,save,update,paired_seed
from streaming_sparse.v08.replay import ReplayDataset
from streaming_sparse.v08.backbone import FunctionalBackbone,load_teacher
from streaming_sparse.capture.capture_runner import sdpa_backend
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));paired_seed(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=m['rows'][rank::world];status=root/f'capture/rank{rank}.json';update(status,status='loading',completed=0,total=len(rows));datasets={s:ReplayDataset(s,require_complete=True)for s in ['train','val']};row=None;seen=0
 class Capture(FunctionalBackbone):
  def _block(self,*args,**kw):
   block,layer,x,e0,grid,freqs=args[:6];hk,hv,start=args[7:10]
   if layer in m['layers']and start==row['history_tokens']//1560:
    dest=root/'group_queries'/event_key(row)/f'layer{layer:02d}.pt'
    if not dest.exists():
     from wan.modules.causal_model import causal_rope_apply
     frames=e0.shape[1];tokens=x.shape[1]//frames;e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2);mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2);att=block.self_attn
     q=att.norm_q(att.q(mod)).view(1,mod.shape[1],12,128);q=causal_rope_apply(q,grid,freqs,start_frame=start).to(x.dtype)[0].transpose(0,1)
     oldpath=Path(m['cache'])/'events'/event_key(row)/f'layer{layer:02d}.pt';old=torch.load(oldpath,map_location='cpu',weights_only=True,mmap=True)
     ids=row['group_query_indices'];common=[i for i in ids if i in row['query_indices']];assert common
     original=[row['query_indices'].index(i)for i in common];assert torch.equal(q[:,common].cpu(),old['q'][:,original]),(row['sample_id'],layer,'Q reuse mismatch')
     save(dest,dict(q=q[:,ids].cpu().contiguous(),indices=ids,sample_id=row['sample_id'],query_reuse_bit_identical=True,teacher_sha256=m['teacher_sha256']))
     if seen==0 and layer==24:
      save(root/f'capture/full_query_rank{rank}.pt',dict(q=q.cpu().contiguous(),sample_id=row['sample_id'],row=row,layer=layer))
   return super()._block(*args,**kw)
 net=Capture(load_teacher('cuda'),None,layers=m['layers']).eval();versions=tuple(p._version for p in net.model.parameters());start_time=time.monotonic()
 with sdpa_backend():
  for i,row in enumerate(rows):
   if (root/'STOP').exists():raise RuntimeError('STOP requested')
   if not all((root/'group_queries'/event_key(row)/f'layer{layer:02d}.pt').exists()for layer in m['layers']):
    sample=datasets[row['split']].load(row['replay_index']);assert sample['sample_id']==row['sample_id'];net.replay(sample,sparse=False,checkpoint=False,collect_aux=False,recent_gradient=False)
   seen+=1;update(status,status='capturing',completed=i+1,total=len(rows),sample_id=row['sample_id'],elapsed_seconds=time.monotonic()-start_time)
 assert versions==tuple(p._version for p in net.model.parameters());update(status,status='completed',completed=len(rows),total=len(rows),backbone_unchanged=True)
if __name__=='__main__':main()
