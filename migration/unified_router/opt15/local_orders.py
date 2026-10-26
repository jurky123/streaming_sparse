"""Window-local grouping retains existing spatial locality; exact masks unchanged."""
import torch,triton as tr
import triton.language as tl
@tr.jit
def _greedy(DOT,COUNTS,OUT,N:tl.constexpr,W:tl.constexpr,NW:tl.constexpr,METRIC:tl.constexpr):
 win=tl.program_id(0);h=tl.program_id(1);p=tl.arange(0,W);valid=p+win*W<N
 count=tl.load(COUNTS+(h*NW+win)*W+p);remaining=valid
 for group in range(W//32):
  seed=tl.min(tl.where(remaining,p,2147483647),0);seed=tl.minimum(seed,W-1)
  shared=tl.load(DOT+((h*NW+win)*W+seed)*W+p).to(tl.float32)
  cseed=tl.sum(tl.where(p==seed,count,0.),0)
  if METRIC==0:distance=1.-shared/tl.maximum(cseed+count-shared,1.)
  else:distance=1.-shared*tl.rsqrt(tl.maximum(cseed*count,1.))
  distance=tl.maximum(distance,0.)
  key=(distance.to(tl.uint32,bitcast=True)&(0xffffffff^(W-1)))|p.to(tl.uint32)
  key=tl.where(remaining,key,0x7fffffff)
  ordered=tl.sort(key,descending=False)
  threshold=tl.sum(tl.where(p==31,ordered,0),0)
  remaining=remaining&(key>threshold)
  idx=(ordered&(W-1)).to(tl.int32);pos=win*W+group*32+p
  tl.store(OUT+h*N+pos,win*W+idx,(p<32)&(pos<N))

def local_order(mask,method):
 h,n,b=mask.shape
 if method.startswith('Greedy'):
  w=int(method.split('_')[1]);metric=1 if 'Cos'in method else 0;pad=torch.nn.functional.pad(mask,(0,0,0,(-n)%w)).half();nw=pad.shape[1]//w;x=pad.reshape(h*nw,w,b)
  dot=torch.bmm(x,x.transpose(1,2));counts=x.sum(-1,dtype=torch.float32);out=torch.empty((h,n),device=mask.device,dtype=torch.int64)
  _greedy[(nw,h)](dot,counts,out,n,w,nw,metric,num_warps=4)
  return out
 k,w=map(int,method.removeprefix('LocalAnchor').split('_'));pad=torch.nn.functional.pad(mask,(0,0,0,(-n)%w)).half();nw=pad.shape[1]//w;x=pad.reshape(h*nw,w,b);centers=x[:,torch.arange(k,device=mask.device)*w//k].contiguous()
 score=2*torch.bmm(x,centers.transpose(1,2)).float()-centers.float().square().sum(-1)[:,None]
 labels=score.argmax(-1).reshape(h,nw,w);valid=(torch.arange(nw*w,device=mask.device)<n).reshape(nw,w)
 labels=torch.where(valid[None],labels,k+1);ix=labels.argsort(dim=-1,stable=True)+torch.arange(nw,device=mask.device)[None,:,None]*w
 return ix.reshape(h,-1)[:,:n].contiguous()
