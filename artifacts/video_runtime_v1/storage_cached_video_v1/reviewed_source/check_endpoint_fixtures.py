"""E49 tiny gains and dominant removal; explicit FP32 packed fallback contract."""
import argparse,json,inspect
import numpy as np
import torch
from .weights import ROOT
from . import scores
from .endpoint_den import denominator
from .scores_packed import packed_reduce
from ..e30_beta_diagnostics.run import local_scores
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cached',action='store_true');args=ap.parse_args()
 from . import endpoint_den
 endpoint_den.CACHED_GROUPS=args.cached
 torch.set_num_threads(4);torch.manual_seed(43);scores.SCAN_EXCLUDE=True
 source=inspect.getsource(scores.grouped_scores)
 old='    den=torch.logaddexp(lb[:,:,None,None],rx[:,:,:,None])\n    den=torch.logaddexp(den,ax[:,:,None,:]);den=torch.logaddexp(den,lr[:,:,:,None]);den=torch.logaddexp(den,ea[:,:,None,:])'
 assert old in source
 ns=dict(vars(scores),denominator=denominator);exec(source.replace(old,'    den=denominator(lb,rx,lr,ax,ea)'),ns);newfn=ns['grouped_scores'];rows=[]
 for path in sorted((ROOT/'artifacts/e49_numeric_v1').glob('reference_input*.npz')):
  z=np.load(path);rp,b,lm,c,mask,r,a=[torch.as_tensor(z[n],device='cuda') for n in ('rp','beta','logmass','center','mask','r','a')];r=r.long();a=a.long();nr=len(torch.unique_consecutive(r));na=len(r)//nr;rr=r.reshape(nr,na)[:,0];aa=a.reshape(nr,na)[0]
  oldval,_=local_scores(rp,b,lm,c,mask,r,a,1.);new=newfn(rp[None],b[None],lm[None],c[None],mask[None],rr[None],aa[None])[0].flatten()
  good=torch.allclose(oldval,new,atol=1e-10,rtol=1e-8) and int(oldval.argmax())==int(new.argmax()) and bool(oldval.max()>0)==bool(new.max()>0)
  assert good,path
  rows.append(dict(file=path.name,max_abs=float((oldval-new).abs().max()),gain=float(new.max()),action=int(new.argmax()),passed=True))
 assert len(rows)>=14
 rp=torch.randn(2,16,410,16,device='cuda',dtype=torch.float64)*.1;b=torch.randn(2,16,410,device='cuda',dtype=torch.float64)*.2;lm=torch.randn_like(b);c=torch.randn_like(rp);lm[:,:,0]+=1000
 mask=(torch.arange(410,device='cuda')<41)[None].expand(2,-1);r=torch.arange(16,device='cuda')[None].expand(2,-1);a=torch.arange(41,73,device='cuda')[None].expand(2,-1)
 oldval=scores.grouped_scores(rp,b,lm,c,mask,r,a);new=newfn(rp,b,lm,c,mask,r,a);assert torch.allclose(oldval,new,atol=1e-11,rtol=1e-9)
 # Force the packed reconstruction branch with maximum-den > 80.
 state=torch.randn(2,16,3,16,device='cuda',dtype=torch.float64);cand=torch.randn(2,16,48,34,device='cuda',dtype=torch.float32);den=torch.zeros(2,16,16,32,device='cuda',dtype=torch.float64)
 lb=torch.zeros(2,16,device='cuda',dtype=torch.float64);rx=torch.zeros(2,16,16,device='cuda',dtype=torch.float64);ax=torch.zeros(2,16,32,device='cuda',dtype=torch.float64);ro=torch.randn(2,16,16,16,device='cuda',dtype=torch.float64);ao=torch.randn(2,16,32,16,device='cuda',dtype=torch.float64)
 fallback=[]
 for val in [0.,1000.]:
  maximum=torch.full_like(lb,val);x=packed_reduce(den,state,cand.double(),lb,rx,ro,ax,ao,maximum);y=packed_reduce(den,state,cand,lb,rx,ro,ax,ao,maximum);assert torch.equal(x,y);fallback.append(dict(maximum=val,bitwise=True))
 root=ROOT/'artifacts/video_runtime_v1/storage_review_v1';root.mkdir(exist_ok=True);(root/('endpoint_cached_fixtures.json' if args.cached else 'endpoint_fixtures.json')).write_text(json.dumps(dict(fixtures=rows,dominant_remove=True,packed_fp32_storage=fallback),indent=2));print('14 E49 fixtures, dominant removal, packed fallback dtype contracts passed',flush=True)
if __name__=='__main__':main()
