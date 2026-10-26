import json,statistics,torch,triton as tr
from ..opt9.select import _packed_select
from ..opt10.compact_route import CompactMomentRoute
from ..opt5.probe import capture,timing
from ..common import atomic_json
from .select import select
from .batched_select import select as batched
from .setup import ROOT,PARENT
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 for f in sorted((PARENT/'captured').glob('*.pt')):
  d=torch.load(f,map_location='cuda',weights_only=False);route=CompactMomentRoute(batch=4680);state=route.begin(d['prepared'],d['sketch']);score,_=route.score(d['q'],state);o=d['sketch'].origins;beta=d['beta'];h,n,b=score.shape
  def old(w):
   mask=torch.empty_like(score,dtype=torch.bool);rem=torch.empty((h*n,8),device=score.device,dtype=torch.int64);add=torch.empty_like(rem)
   _packed_select[(h*n,)](score,mask,o,rem,add,n,b,tr.next_power_of_2(b),beta,'packed32',num_warps=w,enable_fp_fusion=False)
   return mask,rem,add
  expected=old(4);ref,_=capture(lambda:old(4));timing(ref,100)
  configs=[('serial',1,1),('serial',2,1),('top3',1,1),('top3',2,1),('batched',4,4),('batched',4,8),('batched',4,2),('batched',8,4)]
  for method,w,g in configs:
   try:
    fn=lambda:old(w)if method=='serial'else(select(score,o,beta,w)if method=='top3'else batched(score,o,beta,g,w))
    actual=fn();ng,_=capture(fn);timing(ng,100);pairs=[]
    for i in range(4):
     if i%2:nt=timing(ng,80);rt=timing(ref,80)
     else:rt=timing(ref,80);nt=timing(ng,80)
     pairs.append([rt,nt])
    row=dict(file=f.name,pages=b,method=method,warps=w,group=g,reference_ms=statistics.median(x[0]for x in pairs),new_ms=statistics.median(x[1]for x in pairs),changed_rows=[int((a!=e).any(-1).sum())for a,e in zip(actual,expected)],samples=pairs)
   except Exception as e:row=dict(file=f.name,method=method,warps=w,group=g,error=str(e))
   rows.append(row);print(row,flush=True);atomic_json(ROOT/'select_control.json',rows)
if __name__=='__main__':main()
