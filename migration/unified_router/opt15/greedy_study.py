import argparse,json,statistics,torch
from ..common import atomic_json
from ..opt5.probe import capture,timing
from .setup import ROOT,CAPTURE
from .orders import order
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--file',required=True);a=p.parse_args();torch.set_num_threads(4);d=torch.load(CAPTURE/a.file,map_location='cuda',weights_only=False);mask=d['mask'];h,n,b=mask.shape;rows=[]
 methods=['Identity','Greedy_64','Greedy_128','GreedyCos_128','Greedy_256','GreedyCos_256','Greedy_512']
 for method in methods:
  try:
   fn=lambda:order(mask,method);ix=fn();assert torch.equal(ix.sort(-1).values,torch.arange(n,device=mask.device).expand(h,n))
   reordered=mask.gather(1,ix[:,:,None].expand_as(mask));pad=torch.nn.functional.pad(reordered,(0,0,0,(-n)%32));counts=pad.reshape(h,-1,32,b).any(2).sum(-1)
   graph,_=capture(fn);timing(graph,30);ts=[timing(graph,30)for _ in range(3)]
   row=dict(method=method,union_mean=float(counts.float().mean()),union_fraction=float(counts.float().mean()/b),union_max=int(counts.max()),order_ms=statistics.median(ts),samples=ts)
  except Exception as e:row=dict(method=method,error=str(e))
  rows.append(row);print(row,flush=True);atomic_json(ROOT/f'greedy_study_{a.file.removesuffix(".pt")}.json',rows)
if __name__=='__main__':main()
