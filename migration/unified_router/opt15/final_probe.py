import argparse,json,statistics,torch
from ..common import atomic_json
from ..opt9.segmented_attention import attention as reference
from ..opt5.probe import capture,timing
from .setup import ROOT,CAPTURE
from .orders import order
from .attention_packed import attention
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--file',default='layer24_chunk3.pt');p.add_argument('--methods',default='Identity,BitsetFast_64_16_1,BitsetFast_128_32_2');a=p.parse_args();torch.set_num_threads(4)
 d=torch.load(CAPTURE/a.file,map_location='cuda',weights_only=False);q,k,v,m=[d[n]for n in ['q','k','v','mask']];nh=d['sketch'].history_tokens;args=[q,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],m];h,n,b=m.shape;rows=[]
 old=lambda:reference(*args,pack_v=True,fuse_pack=True);expected=old();rg,_=capture(old);timing(rg,100)
 for method in a.methods.split(','):
  try:
   ix=order(m,method);assert torch.equal(ix.sort(-1).values,torch.arange(n,device=m.device).expand(h,n))
   def full():return attention(*args,pack_v=True,fuse_pack=True,method=method)
   def precomputed():return attention(*args,pack_v=True,fuse_pack=True,permutation=ix)
   y=full();assert torch.isfinite(y).all();nmse=float((y.float()-expected.float()).square().sum()/expected.float().square().sum());assert nmse<1e-5,(method,nmse)
   ng,_=capture(full);cg,_=capture(precomputed);timing(ng,100);timing(cg,100);pairs=[]
   for i in range(4):
    if i%2:ct=timing(cg,40);nt=timing(ng,40);rt=timing(rg,40)
    else:rt=timing(rg,40);nt=timing(ng,40);ct=timing(cg,40)
    pairs.append([rt,nt,ct])
   rearranged=m.gather(1,ix[:,:,None].expand_as(m));pad=torch.nn.functional.pad(rearranged,(0,0,0,(-n)%32));uf=float(pad.reshape(h,-1,32,b).any(2).float().mean())
   row=dict(method=method,reference_ms=statistics.median(x[0]for x in pairs),full_ms=statistics.median(x[1]for x in pairs),precomputed_order_ms=statistics.median(x[2]for x in pairs),nmse_vs_original=nmse,max_abs=float((y.float()-expected.float()).abs().max()),union_fraction=uf,samples=pairs)
  except Exception as e:
   import traceback;traceback.print_exc();row=dict(method=method,error=str(e))
  rows.append(row);print(row,flush=True);atomic_json(ROOT/f'final_probe_{a.file.removesuffix(".pt")}.json',rows)
if __name__=='__main__':main()
