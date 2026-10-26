"""Compare executed tile work, useful mask work and Dense SDPA timing."""
import json,statistics,gc
import torch
from ..opt9.segmented_attention import attention
from ..opt5.probe import capture,timing
from ..common import atomic_json
from .setup import ROOT,PARENT
@torch.no_grad()
def main():
 torch.set_num_threads(4);rows=[]
 for f in sorted((PARENT/'captured').glob('*.pt')):
  d=torch.load(f,map_location='cuda',weights_only=False);q,k,v,m=[d[n]for n in ['q','k','v','mask']];nh=d['sketch'].history_tokens;h,n,dim=q.shape;b=m.shape[-1];nk=k.shape[1]
  sparse=lambda:attention(q,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],m,pack_v=True,fuse_pack=True)
  dense=lambda:torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])
  sg,_=capture(sparse);dg,_=capture(dense);timing(sg,100);timing(dg,100);pairs=[]
  for i in range(4):
   if i%2:dt=timing(dg,30);st=timing(sg,30)
   else:st=timing(sg,30);dt=timing(dg,30)
   pairs.append([st,dt])
  tiled=torch.nn.functional.pad(m,(0,0,0,(-n)%32)).reshape(h,-1,32,b);used=tiled.any(2);counts=used.sum(-1)
  useful=float((m*d['sketch'].costs).sum());executed=float(used.sum())*32*64;dense_pairs=h*n*nk
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:sparse();torch.cuda.synchronize()
  trace=ROOT/f'attention_trace_{f.stem}.json';prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents']
  kernels=[dict(name=e['name'],milliseconds=e['dur']/1000,registers=e['args'].get('registers per thread'),occupancy=e['args'].get('est. achieved occupancy %'))for e in events if e.get('cat')=='kernel']
  row=dict(file=f.name,weighted_keep=useful/dense_pairs,executed_pair_ratio=executed/dense_pairs,useful_pair_fraction=useful/executed,executed_gflops=executed*dim*4/1e9,dense_gflops=dense_pairs*dim*4/1e9,sparse_ms=statistics.median(x[0]for x in pairs),dense_ms=statistics.median(x[1]for x in pairs),union_pages_mean=float(counts.float().mean()),union_pages_max=int(counts.max()),union_pages_min=int(counts.min()),kernels=kernels,samples=pairs,scope='Sparse includes INT8 K quantization, V packing, union, attention and merge; Dense is SDPA on already concatenated KV, excludes concatenation. Executed FLOPs count full 32x64 tiles for both QK and PV; not a hardware throughput measurement.')
  rows.append(row);print(row,flush=True);atomic_json(ROOT/'attention_diagnosis.json',rows);del sg,dg;gc.collect()
if __name__=='__main__':main()
