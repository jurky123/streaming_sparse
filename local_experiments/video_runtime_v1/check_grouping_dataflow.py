"""Real captured input, bitwise contracts and interleaved isolated latency."""
import argparse,json,statistics,time
import torch
from .weights import ROOT
from ..review_v1.model import Layer
from ..review_v1.ops import Config
from ..review_v1.cached_ops import GraphGrouping as Old
from .grouping_dataflow import GraphGrouping as New
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--softmax',action='store_true');parser.add_argument('--spatial',action='store_true');args=parser.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=ROOT/'artifacts/video_runtime_v1/grouping_dataflow_v1';saved=torch.load(out/'input.pt',weights_only=True)
 m=Layer({n:x.cuda() for n,x in saved['model'].items()}).eval().requires_grad_(False);k=saved['k'].cuda();v=saved['v'].cuda()
 old=(New if args.spatial else Old)(Config(history_pages=360));new=New(Config(history_pages=360),contiguous_softmax=args.softmax,spatial_softmax=args.spatial);contracts=[]
 # Replays with changed inputs and weights verify graph cache refresh as well as output equality.
 for i in range(4):
  kk=k if i==0 else k.roll(i*64,1);vv=v if i==0 else v.roll(i*64,1)
  if i==3:m.group_gate.add_(.001)
  ref=old(m,kk,vv);got=new(m,kk,vv)
  eq=[torch.equal(a,b) for a,b in zip(ref,got)];contracts.append(eq)
  if not args.softmax:assert all(eq),eq
 def bench(fn):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();t=time.perf_counter();a.record()
  for _ in range(20):fn(m,k,v)
  b.record();b.synchronize();return dict(gpu_ms=a.elapsed_time(b)/20,wall_ms=(time.perf_counter()-t)*1000/20)
 rows={'old':[],'new':[]}
 for i in range(10):
  for name in (('old','new') if i%2==0 else ('new','old')):rows[name].append(bench(old if name=='old' else new))
 med={n:{key:statistics.median(r[key] for r in rs) for key in ('gpu_ms','wall_ms')} for n,rs in rows.items()}
 if args.softmax or args.spatial:out=out/('spatial' if args.spatial else 'softmax');out.mkdir(exist_ok=True)
 for name,fn in [('old',old),('new',new)]:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn(m,k,v);torch.cuda.synchronize()
  prof.export_chrome_trace(str(out/(name+'.trace.json')))
  (out/(name+'_profile.json')).write_text(json.dumps([dict(name=x.key,calls=x.count,cuda_us=x.self_device_time_total) for x in prof.key_averages()],indent=2))
 result=dict(contracts=contracts,medians=med,speedup=med['old']['gpu_ms']/med['new']['gpu_ms'],rows=rows,removed_bytes=dict(dead_feature_gather_read_write=2*2*12*4608*32*4,output_clone_read_write=2*(2*12*4608*128*2+12*4608*8)))
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
