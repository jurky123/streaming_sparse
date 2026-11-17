"""One warmed final-chunk GPU timeline per tier; never enable legacy block capture."""
import argparse,json
import torch
from .weights import ROOT
from .deployment import current,configure,load_weights,make_runtime
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tier',choices=['fast','full'],required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 cfg=current(full_pset=a.tier=='full',route_m16=False,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 rt.reset()
 for c in range(8):generate_chunk(engine,emb,701,c)
 torch.cuda.synchronize();root=ROOT/'artifacts/video_runtime_v1/tiers_m16_v1/timeline';root.mkdir(exist_ok=True)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:generate_chunk(engine,emb,701,8);torch.cuda.synchronize()
 path=root/(a.tier+'_trace.json');prof.export_chrome_trace(str(path))
 rows=[dict(name=x.key,cuda_us=x.self_device_time_total,cpu_us=x.self_cpu_time_total,count=x.count) for x in prof.key_averages()]
 (root/(a.tier+'_profile.json')).write_text(json.dumps(rows,indent=2))
 events=json.loads(path.read_text())['traceEvents'];gpu=sorted((e['ts'],e['ts']+e['dur']) for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('ph')=='X')
 end=gpu[0][0];busy=0.;gaps=[]
 for start,stop in gpu:
  if start>end:gaps.append(start-end)
  busy+=max(0.,stop-max(start,end));end=max(end,stop)
 span=end-gpu[0][0]
 (root/(a.tier+'_summary.json')).write_text(json.dumps(dict(chunk=8,kernels=len(gpu),gpu_span_us=span,gpu_union_busy_us=busy,gap_us=span-busy,max_gap_us=max(gaps),gap_over_100us=sum(x for x in gaps if x>100),scope='CPU/CUDA profiler of warmed final chunk; instrumentation changes host dispatch; gaps are not promised recoverable Python time'),indent=2));rt.close()
if __name__=='__main__':main()
