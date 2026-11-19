"""Capture real-model steady-state routing inputs and profile the accepted path."""
import os,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from local_experiments.broad_common import Engine
from local_experiments.review_v1.model import Model
from local_experiments.review_v1.fast_runtime import FastLatestRuntime
from local_experiments.review_v1.fast_route import GraphRoute
from local_experiments.review_v1.ops import Config,Summary
from torch.profiler import profile,ProfilerActivity
out=Path('/tmp/latest_kernel_optimization');out.mkdir(exist_ok=True)
class Captured(Exception):pass
with torch.no_grad():
 engine=Engine();base='artifacts/four_gaps_20261031/shared/epoch1.pt';m=Model(base);m.load_state_dict(torch.load('artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt',weights_only=False)['model'])
 old=torch.load(base,weights_only=False,map_location='cpu')['model'];centers={l:old[f'layers.{l}.center'].cuda() for l in range(1,30)}
 rt=FastLatestRuntime(engine,m,'no_downstream',centers,config=Config(history_pages=360))
 inputs={k:v.cuda() for k,v in torch.load('artifacts/comparison_20261113/horse_field/shared_inputs.pt',weights_only=True).items()}
 z=torch.load('artifacts/comparison_20261113/horse_field/ours_latest_fast/latents.pt',weights_only=True).cuda()
 def capture(q,s,model):
  global sample
  sample={k:v.detach().contiguous() for k,v in dict(q=q,keys=s.keys,mean=s.mean,detail=s.detail,stats=s.stats,theta=model.theta).items()}
  torch.save({k:v.cpu() for k,v in sample.items()},out/'route_inputs.pt');raise Captured()
 rt.route_scores=capture
 try:
  for c in range(8):engine.forward(z[:,c*3:c*3+3],inputs['prompt_embeds'],c,4)
  engine.forward(inputs['initial_noise'][:,24:27],inputs['prompt_embeds'],8,0)
 except Captured:pass
 finally:rt.close()
 route=GraphRoute();s=Summary(sample['keys'],sample['mean'],sample['detail'],sample['stats']);q=sample['q'];theta=sample['theta']
 for _ in range(3):route(q,s,theta)
 torch.cuda.synchronize()
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False) as prof:
  for _ in range(5):
   with torch.profiler.record_function('latest_route_8_tiles'):route(q,s,theta)
  torch.cuda.synchronize()
 prof.export_chrome_trace(str(out/'route_before.trace.json'))
 rows=[]
 for e in prof.key_averages():
  us=getattr(e,'self_device_time_total',getattr(e,'self_cuda_time_total',0))
  if us:rows.append(dict(name=e.key,calls=e.count,self_device_us=us))
 rows.sort(key=lambda x:x['self_device_us'],reverse=True)
 (out/'route_before.profile.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[:20],indent=2))
