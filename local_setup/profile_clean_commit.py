"""Capture real-model steady-state routing inputs and profile the accepted path."""
import os,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from local_experiments.broad_common import Engine
from local_experiments.review_v1.model import Model
from local_experiments.review_v1.optimized_runtime import StableLatestRuntime
from local_experiments.review_v1.fast_route import GraphRoute
from local_experiments.review_v1.ops import Config,Summary
from torch.profiler import profile,ProfilerActivity
out=Path('/tmp/latest_kernel_optimization3');out.mkdir(exist_ok=True)
class Captured(Exception):pass
with torch.no_grad():
 engine=Engine();base='artifacts/four_gaps_20261031/shared/epoch1.pt';m=Model(base);m.load_state_dict(torch.load('artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt',weights_only=False)['model'])
 old=torch.load(base,weights_only=False,map_location='cpu')['model'];centers={l:old[f'layers.{l}.center'].cuda() for l in range(1,30)}
 rt=StableLatestRuntime(engine,m,'no_downstream',centers,config=Config(history_pages=360))
 inputs={k:v.cuda() for k,v in torch.load('artifacts/comparison_20261113/horse_field/shared_inputs.pt',weights_only=True).items()}
 z=torch.load('artifacts/comparison_20261113/horse_field/ours_latest_fast/latents.pt',weights_only=True).cuda()

 from local_experiments.review_v1.ops import SummaryOps,RouteOps
 import time
 for c in range(8):engine.forward(z[:,c*3:c*3+3],inputs['prompt_embeds'],c,4)
 x=inputs['initial_noise'][:,24:27]
 engine.forward(x,inputs['prompt_embeds'],8,0)
 def tagged(fn,label):
  def wrap(*a,**kw):
   with torch.profiler.record_function(label):return fn(*a,**kw)
  return wrap
 rt.group=tagged(rt.group,'component.group')
 rt.query_route_scores=tagged(rt.query_route_scores,'component.route')
 rt.selected_attention=tagged(rt.selected_attention,'component.attention')
 rt.build_summary=tagged(rt.build_summary,'component.summary')
 RouteOps.select=tagged(RouteOps.select,'component.selection')
 torch.cuda.synchronize();start=time.perf_counter()
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
  engine.forward(z[:,24:27],inputs['prompt_embeds'],8,4)
  torch.cuda.synchronize()
 elapsed=time.perf_counter()-start
 prof.export_chrome_trace(str(out/'clean_before.trace.json'))
 rows=[dict(name=e.key,device_type=str(e.device_type),calls=e.count,self_device_us=e.self_device_time_total,device_us=e.device_time_total,cpu_us=e.cpu_time_total) for e in prof.key_averages() if e.self_device_time_total or e.key.startswith('component.')]
 rows.sort(key=lambda x:x['self_device_us'],reverse=True)
 result=dict(profiled_wall_seconds=elapsed,scope='Clean chunk8 step4 with warmed grouping; profiler overhead included, diagnostic only',components=[r for r in rows if r['name'].startswith('component.')],events=rows)
 (out/'clean_before.profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['components'],indent=2));print(json.dumps(rows[:20],indent=2));rt.close()
