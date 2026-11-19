"""Capture real-model steady-state routing inputs and profile the accepted path."""
import os,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from local_experiments.broad_common import Engine
from local_experiments.review_v1.model import Model
from local_experiments.review_v1.fused_runtime import FusedLatestRuntime
from local_experiments.review_v1.fast_route import GraphRoute
from local_experiments.review_v1.ops import Config,Summary
from torch.profiler import profile,ProfilerActivity
out=Path('/tmp/latest_kernel_optimization2');out.mkdir(exist_ok=True)
class Captured(Exception):pass
with torch.no_grad():
 engine=Engine();base='artifacts/four_gaps_20261031/shared/epoch1.pt';m=Model(base);m.load_state_dict(torch.load('artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt',weights_only=False)['model'])
 old=torch.load(base,weights_only=False,map_location='cpu')['model'];centers={l:old[f'layers.{l}.center'].cuda() for l in range(1,30)}
 rt=FusedLatestRuntime(engine,m,'no_downstream',centers,config=Config(history_pages=360))
 inputs={k:v.cuda() for k,v in torch.load('artifacts/comparison_20261113/horse_field/shared_inputs.pt',weights_only=True).items()}
 z=torch.load('artifacts/comparison_20261113/horse_field/ours_latest_fast/latents.pt',weights_only=True).cuda()

 from local_experiments.review_v1.ops import SummaryOps
 original_build=SummaryOps.build;captured={}
 def build(layer,k,v,cheap=False):
  if engine.current['layer']==1:
   captured['cheap' if cheap else 'full']=dict(k=k.detach().cpu(),v=v.detach().cpu(),state={n:t.detach().cpu() for n,t in layer.state_dict().items()})
  return original_build(layer,k,v,cheap)
 SummaryOps.build=build
 def capture(q,k,v,s,basis,selection):
  torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),summary={n:getattr(s,n).cpu() for n in ['keys','mean','detail','stats']},basis=basis.detach().cpu(),selection={n:t.cpu() if isinstance(t,torch.Tensor) else t for n,t in selection.items()}),out/'attention_inputs.pt')
  torch.save(captured,out/'summary_inputs.pt');raise Captured()
 rt.selected_attention=capture
 try:
  for c in range(8):engine.forward(z[:,c*3:c*3+3],inputs['prompt_embeds'],c,4)
  engine.forward(inputs['initial_noise'][:,24:27],inputs['prompt_embeds'],8,0)
 except Captured:print('Captured real steady-state attention and cheap/full summaries.',flush=True)
 finally:rt.close()
