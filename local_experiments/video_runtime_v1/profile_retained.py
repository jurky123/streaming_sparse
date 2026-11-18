"""Attribute the retained tiers without changing runtime numerical operations."""
import argparse,contextlib,json
import torch
from .accepted_tiers import config
from .deployment import CURRENT,configure,load_weights,make_runtime
from .weights import ROOT
from ..broad_common import Engine
from ..rollout_suite import generate_chunk

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--risk-online',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--native-rope',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--exact-fold',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--qk-tile64',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--rms-mean',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--rms-dataflow',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--block-pointwise',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--grouping-spatial',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--scale-in-qk', action=argparse.BooleanOptionalAction, default=CURRENT['scale_in_qk'])
 p.add_argument('--native-route-qk', action=argparse.BooleanOptionalAction, default=CURRENT['native_route_qk']);p.add_argument('--rms-row',action=argparse.BooleanOptionalAction,default=CURRENT['rms_row']);p.add_argument('--tier',choices=['base','intersection','step'],required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=config(a.tier,scale_in_qk=a.scale_in_qk and a.native_route_qk,native_route_qk=a.native_route_qk,rms_row=a.rms_row,risk_online=a.risk_online,native_rope=a.native_rope,exact_fold=a.exact_fold,qk_tile64=a.qk_tile64,rms_mean=a.rms_mean,rms_dataflow=a.rms_dataflow,block_pointwise=a.block_pointwise,grouping_spatial=a.grouping_spatial,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
 for c in range(9):generate_chunk(engine,emb,701,c)
 rt.reset()
 for c in range(8):generate_chunk(engine,emb,701,c)
 def tag(fn,label):
  def wrapped(*args,**kw):
   with torch.profiler.record_function('component.'+label):return fn(*args,**kw)
  return wrapped
 rt.group_graph=tag(rt.group_graph,'group');rt.summary_graph=tag(rt.summary_graph,'summary');rt.encode=tag(rt.encode,'encode');engine.dense=tag(engine.dense,'dense')
 for store in rt.stores.values():store.commit=tag(store.commit,'commit_store')
 from . import legacy_components as legacy
 class TaggedKernel:
  def __init__(self,kernel,label):self.kernel=kernel;self.label=label
  def __getitem__(self,grid):return tag(self.kernel[grid],self.label)
 for attr,label in [('_attn_fwd','exact'),('_budget_mask','mask'),('_tail','tail'),('_merge','merge')]:setattr(legacy,attr,TaggedKernel(getattr(legacy,attr),label))
 @contextlib.contextmanager
 def stage(name):
  with torch.profiler.record_function('stage.'+name):yield
 rt.timings.stage=stage
 if hasattr(rt,'rms_dataflow'):rt.rms_dataflow.profile=True
 if hasattr(rt,'pointwise_blocks'):rt.pointwise_blocks.profile=True
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  generate_chunk(engine,emb,701,8);torch.cuda.synchronize()
 out=ROOT/a.output if a.output else ROOT/'artifacts/video_runtime_v1/retained_tiers_profile_v1'/a.tier;out.mkdir(parents=True,exist_ok=True);prof.export_chrome_trace(str(out/'detail.trace.json'))
 rows=[dict(name=e.key,calls=e.count,self_device_us=e.self_device_time_total,device_us=e.device_time_total,cpu_us=e.cpu_time_total) for e in prof.key_averages()]
 rows.sort(key=lambda x:x['self_device_us'],reverse=True)
 (out/'detail.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps([r for r in rows if r['name'].startswith(('component.','stage.'))],indent=2));rt.close()
if __name__=='__main__':main()
