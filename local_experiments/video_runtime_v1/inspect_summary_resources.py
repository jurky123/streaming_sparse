"""Compile one original/new summary call and record resource use."""
import json,torch
from .weights import ROOT,load
from ..review_v1 import split_summary as old
from . import summary_moments as new
@torch.no_grad()
def main():
 torch.set_num_threads(4)
 model,*_=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt');rows=[]
 class Inspect:
  def __init__(self,kernel,label):self.kernel=kernel;self.label=label
  def __getitem__(self,grid):
   def call(*a,**kw):
    compiled=self.kernel[grid](*a,**kw);rows.append(dict(name=self.label,registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared));return compiled
   return call
 k=torch.zeros((1,4608,12,128),device='cuda',dtype=torch.bfloat16)[0].transpose(0,1).reshape(12,72,64,128);v=torch.ones_like(k)
 for label,module in [('old',old),('new',new)]:
  for name in ['_pool_rep','_rep_stats']:setattr(module,name,Inspect(getattr(module,name),label+name))
  module.build(model[15],k,v,True)
 torch.cuda.synchronize();(ROOT/'artifacts/video_runtime_v1/post_rope_round1/resources.json').write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
