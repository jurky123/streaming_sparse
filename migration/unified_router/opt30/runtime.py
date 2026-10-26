import torch,json,gc
from .pipeline import Pipeline
from .setup import ROOT,PARENT,VARIANTS,source_step
from ..opt25.pipeline import Workspace
from ..opt27.pipeline import Pipeline as Baseline
from ..opt28.pipeline import Pipeline as Fixed64
from ..opt28.retile_v2 import select
class PipelineState:
 workspaces={};kind=None
 def __init__(self,model,raw):
  key=id(model)
  if key not in self.workspaces:self.workspaces[key]=Workspace(raw)
  ws=self.workspaces[key];self.identity=None
  if self.kind=='FixedQ32':self.pipe=Baseline(model,raw,True,ws)
  else:
   self.pipe=Pipeline(model,raw,workspace=ws)
   if self.kind=='FixedQ64':
    import types
    self.pipe.select=select;self.pipe.run=types.MethodType(Fixed64.run,self.pipe)
 def refresh_history(self,kh,vh,identity):
  if self.kind=='FixedQ32':
   cache=self.pipe.cache;old=cache.history;cache.refresh_history(kh,vh);fresh=cache.history
   for a,b in zip(old.parts,fresh.parts):
    for x,y in zip(a,b):
     if x is not None:x.copy_(y)
   cache.history=old;self.pipe.quant.history();self.pipe.sync(0,self.pipe.quant.nhpages)
  else:self.pipe.refresh_history(kh,vh)
  self.identity=identity
