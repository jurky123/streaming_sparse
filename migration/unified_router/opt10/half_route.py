import torch
from ..opt5.prepare import prepare_weights
from .head_route import HeadRoute,HeadMixin
from .diag_route import DiagRoute

def prepare_half(prepared,content):
 state=prepare_weights(prepared,content);state['value']=state['value'].half();return state
class HalfValueRoute(HeadRoute):
 def __init__(self,**kwargs):
  super().__init__(**kwargs);self.prepare_weights=torch.compile(prepare_half,fullgraph=True,dynamic=False)
class HalfDiagRoute(HeadMixin,DiagRoute):
 def __init__(self,**kwargs):
  super().__init__(**kwargs);self.prepare_weights=torch.compile(prepare_half,fullgraph=True,dynamic=False)
