import torch,json
from pathlib import Path
from ..opt17.data import Events as Base
from ..common import event_key
from .setup import ROOT,source_step
class Events(Base):
 def __init__(self,m,layer):
  super().__init__(m,layer);self.m=m;self.layer=layer;self.lookup={(r['trajectory_id'],r['chunk'],r['step']):r for r in m['sequence_rows']};self.previous_cache={}
 def previous(self,item,refresh):
  row=item['row'];s=source_step(row['step'],refresh)
  if s==row['step']:return None
  key=(row['trajectory_id'],row['chunk'],s);prior=self.lookup[key];assert prior['split']==row['split']and s<row['step']
  if key not in self.previous_cache:
   p=ROOT/'full_queries'/event_key(prior)/f'layer{self.layer:02d}.pt';v=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
   assert v['sample_id']==prior['sample_id']and v['teacher_sha256']==self.m['teacher_sha256']and v['indices']==list(range(4680));self.previous_cache[key]=v['q']
  return self.previous_cache[key].cuda()
