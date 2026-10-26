import torch
from .probe import load,measure
from .setup import ROOT
from .pipeline import Pipeline,install
from ..opt21.pipeline import Pipeline as Old
from ..opt21.step_graph import PipelineState
from ..opt5.probe import capture,timing
from ..opt23.bench import nmse
from ..common import atomic_json
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,_,ref,aux=load(480,832,4,24);old=Old(model,raw,32,.2,'shared','fp16');new=Pipeline(model,raw,32,.2,'shared','fp16')
 expected,dd=old.run(raw['q'],raw['kc'],raw['vc'],True);actual,nd=new.run(raw['q'],raw['kc'],raw['vc'],True)
 assert torch.equal(dd['selection'],nd['selection']) and nmse(actual,expected)<1e-4
 bg,bo=capture(lambda:old.run(raw['q'],raw['kc'],raw['vc']));timing(bg,300)
 measured,_=measure(lambda:new.run(raw['q'],raw['kc'],raw['vc']),bg,expected,reps=60)
 install();state=PipelineState(model,raw);fn=lambda:state.pipe.run(raw['q'],raw['kc'],raw['vc']);g,y=capture(fn);g.replay();assert torch.equal(y,actual)
 fresh=dict(raw,kh=raw['kh'].roll(13,1)*1.01,vh=raw['vh'].roll(13,1))
 state.refresh_history(fresh['kh'],fresh['vh'],'changed-history');g.replay();torch.cuda.synchronize();newexpected=Pipeline(model,fresh,32,.2,'shared','fp16').run(raw['q'],raw['kc'],raw['vc'])
 assert torch.equal(y,newexpected),'History refresh must update native pages and captured prototype allocations'
 assert not torch.equal(newexpected,actual)
 atomic_json(ROOT/'preflight.json',dict(passed=True,selection_exact=True,graph_exact=True,changed_history_exact=True,attention_nmse=nmse(actual,expected),pipeline_timing=measured));print(measured)
if __name__=='__main__':main()
