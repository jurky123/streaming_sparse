import argparse,torch,math
from .setup import ROOT,tag
from .prep import query,current
from .pipeline import Pipeline as New
from ..opt26.pipeline import Pipeline as Old
from ..opt25.pipeline import Workspace
from ..opt24.probe import load
from ..opt26.probe import measure
from ..opt16.grouping import order,gather
from ..opt8.projection import project
from ..opt24.sparge import quant
from ..opt5.probe import capture
from ..common import atomic_json
class LargeWorkspace(Workspace):
 def view(self,name,shape,dtype,max_elements):return super().view(name,shape,dtype,max(max_elements,math.prod(shape)))
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',720),('width',1280),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,_,ref,aux=load(a.height,a.width,a.chunks,a.layer);old=Old(model,raw,True,LargeWorkspace(raw));new=New(model,raw,True,LargeWorkspace(raw))
 # Inspect every fused intermediate rather than relying on a single selected mask.
 oldkv=old.cache.current(raw['kc'],raw['vc']);old.quant.current();old.sync(old.quant.nhpages,old.quant.b)
 newkv=current(new,raw['kc'],raw['vc']);checks={}
 for name,x,y in [('grouped_k',old.cache.k,new.cache.k),('grouped_v',old.cache.v,new.cache.v),('quant_k',old.quant.k,new.quant.k),('quant_scale',old.quant.scale,new.quant.scale),('packed_k',old.pk,new.pk),('packed_v',old.pv,new.pv),('prototypes',oldkv.state['key'],newkv.state['key']),('counts',oldkv.state['logcount'],newkv.state['logcount'])]:checks[name]=torch.equal(x,y)
 ix=order(raw['q'],model.tree('q'));gq=gather(raw['q'],ix);expected_proj=project(gq,oldkv.state['basis']);expected_qi,expected_qs=quant(gq,32);qp,qi,qs=query(raw['q'],ix,oldkv.state['basis'])
 checks.update(projection=torch.equal(qp,expected_proj),query_quant=torch.equal(qi,expected_qi),query_scale=torch.equal(qs,expected_qs));print('checks',checks,flush=True);assert all(checks.values()),checks
 bg,bo=capture(lambda:old.run(raw['q'],raw['kc'],raw['vc']));bg.replay();ref=bo.clone();rows=[]
 for mode in ['both']:
  new.mode=mode;r=measure(lambda:new.run(raw['q'],raw['kc'],raw['vc']),bg,ref,reps=25);r['mode']=mode;rows.append(r);print(r,flush=True)
 out=ROOT/'confirm'/tag(a.height,a.width,a.chunks)/f'layer{a.layer:02d}.json';out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,dict(completed=True,checks=checks,rows=rows))
if __name__=='__main__':main()
