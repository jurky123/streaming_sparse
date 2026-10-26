"""Confirm production configuration on held-out real attention blocks."""
import argparse,torch,gzip,json
from .setup import ROOT,tag
from .route import optimized
from .probe import measure
from ..opt24.probe import load
from ..opt23.route import scores
from ..opt19.tile import select
from ..opt5.probe import capture
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',720),('width',1280),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];state=aux['kv'].state;b=state['key'].shape[1];seg=pipe.seg
 def old():
  f,c=scores(q,state,.35,8,seg);return select(f,c,b,32,.2)
 bg,bo=capture(old);bg.replay();reference=bo.clone();r=measure(lambda:optimized(q,state,seg),bg,reference,reps=40)
 out=ROOT/'verify'/tag(a.height,a.width,a.chunks)/f'layer{a.layer:02d}.json';out.parent.mkdir(parents=True,exist_ok=True)
 r.update(height=a.height,width=a.width,chunks=a.chunks,layer=a.layer,completed=True)
 if a.layer==24:
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
   optimized(q,state,seg);torch.cuda.synchronize()
  r['kernels']=[dict(name=e.key,ms=e.device_time_total/1000,calls=e.count) for e in prof.key_averages() if e.device_type==torch.autograd.DeviceType.CUDA]
 if (a.height,a.width,a.chunks,a.layer)==(720,1280,4,24):
  drift=torch.load(ROOT/'first_drift.pt',map_location='cuda',weights_only=True)
  df,dc=scores(drift['q'],drift['state'],.35,8,drift['seg']);db=drift['state']['key'].shape[1]
  expected=select(df,dc,db,32,.2);actual=optimized(drift['q'],drift['state'],drift['seg'])
  r['previous_drift_regression_exact']=torch.equal(actual,expected);assert r['previous_drift_regression_exact']
 atomic_json(out,r);print(r,flush=True);assert r['exact'],'Selected pages changed'
if __name__=='__main__':main()
