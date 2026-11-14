import time,json
import numpy as np
import torch
from .core import *
from .train import Net

def run():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 networks={};models={};selections={}
 for arch in ('A0','A1','A2'):
  name=arch+'_s17';m=np.load(OUT/'models'/name/'weights.npz');net=Net();net.load_state_dict({k:torch.from_numpy(m[k]) for k in ('fc1.weight','fc1.bias','fc2.weight','fc2.bias')});net.eval();networks[name]=net;models[name]=m;selections[name]=json.loads((OUT/'models'/name/'selection.json').read_text())['selected']['threshold']
 rows=[]
 for layer in (1,15,29):
  path=sorted((DATA/'contributions').glob(f'horse_field_s101_c5_l{layer}_s0_*'))[0];f=load_online(path);s=np.load(OUT/'states'/f'{path.name}.npz');own=int(s['own']);K=int(s['K'])
  for name in networks:
   m=models[name];net=networks[name];threshold=selections[name];threshold=float('inf') if threshold is None else threshold
   def once():
    t=time.perf_counter();p=pages(f);t1=time.perf_counter();x=initial_mask(p,own,K);F,r,a=features(p,x,own);t2=time.perf_counter();xx=(F-m['mean'])/m['std'];xx[:,int(m['slots']):]=0
    with torch.no_grad():scores=net(torch.tensor(xx,dtype=torch.float32)).numpy()
    idx=scores.argmax();take=bool(scores[idx]>threshold);t3=time.perf_counter();return dict(page_ms=(t1-t)*1000,candidate_ms=(t2-t1)*1000,normalize_score_decide_ms=(t3-t2)*1000,total_ms=(t3-t)*1000,action=[int(r[idx]),int(a[idx])] if take else None),xx
   for _ in range(3):once()
   samples=[]
   for _ in range(10):row,xx=once();samples.append(row)
   gpu=Net().cuda();gpu.load_state_dict(net.state_dict());gx=torch.tensor(xx,dtype=torch.float32,device='cuda')
   with torch.no_grad():
    for _ in range(3):gpu(gx)
    torch.cuda.synchronize();gt=[]
    for _ in range(10):
     begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();gpu(gx);end.record();end.synchronize();gt.append(begin.elapsed_time(end))
   rows.append(dict(event_id=path.name,layer=layer,name=name,**{k:float(np.median([r[k] for r in samples])) for k in ('page_ms','candidate_ms','normalize_score_decide_ms','total_ms')},gpu_scoring_only_ms=float(np.median(gt)),samples=samples))
 result=dict(protocol='preloaded same chunk5 layer1/15/29, 3 warmups,10 measurements; CPU one thread reference includes complete32-slot feature generation even for ablations; GPU measurement is scoring-only with preloaded features, excludes feature generation and transfers. Not fused production latency, not directly comparable to prior GPU total-router timing.',torch_threads=1,gpu=torch.cuda.get_device_name(),parameters=1089,score_flops_per_tile=2*512*(32*32+32),score_flops_per_layer=2*72*12*512*(32*32+32),excluded_from_flops=['GELU','bias','feature generation','candidate selection','normalization','memory','launch'],rows=rows)
 dump(OUT/'reference_cost.json',result);print('BENCHMARK_DONE',len(rows))
if __name__=='__main__':run()
