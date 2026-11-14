"""Actual CUDA FP32 minimal prototype; reference Risk cost, not fused production latency."""
import sys,importlib.util,time,json
import numpy as np
import torch
from .stage34 import ROOT,OUT,online_load,online_features,Objective,axes
from .common import dump
from .problem import ResidualProblem
ops_path=OUT/'code'/'baseline_review_ops.py'
if not ops_path.exists():ops_path=__import__('pathlib').Path('/tmp/group_response_178f95a8/local_experiments/review_v1/ops.py')
spec=importlib.util.spec_from_file_location('frozen_review_ops',str(ops_path));ops=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ops;spec.loader.exec_module(ops)

def run():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 selected=json.loads((OUT/'selected_model.json').read_text());name=selected['method'];lam=float(name.split('_')[0][5:]);nq=int(name.split('_')[1][1:]);pool=name.split('_')[-1];steps=selected['steps'];model=np.load(OUT/f'ridge_{lam}.npz');W=torch.tensor(model['feature'],device='cuda',dtype=torch.float32);coef=torch.tensor(model['coef'],device='cuda',dtype=torch.float32);haar=torch.tensor([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],device='cuda',dtype=torch.float32)
 rows=[]
 for layer in (1,15,29):
  paths=sorted((ROOT/'contributions').glob(f'horse_field_s101_c5_l{layer}_s0_*'));path=paths[0];p=ResidualProblem(path);f=dict(np.load(path/'summary_and_route_features.npz'));t={k:torch.tensor(v,device='cuda') for k,v in f.items() if k in ['query','summary_keys','summary_mean','summary_detail','summary_stats','basis','theta','page_age','page_is_history','probe_indices']};summ=ops.Summary(*(t['summary_'+key][None] for key in ('keys','mean','detail','stats')));probe=t['probe_indices'].long();own=torch.tensor([[p.own]],device='cuda');cfg=ops.Config()
  def route(features):
   scores=(features*t['theta'][None,None,None]).sum(-1).mean(1)[:,None]
   x=ops.RouteOps.select(scores,own,cfg)['mask'][0,0]
   return x,scores[0,0]
  def risk():
   ft,_=ops.RouteOps.features(t['query'][probe][None],summ);return route(ft)[0]
  def predict():
   q=t['query'] if nq==64 else t['query'][probe];ft,_=ops.RouteOps.features(q[None],summ)
   x,scores=route(ft[:,probe] if nq==64 else ft)
   k=t['summary_keys'];v=t['summary_mean'];logits=torch.einsum('qd,prd->qpr',q,k)/(128**.5);weight=(logits-logits.amax((1,2),keepdim=True)).exp();mass=weight.sum(-1);mass=mass/mass.sum(1,keepdim=True);within=weight/weight.sum(-1,keepdim=True)
   det=t['summary_detail']@t['basis'].T;det=torch.einsum('ar,pad->prd',haar,det);pagev=v[None]+torch.einsum('qpr,prd->qpd',within,det);allout=torch.einsum('qp,qpd->qd',mass,pagev)
   Q,P=mass.shape;qp=(q@W)[:,None].expand(Q,P,16);kp=(k.mean(1)@W)[None].expand(Q,P,16);vp=(v@W)[None].expand(Q,P,16);oo=(allout@W)[:,None].expand(Q,P,16)
   feat=torch.cat((qp,kp,vp,pagev@W-oo,qp*kp,ft[0].clamp_min(0).log1p(),mass.clamp_min(1e-30).log()[...,None],(t['page_age']/5)[None,:,None].expand(Q,P,1),t['page_is_history'][None,:,None].expand(Q,P,1),torch.ones((Q,P,1),device='cuda')),-1)
   pred=feat@coef;e=pred[...,:16];d=mass*(pred[...,-1].clamp(-20,20).exp()-1);diag=e.square().sum(-1)
   for _ in range(steps):
    r=torch.where(x)[0];r=r[r!=p.own];a=torch.where(~x)[0];nr,na={'small':(8,16),'medium':(16,32),'large':(32,64)}[pool];r=r[torch.argsort(scores[r],stable=True)[:nr]];a=a[torch.argsort(scores[a],descending=True,stable=True)[:na]]
    E=e[:,~x].sum(1);z=mass.sum(1)+d[:,x].sum(1);h=torch.einsum('qd,qpd->qp',E,e);norm=E.square().sum(-1);nn=norm[:,None,None]+2*(h[:,r,None]-h[:,None,a])+diag[:,r,None]+diag[:,None,a]-2*(e[:,r]@e[:,a].transpose(1,2));zz=z[:,None,None]-d[:,r,None]+d[:,None,a];vals=(nn/zz.square()).sum(0);ij=vals.argmin();ri=ij//len(a);ai=ij%len(a);current=(norm/z.square()).sum()
    if vals[ri,ai]>=current-max(1e-12,abs(float(current))*1e-9):break
    x[r[ri]]=False;x[a[ai]]=True
   return x
  assert np.array_equal(risk().cpu().numpy(),p.data['risk'])
  for _ in range(3):risk();predict()
  torch.cuda.synchronize();times={'risk':[],'predict':[]}
  for i in range(10):
   for tag,fn in ([('risk',risk),('predict',predict)] if i%2==0 else [('predict',predict),('risk',risk)]):
    torch.cuda.synchronize();s=time.perf_counter();x=fn();torch.cuda.synchronize();times[tag].append((time.perf_counter()-s)*1000)
  x=predict().cpu().numpy();stored=json.loads((OUT/'t4'/f'{path.name}.json').read_text())['results'][name]['trace'][steps];xcpu=np.zeros(p.P,bool);xcpu[stored['selected']]=1
  rows.append(dict(event_id=path.name,layer=layer,method=name,steps=steps,risk_reference_ms=float(np.median(times['risk'])),predictor_total_ms=float(np.median(times['predict'])),extra_ms=float(np.median(times['predict'])-np.median(times['risk'])),samples=times,cuda_cpu_mask_equal=bool(np.array_equal(x,xcpu)),cuda_selected=np.flatnonzero(x).tolist(),cuda_true_nmse=p.value(x),cpu_true_nmse=p.value(xcpu),risk_nmse=p.value(p.data['risk'])))
 dump(OUT/'gpu_cost.json',dict(gpu=torch.cuda.get_device_name(),precision='FP32 TF32 disabled',protocol='3 warmups, 10 alternating measurements, same preloaded real chunk5 layer1/15/29; synchronized wall time; includes frozen unfused Risk features and Python dispatch; excludes cache/transfer/attention; NOT fused production kernel cost',rows=rows));print(json.dumps(rows,indent=2))
if __name__=='__main__':run()
