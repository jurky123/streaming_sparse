import os,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4)
from local_experiments.review_v1.model import Layer
from local_experiments.review_v1.fused_summary import build,_pool,_summary
r=Path('/tmp/latest_kernel_optimization4');samples=torch.load('/tmp/latest_kernel_optimization2/summary_inputs.pt',map_location='cuda',weights_only=True);rows=[]
for mode,x in samples.items():
 m=Layer(x['state']);m.load_state_dict(x['state']);k=x['k'];v=x['v'];h,p,t,d=k.shape;cheap=mode=='cheap'
 pk=torch.empty(h,p,4,128,device='cuda',dtype=torch.bfloat16);pv=torch.empty(h,p,4,128,device='cuda');mu=torch.empty(h,p,128,device='cuda',dtype=torch.bfloat16);detail=torch.empty(h,p,3,8,device='cuda',dtype=torch.bfloat16);st=torch.empty(h,p,4,12,device='cuda',dtype=torch.bfloat16)
 a=_pool[(h*p,)](k,v,m.bank.contiguous(),m.group_v.contiguous(),m.group_gate.contiguous(),m.pool_k.contiguous(),m.pool_v.contiguous(),m.pool_feature.contiguous(),pk,pv,p,*k.stride()[:3],*v.stride()[:3],cheap,num_warps=4,num_stages=2,enable_fp_fusion=False)
 b=_summary[(h*p,)](k,v,pk,pv,m.basis,mu,detail,st,p,*k.stride()[:3],*v.stride()[:3],*m.basis.stride(),num_warps=4,enable_fp_fusion=False)
 for _ in range(5):build(m,k,v,cheap)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  for _ in range(20):build(m,k,v,cheap)
  torch.cuda.synchronize()
 trace=r/(mode+'_summary.trace.json');prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents'];times={n:sum(e['dur'] for e in events if e.get('cat')=='kernel' and e['name']==n)/20000 for n in ['_pool','_summary']}
 row=dict(mode=mode,ms_per_kernel=times,metadata={n:dict(registers=f.n_regs,spills=f.n_spills,shared=f.metadata.shared) for n,f in [('_pool',a),('_summary',b)]});rows.append(row);print(row,flush=True)
(r/'summary_profile.json').write_text(json.dumps(rows,indent=2))
