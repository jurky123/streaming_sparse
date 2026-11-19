exec(open('/home/shuchenweng/jzk/streaming_sparse/local_setup/check_three_term_summary.py').read().split('rows=[];graph=GraphSummary()')[0])
rows=[]
with torch.no_grad():
 for mode,x in samples.items():
  m=Layer(x['state']);m.load_state_dict(x['state']);cheap=mode=='cheap'
  for scale in [0.,.0001,.01,1.]:
   k=(x['k'].float().mean(-2,keepdim=True)+scale*(x['k'].float()-x['k'].float().mean(-2,keepdim=True))).bfloat16();v=x['v']
   ref=SummaryOps.build(m,k,v,cheap);out=build(m,k,v,cheap);errors={}
   for n in ['keys','mean','detail','stats']:
    a=getattr(out,n).float();b=getattr(ref,n).float();d=a-b;errors[n]=dict(nmse=float(d.square().mean()/b.square().mean().clamp_min(1e-12)),max_abs=float(d.abs().max()),finite=bool(a.isfinite().all()))
   row=dict(mode=mode,scale=scale,errors=errors);rows.append(row);print(row,flush=True)
   assert all(e['finite'] and e['nmse']<2e-7 for e in errors.values()),row
 (r/'summary_stress_checks.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
