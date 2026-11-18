import ctypes as C
import json
import sys
from pathlib import Path
from statistics import mean,median
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,pk,pv,ids,basis,p,hp,count,summary,keys,mean as value_mean,detail,implementation,timed,times
tiles=(p+15)//16
encode[(12,tiles,42)](keys,value_mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
base,expected,flags=implementation(library())
so=C.CDLL('/tmp/jzk_issue_stamps.so')
so.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]*2
out=torch.empty_like(q);bad=torch.empty_like(flags);marks=torch.empty((864,3),device=q.device,dtype=torch.int64)
args=[t.data_ptr() for t in (q,pk,pv,ids,out,summary,basis,bad)]
def run():
    assert so.run(*args,p,count,marks.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
for _ in range(30):run()
samples=[]
for _ in range(12):
    run()
    blocks=marks.tolist()
    by_sm={}
    for sm,a,b in blocks:
        assert b>a
        by_sm.setdefault(sm,[]).append((a,b))
    assert len(by_sm)==108
    start=min(a for sm,a,b in blocks);end=max(b for sm,a,b in blocks)
    stops=[max(b for a,b in v) for v in by_sm.values()]
    starts=[min(a for a,b in v) for v in by_sm.values()]
    samples.append(dict(span_us=(end-start)/1000,mean_start_lag_us=(mean(starts)-start)/1000,
                        mean_finish_slack_us=(end-mean(stops))/1000,
                        min_ctas=min(map(len,by_sm.values())),max_ctas=max(map(len,by_sm.values())),
                        last_sm_margin_us=(end-sorted(stops)[-2])/1000))
assert torch.equal(out.view(torch.uint8),expected.view(torch.uint8)) and not bad.any()
x,y=times([timed(base),timed(run)])
row=dict(p=p,bitwise=True,instrumented_over_base=median(b/a for a,b in zip(x,y)),samples=samples,
         medians={k:median(r[k] for r in samples) for k in samples[0]})
(root/f'waves_{p}.json').write_text(json.dumps(row,indent=2));print(row['medians'])
