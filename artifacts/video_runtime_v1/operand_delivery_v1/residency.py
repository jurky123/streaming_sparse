"""Vary residency before each timed P432 launch."""
import json
import torch
import ctypes as C
from statistics import median
from local_experiments.video_runtime_v1.fixed15_attention import library
from capture import detail, hp, implementation, k, keys, mean, p, pk, pv, q, root, summary, timed, v

control, _, _ = implementation(C.CDLL('/tmp/jzk_delivery_control.so'))
candidate, _, _ = implementation(C.CDLL('/tmp/jzk_delivery_fused_center.so'))
graphs=[timed(f) for f in (control,candidate)]
trash=torch.empty(96*1024*1024,device=q.device,dtype=torch.uint8)
def prepare():
    lib=library()
    stream=torch.cuda.current_stream().cuda_stream
    assert lib.prepare_fixed15(*(x.data_ptr() for x in (k,v,pk,pv,keys,mean,detail,summary)),p,hp,stream)==0

samples={}
for label in ('resident','cold','after_prepare'):
    results=[[],[]]
    for rep in range(36):
        for i in ((1,0) if rep%2 else (0,1)):
            if label!='resident':
                trash.zero_()
            if label=='after_prepare':
                prepare()
            first,last=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            first.record()
            graphs[i].replay()
            last.record()
            last.synchronize()
            results[i].append(first.elapsed_time(last))
    samples[label]=dict(ms=list(map(median,results)),speedup=median(x/y for x,y in zip(*results)),samples=results)
(root/'residency.json').write_text(json.dumps(samples,indent=2))
