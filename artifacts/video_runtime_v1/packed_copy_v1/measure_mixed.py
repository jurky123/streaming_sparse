"""Compare preparation alone and the actual Q64 attention module on a captured window."""
import ctypes as C,json,sys
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_layout import encode,pack_range,retry
from local_experiments.video_runtime_v1.fixed15_attention import library
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import q,k,v,keys,mean,detail,ids,basis,p,hp,count,timed,times
pk,pv=torch.empty_like(k),torch.empty_like(v)
pack_range[(p,12,16)](k,v,pk,pv,0)
tiles=(p+15)//16
summ=q.new_empty((12,tiles,10624))
out=torch.empty_like(q);flags=q.new_empty((12,72),dtype=torch.int32)
lib=C.CDLL('/tmp/jzk_mixed_current.so')
lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]*3
def control():
    pack_range[(72,12,16)](k,v,pk,pv,p-72)
    encode[(12,tiles,42)](keys,mean,detail,summ,p,hp,tiles,num_warps=4)
def candidate():
    encode[(12,tiles,42)](keys,mean,detail,summ,p,hp,tiles,num_warps=4)
def full(prep):
    prep()
    args=[t.data_ptr() for t in (q,pk,pv,ids,out,summ,basis,flags)]+[p,count]
    fn=library().run
    if prep is candidate:
        fn=lib.run
        args.extend([k.data_ptr(),v.data_ptr()])
    assert fn(*args,torch.cuda.current_stream().cuda_stream)==0
    retry[(72,12)](flags,q,k,v,ids,keys,mean,detail,basis,out,p,hp,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
full(control);expected=[t.clone() for t in (pk,pv,summ,out,flags)]
pk[:,(p-72)*64:].zero_();pv[:,(p-72)*64:].zero_();summ.zero_()
full(candidate)
assert all(torch.equal(t,e) for t,e in zip((summ,out,flags),expected[2:]))
xs=times([timed(f) for f in (control,candidate,lambda:full(control),lambda:full(candidate))])
row=dict(p=p,current_fraction=float((ids>=p-72).float().mean()),bitwise=True,ms=list(map(median,xs)),paired_speedups=[median(a/b for a,b in zip(*xs[i:i+2])) for i in (0,2)],samples=xs)
(root/f'mixed_{p}.json').write_text(json.dumps(row,indent=2));print(row['ms'],row['paired_speedups'])
