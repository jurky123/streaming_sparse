from bench import *
from local_experiments.gap_exact_round_kernel import _anchored_exact
import triton as tr
import triton.language as tl
import ctypes as C
@tr.jit
def reference(Q,K,V,IDS,O,L,M,CNT:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)
    i=(t*64+tl.arange(0,64))*12+h
    address=i[:,None]*128+tl.arange(0,128)[None,:]
    q=tl.load(Q+address)
    o,l,m=_anchored_exact(q,K,V,IDS+(h*72+t)*CNT,h,12,128,CNT,128**-.5,True)
    tl.store(O+address,o);tl.store(L+i,l);tl.store(M+i,m)
lib=C.CDLL('/tmp/jzk_fixed15_exact.so')
lib.run.argtypes=[C.c_void_p]*7+[C.c_int,C.c_void_p]
def outputs():return [torch.empty_like(q,dtype=torch.float32),*(torch.empty(q.shape[:-1],device='cuda') for _ in range(2))]
ref=outputs();out=outputs()
def triton_ref():return reference[(72,12)](q,k,v,ids,*ref,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
def cuda_run():
    status=lib.run(*(x.data_ptr() for x in (q,k,v,ids,*out)),count,torch.cuda.current_stream().cuda_stream)
    assert status==0,status
comp=triton_ref();cuda_run()
(d/'reference.ptx').write_text(comp.asm['ptx'])
ts=times([timed(triton_ref),timed(cuda_run),timed(base)])
r=dict(ms=list(map(median,ts)),exact_speedup=median(x/y for x,y in zip(*ts[:2])),guard_pass=bool(((ref[-1]>=1)&(ref[-1]<1.21e24)).all()),outputs=[])
for a,b in zip(ref,out):
    diff=b-a
    r['outputs'].append(dict(bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),nmse=float(diff.square().sum()/a.square().sum()),max_abs=float(diff.abs().max()),changed=int((a!=b).sum())))
print(r);(d/f'cuda_gate_{p}.json').write_text(json.dumps(r,indent=2))
