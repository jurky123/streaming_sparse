"""Measure writing MMA-ready KV at clean commit and consuming it for clean attention."""
import json
from pathlib import Path
from statistics import median
import sys
import torch
import triton
import triton.language as tl
from local_experiments.video_runtime_v1.fixed15_layout import pack_range


@triton.jit
def packed_indices(token, dim):
    # Invert the SM80 operand fragment permutation for a single head and page.
    key = (dim//16*1024 + token//16*256 + (token%8*4+dim%8//2)*8
           + token%16//8*4 + dim%16//8*2 + dim%2)
    value = (token//16*2048 + dim//16*256 + (dim%8*4+token%8//2)*8
             + dim%16//8*4 + token%16//8*2 + token%2)
    return key, value


@triton.jit
def commit(KG,VG,K,V,PK,PV,START,SK0,SK1,SV0,SV1):
    page=tl.program_id(0)
    head=tl.program_id(1)
    i=tl.program_id(2)*512+tl.arange(0,512)
    row=i//128
    col=i%128
    slot=START+page
    slot=tl.where(slot>=360,slot-336,slot)
    native=((slot*64+row)*12+head)*128+col
    kin,vin=packed_indices(row,col)
    base=(slot*12+head)*8192
    k=tl.load(KG+head*SK0+(page*64+row)*SK1+col)
    v=tl.load(VG+head*SV0+(page*64+row)*SV1+col)
    tl.store(K+native,k)
    tl.store(V+native,v)
    tl.store(PK+base+kin,k)
    tl.store(PV+base+vin,v)


@triton.jit
def clean(Q,PK,PV,OUT,P:tl.constexpr):
    block=tl.program_id(0)
    head=tl.program_id(1)
    row=tl.arange(0,64)
    dim=tl.arange(0,128)
    qptr=((block*64+row[:,None])*12+head)*128+dim[None,:]
    q=tl.load(Q+qptr)
    ik,iv=packed_indices(row[:,None],dim[None,:])
    m=tl.full((64,),-float('inf'),tl.float32)
    den=tl.full((64,),0,tl.float32)
    acc=tl.full((64,128),0,tl.float32)
    for page in range(P):
        base=(page*12+head)*8192
        k=tl.load(PK+base+ik)
        v=tl.load(PV+base+iv)
        z=tl.dot(q,tl.trans(k))*(1.4426950408889634*0.08838834764831845)
        nm=tl.maximum(m,tl.max(z,1))
        prob=tl.exp2(z-nm[:,None])
        fade=tl.exp2(m-nm)
        acc=tl.dot(prob.to(tl.bfloat16),v,acc*fade[:,None])
        den=den*fade+tl.sum(prob,1)
        m=nm
    tl.store(OUT+qptr,acc/den[:,None])


if __name__=='__main__':
    root=Path(__file__).parent
    sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
    tiled="--tile" in sys.argv
    if tiled:sys.argv.remove("--tile")
    from bench import k,v,q,p,timed,times
    kg,vg=[x[0,-4608:].transpose(0,1).contiguous() for x in (k,v)]
    nk,nv,pk,pv=[torch.empty_like(k) for _ in range(4)]
    rpk,rpv=[torch.empty_like(k) for _ in range(2)]
    if tiled:
        import ctypes as C
        lib=C.CDLL('/tmp/jzk_commit_tile.so')
        lib.launch.argtypes=[C.c_void_p]*6+[C.c_int]*5+[C.c_void_p]
    def direct(start):
        if tiled:
            assert lib.launch(*(x.data_ptr() for x in (kg,vg,nk,nv,pk,pv)),start,*kg.stride()[:2],*vg.stride()[:2],torch.cuda.current_stream().cuda_stream)==0
            return
        commit[(72,12,16)](kg,vg,nk,nv,pk,pv,start,*kg.stride()[:2],*vg.stride()[:2])
    def copy(start):
        nk[0,start*64:(start+72)*64].copy_(kg.transpose(0,1))
        nv[0,start*64:(start+72)*64].copy_(vg.transpose(0,1))
        pack_range[(72,12,16)](nk,nv,rpk,rpv,start)
    copy(72)
    direct(72)
    for start in (0,312):
        direct(start)
        for a,b,count in ((start,0,min(360-start,72)),(24,360-start,max(start+72-360,0))):
            if not count:continue
            sl=slice(a*64,(a+count)*64)
            for actual,source in ((nk,kg),(nv,vg)):
                assert torch.equal(actual[0,sl],source[:,b*64:(b+count)*64].transpose(0,1))
            pack_range[(count,12,16)](nk,nv,rpk,rpv,a)
            for actual,expected in ((pk,rpk),(pv,rpv)):
                assert torch.equal(actual[0,sl],expected[0,sl])
    ts=times([timed(lambda:copy(72)),timed(lambda:direct(72))])
    result=dict(commit_reference_ms=median(ts[0]),commit_direct_ms=median(ts[1]),commit_bitwise=True)
    if tiled:
        (root/"commit_tile.json").write_text(json.dumps(result,indent=2));print(result);sys.exit()
    pack_range[(p,12,16)](k,v,pk,pv,0)
    out=torch.empty_like(q)
    def native():
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
    def candidate():clean[(72,12)](q,pk,pv,out,p,num_stages=2,num_warps=4)
    ref=native()
    candidate()
    err=(ref.float()-out.float())
    ts=times([timed(native),timed(candidate)])
    result.update(clean_native_ms=median(ts[0]),clean_direct_ms=median(ts[1]),clean_nmse=float(err.square().sum()/ref.float().square().sum()),clean_max=float(err.abs().max()))
    print(result)
    (root/f'layout_{p}.json').write_text(json.dumps(result,indent=2))
