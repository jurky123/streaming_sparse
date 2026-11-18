"""Test the tiled lane traversal in large transposed GEMVs, including vector unrolling."""
import json
import torch
import triton as tr
import triton.language as tl
from .weights import ROOT

@tr.jit
def _trans(X,V,O,N:tl.constexpr,L:tl.constexpr,PACK:tl.constexpr,ACC:tl.constexpr,BITS:tl.constexpr):
    d=tl.program_id(0);g=tl.program_id(1);lane=tl.arange(0,L)
    a=tl.full((L,),0,tl.float32);b=tl.full((L,),0,tl.float32)
    for j in range(N//L//2):
        i=2*j
        i0=i//PACK*L*PACK+lane*PACK+i%PACK
        i1=(i+1)//PACK*L*PACK+lane*PACK+(i+1)%PACK
        a=tl.fma(tl.load(X+(g*N+i0)*32+d),tl.load(V+g*N+i0),a)
        if ACC:b=tl.fma(tl.load(X+(g*N+i1)*32+d),tl.load(V+g*N+i1),b)
        else:a=tl.fma(tl.load(X+(g*N+i1)*32+d),tl.load(V+g*N+i1),a)
    if ACC:a+=b
    for i in tl.static_range(BITS):
        a+=tl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",constraints="=f,f,r",args=[a,L>>(i+1)],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(O+g*32+d+lane,a,lane==0)

def main():
    torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for n in (256,1536,4608):
        g=12*4608//n
        x=torch.randn(g,n,32,device='cuda')
        v=torch.randn(g,n,1,device='cuda')
        ref=torch.bmm(x.transpose(1,2),v).squeeze(-1)
        out=torch.empty_like(ref);results=[]
        for l in ((2,4) if n==256 else (8,16,32)):
            for pack in (1,2,4,8,16,32):
                if n%(l*pack):continue
                for acc in (False,True):
                    _trans[(32,g)](x,v,out,n,l,pack,acc,l.bit_length()-1,num_warps=1,enable_fp_fusion=False)
                    results.append(dict(lanes=l,pack=pack,acc=acc,equal=float((ref==out).float().mean())))
        best=sorted(results,key=lambda r:-r['equal'])[:5]
        print(n,best,flush=True);rows.append(dict(n=n,best=best))
    (ROOT/'artifacts/video_runtime_v1/group_gemv_order_v1/segments.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
