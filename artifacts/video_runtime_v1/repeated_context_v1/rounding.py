import json
from pathlib import Path
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def rounded(Q,K,V,O,H:tl.constexpr,D:tl.constexpr,M:tl.constexpr,S:tl.constexpr,MODE:tl.constexpr):
    m=tl.program_id(0)*M+tl.arange(0,M)
    h=tl.program_id(1)
    n=tl.arange(0,64)
    d=tl.arange(0,D)
    lane=tl.arange(0,4)
    q=tl.load(Q+(m[:,None]*H+h)*D+d[None,:])
    k=tl.load(K+(n[None,:]*H+h)*D+d[:,None])
    v=tl.load(V+(n[:,None]*H+h)*D+d[None,:])
    z=tl.dot(q,k)
    mx=tl.max(z,1)
    pad=tl.sum(tl.where(n[None,:]==63,z,0.),1)
    prev=tl.exp2((pad-mx)*S)
    w=tl.exp2(tl.fma(z,S,-(mx*S)[:,None]))
    if MODE&16:
        w=tl.exp2(z*S-(mx*S)[:,None])
    if MODE&128:
        w=libdevice.exp2(z*S-(mx*S)[:,None])
        prev=libdevice.exp2((pad-mx)*S)
    pval=tl.load(V+(63*H+h)*D+d).to(tl.float32)
    accum=(448.*pval[None,:])*prev[:,None]
    out=tl.dot(w.to(tl.bfloat16),v,accum)
    weight=tl.exp2(tl.fma(pad,S,-(pad*S)))
    if MODE&8:
        local=tl.full((M,4),112.,tl.float32)
    else:
        local=tl.full((M,4),0.,tl.float32)
        for j in tl.static_range(16):
            local+=weight[:,None]
        one=local
        for block in tl.static_range(6):
            local+=one
    old=local*prev[:,None]
    if MODE&1:
        local=old
    else:
        local=tl.full((M,4),0.,tl.float32)
    even,odd=tl.split(tl.reshape(w,(M,8,4,2)))
    for j in tl.static_range(16):
        part=odd if ((j//8) if MODE&4 else (j%2)) else even
        slot=j%8 if MODE&4 else j//2
        local+=tl.sum(tl.where(tl.arange(0,8)[None,:,None]==slot,part,0.),1)
    if not MODE&1:
        local+=old
    if MODE&32:
        lo,hi=tl.split(tl.reshape(local,(M,2,2)))
        den=tl.sum(lo,1)+tl.sum(hi,1)
    else:
        den=tl.sum(local,1)
    if MODE&2:
        inv=tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;',constraints='=f,f',args=[den],dtype=tl.float32,is_pure=True,pack=1)
    else:
        inv=1./den
    if MODE&64:
        inv=tl.div_rn(1.,den)
    tl.store(O+(m[:,None]*H+h)*D+d[None,:],out*inv[:,None])

def main():
    inputs=torch.load('/tmp/jzk_cross_three.pt',weights_only=True)
    rows=[]
    import sys
    modes=range(16)
    if len(sys.argv)>1:
        modes=range(16,64) if sys.argv[1]=="more" else [25+g*32 for g in range(2,8)]
    for mode in modes:
        cases=[]
        for q,k,v in inputs:
            q,k,v=(a.cuda() for a in (q,k,v))
            ref=torch.nn.functional.scaled_dot_product_attention(*(a.transpose(1,2) for a in (q,k,v))).transpose(1,2)
            o=torch.empty_like(q)
            rounded[(72,12)](q,k,v,o,12,128,64,128**-.5*1.4426950408889634,mode,num_warps=4,enable_fp_fusion=False)
            cases.append(int((o!=ref).sum()))
        rows.append(cases);print(mode,cases,flush=True)
        Path(__file__).with_name('rounding_'+sys.argv[1]+'.json' if len(sys.argv)>1 else 'rounding.json').write_text(json.dumps(rows))
if __name__=='__main__':main()
