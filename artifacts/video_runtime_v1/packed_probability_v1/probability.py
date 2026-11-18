import importlib
from pathlib import Path
import sys
import tempfile
from local_experiments.video_runtime_v1.weights import ROOT

_folder=tempfile.TemporaryDirectory()
sys.path.insert(0,_folder.name)
split='''
@triton.jit
def split_pair(x):
    bits=tl.inline_asm_elementwise(
        asm="""{
        .reg .b32 a,b,hp,lp;
        cvt.rn.bf16x2.f32 hp,$3,$2;
        shl.b32 a,hp,16;
        and.b32 b,hp,0xffff0000;
        sub.rn.f32 a,$2,a;
        sub.rn.f32 b,$3,b;
        cvt.rn.bf16x2.f32 lp,b,a;
        prmt.b32 $0,hp,lp,0x1054;
        prmt.b32 $1,hp,lp,0x3276;
        }""",constraints="=r,=r,f,f",args=[x],dtype=tl.uint32,is_pure=True,pack=2)
    hi=(bits>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    lo=bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)
    return hi,lo
'''
def variant_for(src,mode):
    ex=(ROOT/'local_experiments/gap_exact_round_kernel.py').read_text()
    name='exact_packed_'+mode
    if mode in ('exact','both'):
        ex=ex.replace('@triton.jit\ndef _attn_fwd',split+'\n@triton.jit\ndef _attn_fwd').replace('ph = p.to(v.dtype)\n            pl = (p - ph.to(tl.float32)).to(v.dtype)','ph,pl=split_pair(p)')
    if mode.startswith('convert'):
        cvt="""
@triton.jit
def to_bf16_pair(x):
    return tl.inline_asm_elementwise("cvt.rn.bf16x2.f32 $0,$2,$1;",constraints="=r,f,f",
                                     args=[x],dtype=tl.bfloat16,is_pure=True,pack=2)
"""
        ex=ex.replace('@triton.jit\ndef _attn_fwd',cvt+'\n@triton.jit\ndef _attn_fwd')
        if mode in ('convertboth','converthi'):
            ex=ex.replace('ph = p.to(v.dtype)','ph = to_bf16_pair(p)')
        if mode in ('convertboth','convertlo'):
            ex=ex.replace('pl = (p - ph.to(tl.float32)).to(v.dtype)','pl = to_bf16_pair(p-ph.to(tl.float32))')
    if mode.startswith('stack'):
        if mode=='stackrow':
            code="""paired=tl.reshape(tl.permute(tl.join(ph,pl),(0,2,1)),(2*BLOCK_M,BLOCK_N))
            both=tl.dot(paired,v)
            hi,lo=tl.split(tl.permute(tl.reshape(both,(BLOCK_M,2,D)),(0,2,1)))
            page_av=hi+lo"""
        else:
            code="""paired=tl.reshape(tl.permute(tl.join(ph,pl),(2,0,1)),(2*BLOCK_M,BLOCK_N))
            both=tl.dot(paired,v)
            hi,lo=tl.split(tl.permute(tl.reshape(both,(2,BLOCK_M,D)),(1,2,0)))
            page_av=hi+lo"""
        ex=ex.replace('page_av = tl.dot(ph,v) + tl.dot(pl,v)',code)
    if mode.startswith('max'):
        ex=ex.replace('        qk = tl.dot(q, k)','        raw = tl.dot(q, k)').replace('args=[qk, qk_scale','args=[raw, qk_scale')
        ex=ex.replace('local_m = tl.max(qk, 1)','local_m = tl.max(raw,1)*(qk_scale*1.4426950408889634)')
        if mode=='maxsplit':
            ex=ex.replace('ph = p.to(v.dtype)\n            pl = (p - ph.to(tl.float32)).to(v.dtype)','ph,pl=split_pair(p)')
            ex=ex.replace('@triton.jit\ndef _attn_fwd',split+'\n@triton.jit\ndef _attn_fwd')
    if mode.startswith('qk') or mode=='qkv64':
        width=64 if mode=='qkv64' else int(mode[2:])
        qloads=[f'    band=tl.arange(0,{width})']
        dots=['        qk=tl.full((BLOCK_M,BLOCK_N),0,tl.float32)']
        for part in range(128//width):
            qloads.append(f'    q{part}=tl.load(Q+q_offset+offs_m[:,None]*HD+{part*width}+band[None,:])')
            dots.extend([f'        k{part}=tl.load(K+kv_offset+(k_start+offs_n)[None,:]*kv_stride+{part*width}+band[:,None])',
                         f'        qk=tl.dot(q{part},k{part},qk)'])
        ex=ex.replace('    q = tl.load(Q_ptrs)','\n'.join(qloads))
        ex=ex.replace('        k = tl.load(K_ptrs)\n        qk = tl.dot(q, k)','\n'.join(dots))
    if mode in ('av64','qkv64'):
        ex=ex.replace('        v = tl.load(V_ptrs)','        halfd=tl.arange(0,64)\n        vp=V+kv_offset+(k_start+offs_n)[:,None]*kv_stride+halfd[None,:]\n        v0=tl.load(vp)\n        v1=tl.load(vp+64)')
        ex=ex.replace('p.to(v.dtype)','p.to(tl.bfloat16)').replace(')).to(v.dtype)',')).to(tl.bfloat16)')
        ex=ex.replace('page_av = tl.dot(ph,v) + tl.dot(pl,v)',"""lo=tl.dot(ph,v0)+tl.dot(pl,v0)
            hi=tl.dot(ph,v1)+tl.dot(pl,v1)
            page_av=tl.reshape(tl.permute(tl.join(lo,hi),(0,2,1)),(BLOCK_M,D))""")
    Path(_folder.name,name+'.py').write_text(ex)
    src=src.replace('from local_experiments.gap_exact_round_kernel import',f'from {name} import')
    if mode in ('tail','both'):
        src=src.replace('@tr.jit\ndef _av',split.replace('@triton.jit','@tr.jit')+'\n@tr.jit\ndef _av').replace('hi=a.to(tl.bfloat16);lo=(a-hi.to(tl.float32)).to(tl.bfloat16)','hi,lo=split_pair(a)')
    mod='tail_packed_'+mode
    Path(_folder.name,mod+'.py').write_text(src)
    return importlib.import_module(mod)._tail
