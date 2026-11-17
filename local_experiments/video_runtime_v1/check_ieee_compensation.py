"""Isolate P×V precision: IEEE FP32 versus historical four-BF16-products tail."""
import json
from pathlib import Path
import torch
import triton as tr
import triton.language as tl
from .benchmark import measure
from .profile_route import capture

@tr.jit
def kernel(P,V,O,MODE:tl.constexpr):
    i=tl.program_id(0);m=tl.arange(0,64);k=tl.arange(0,64);d=tl.arange(0,128)
    p=tl.load(P+i*4096+m[:,None]*64+k[None,:]);v=tl.load(V+i*8192+k[:,None]*128+d[None,:])
    if MODE==1:y=tl.dot(p,v,input_precision='ieee')
    else:
        ph=p.to(tl.bfloat16);pl=(p-ph.to(tl.float32)).to(tl.bfloat16)
        vh=v.to(tl.bfloat16);vl=(v-vh.to(tl.float32)).to(tl.bfloat16)
        y=tl.dot(ph,vh)+tl.dot(ph,vl)+tl.dot(pl,vh)+tl.dot(pl,vl)
    tl.store(O+i*8192+m[:,None]*128+d[None,:],y)

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(701)
    p=torch.rand(128,64,64,device='cuda');v=torch.randn(128,64,128,device='cuda');ref=p.double()@v.double();rows={}
    out=Path('artifacts/video_runtime_v1/ieee_compensation');out.mkdir(parents=True,exist_ok=True)
    for mode in (4,1):
        y=torch.empty(128,64,128,device='cuda');compiled=kernel[(128,)](p,v,y,mode,num_warps=8)
        g,_=capture(lambda:kernel[(128,)](p,v,y,mode,num_warps=8));diff=y.double()-ref
        rows[str(mode)]=dict(nmse=float(diff.square().sum()/ref.square().sum()),max_abs=float(diff.abs().max()),timing=measure(g.replay,25),ptx_has_mma='mma.sync' in compiled.asm['ptx'])
        (out/f'dot_mode{mode}.ptx').write_text(compiled.asm['ptx'])
        if mode==1:assert not rows['1']['ptx_has_mma'] and rows['1']['nmse']<1e-12
    (out/'dot_precision.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':run()
