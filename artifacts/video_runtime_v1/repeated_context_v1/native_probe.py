import ctypes
import json
import statistics
import sys
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.profile_route import capture

def library(path):
    lib=ctypes.CDLL(str(path))
    lib.run_ctx.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p]
    lib.run_ctx.restype=ctypes.c_int
    return lib

def call(lib,q,k,v,compress=True):
    out=torch.empty_like(q)
    lse=torch.empty(12,4608,device=q.device)
    error=lib.run_ctx(*(x.data_ptr() for x in (q,k,v,out,lse)),int(compress),torch.cuda.current_stream().cuda_stream)
    if error:raise RuntimeError(error)
    return out

def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'ctx'
    lib=library('/tmp/jzk_native_context/'+name+'.so')
    rows=[]
    for triple in torch.load('/tmp/jzk_cross_three.pt',weights_only=True):
        q,k,v=[x.cuda() for x in triple]
        funcs=[lambda:torch.nn.functional.scaled_dot_product_attention(*(a.transpose(1,2) for a in (q,k,v))).transpose(1,2),
               lambda:call(lib,q,k,v,False),lambda:call(lib,q,k,v,True)]
        cases=[capture(f) for f in funcs]
        samples=[[] for _ in cases]
        for rep in range(30):
            for j in range(3):
                i=(j+rep)%3
                a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                for _ in range(16):cases[i][0].replay()
                b.record();b.synchronize()
                if rep>=4:samples[i].append(a.elapsed_time(b)/16)
        ref=cases[0][1].float()
        row=[dict(ms=statistics.median(t),different=int((out!=ref).sum()),max_abs=float((out.float()-ref).abs().max()),samples=t) for t,(_,out) in zip(samples,cases)]
        rows.append(row)
        print([(r['ms'],r['different']) for r in row],flush=True)
    Path(__file__).with_name(name+'_module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
