"""Time complete FFNs with the rounded BF16 GELU folded into the first GEMM."""
import argparse
import ctypes
import json
import statistics
import torch
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from .weights import ROOT
from .accepted_tiers import config
from .deployment import configure
from .profile_route import capture

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',help='Optional prebuilt benchmark kernel')
    args=ap.parse_args()
    configure(config('base',local_assets='/tmp/streaming_sparse-0/runtime'))
    torch.set_num_threads(4)
    engine=Engine()
    ffn=engine.wrapper.model.blocks[15].ffn
    inputs=[]
    def save(_,args):
        if not inputs:inputs.append(args[0].detach().clone())
    hook=ffn[0].register_forward_pre_hook(save)
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    generate_chunk(engine,emb,701,0)
    hook.remove()
    x=inputs[0].reshape(-1,1536).contiguous()
    from .ffn_rounded_gelu import library
    lib=ctypes.CDLL(args.library) if args.library else library()
    ptr=ctypes.c_void_p
    lib.set_gelu_table.argtypes=[ptr]
    lib.ffn_gelu.argtypes=[ctypes.c_int]*4+[ptr]*5
    table=ffn[1](torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
    torch.cuda.synchronize()
    assert lib.set_gelu_table(table.data_ptr())==0
    layer=ffn[0]
    m,k=x.shape
    n=layer.out_features
    variants=[]
    for bn in (0,128,256):
        def run():
            if bn:
                y=torch.empty((m,n),device=x.device,dtype=x.dtype)
                status=lib.ffn_gelu(bn,m,n,k,x.data_ptr(),layer.weight.data_ptr(),layer.bias.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
                assert status==0,status
            else:y=ffn[1](layer(x))
            return y,ffn[2](y)
        variants.append(capture(run))
    times=[[] for _ in variants]
    for r in range(24):
        for j in range(3):
            i=(r+j)%3
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(10):variants[i][0].replay()
            b.record();b.synchronize()
            if r>=4:times[i].append(a.elapsed_time(b)/10)
    ms=list(map(statistics.median,times));rows=[]
    for i,bn in enumerate((0,128,256)):
        errors=[]
        for ref,candidate in zip(variants[0][1],variants[i][1]):
            delta=candidate.float()-ref.float()
            errors.append(dict(mismatch=(delta!=0).sum().item(),max_abs=delta.abs().max().item(),
                               nmse=(delta.square().sum()/ref.float().square().sum()).item()))
        row=dict(bn=bn,ms=ms[i],speedup=ms[0]/ms[i],errors=errors)
        print(row,flush=True);rows.append(dict(**row,samples_ms=times[i]))
    path=ROOT/'artifacts/video_runtime_v1/ffn_rounded_gelu_v1'
    path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
