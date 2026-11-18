"""Three BF16 projections versus a packed GEMM, including contiguous output materialization."""
import json
import statistics
import torch
from torch.nn import functional as F
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from .weights import ROOT
from .deployment import configure
from .accepted_tiers import config
from .profile_route import capture


@torch.no_grad()
def main():
    configure(config('base',local_assets='/tmp/streaming_sparse-0/runtime'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine()
    attn=engine.wrapper.model.blocks[15].self_attn
    inputs=[]
    def keep(_,args):
        if not inputs:inputs.append(args[0].detach().clone())
    hook=attn.q.register_forward_pre_hook(keep)
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    generate_chunk(engine,emb,701,0)
    hook.remove()
    x=inputs[0]
    layers=(attn.q,attn.k,attn.v)
    w=torch.cat([layer.weight for layer in layers])
    b=torch.cat([layer.bias for layer in layers])
    def separate():
        return tuple(layer(x) for layer in layers)
    def packed():
        return tuple(y.contiguous() for y in F.linear(x,w,b).chunk(3,-1))
    graphs=[capture(fn) for fn in (separate,packed)]
    samples=[[],[]]
    for rep in range(24):
        for idx in (rep%2,1-rep%2):
            a,btime=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(10):graphs[idx][0].replay()
            btime.record();btime.synchronize()
            if rep>=4:samples[idx].append(a.elapsed_time(btime)/10)
    ms=list(map(statistics.median,samples))
    outputs=[]
    for a,b in zip(graphs[0][1],graphs[1][1]):
        delta=b.float()-a.float()
        outputs.append(dict(equal=torch.equal(a,b),max_abs=delta.abs().max().item(),
                            nmse=(delta.square().sum()/a.float().square().sum()).item()))
    row=dict(input_shape=list(x.shape),ms=ms,speedup=ms[0]/ms[1],outputs=outputs,samples_ms=samples)
    path=ROOT/'artifacts/video_runtime_v1/packed_qkv_v1'
    path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(row,indent=2))
    print(ms,outputs,flush=True)

if __name__=='__main__':main()
