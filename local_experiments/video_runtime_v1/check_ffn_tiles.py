"""Full two-linear FFN timing, measuring working-set reduction and output bytes."""
import json
import statistics
import torch
from .weights import ROOT
from .accepted_tiers import config
from .deployment import configure
from .profile_route import capture
from ..broad_common import Engine

@torch.no_grad()
def main():
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    torch.manual_seed(701)
    torch.set_num_threads(4)
    engine=Engine()
    block=engine.wrapper.model.blocks[15]
    net=block.ffn
    x=torch.randn((4608,1536),device='cuda',dtype=net[0].weight.dtype)
    variants=[]
    for tile in (4608,1536,768,0):
        def run():
            if not tile:
                layer=net[0]
                y=torch.ops.aten._addmm_activation(layer.bias,x,layer.weight.T,use_gelu=True)
                return net[2](y)
            if tile==4608:return net(x)
            return torch.cat([net(part) for part in x.split(tile)])
        graph,out=capture(run)
        variants.append((graph,out))
    samples=[[] for _ in variants]
    for r in range(24):
        for j in range(4):
            i=(j+r)%4
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(10):variants[i][0].replay()
            b.record();b.synchronize()
            if r>=4:samples[i].append(a.elapsed_time(b)/10)
    rows=[];ref=variants[0][1].float()
    for tile,(_,out),ms in zip((4608,1536,768,0),variants,samples):
        delta=out.float()-ref
        row=dict(tile=tile,ms=statistics.median(ms),max_abs=delta.abs().max().item(),
                 nmse=(delta.square().sum()/ref.square().sum()).item(),mismatch=(delta!=0).sum().item())
        print(row,flush=True);rows.append(dict(**row,samples_ms=ms))
    path=ROOT/'artifacts/video_runtime_v1/ffn_tiles_v1';path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
