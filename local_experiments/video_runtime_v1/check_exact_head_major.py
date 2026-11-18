"""Real full-window exact-attention LUT, compare KV layout at fixed arithmetic."""
import json
import statistics
import torch
from .weights import ROOT
from .profile_route import capture
from ..gap_exact_round_kernel import _attn_fwd

@torch.no_grad()
def main():
    folder = ROOT / 'artifacts/video_runtime_v1'
    data = torch.load(folder / 'tail_liveness_v1/input.pt', weights_only=True)
    q, k, v = (data[key].cuda().contiguous() for key in ('q','k','v'))
    ids = data['ids'].cuda().contiguous()[None]
    graphs, values = [], []
    for major, high_precision in ((False, True), (True, True), (False, False)):
        kk, vv = (x.transpose(1,2).contiguous() if major else x for x in (k,v))
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(12,4608,device=q.device)
        def call():
            _attn_fwd[(72,12)](q,kk,vv,128**-.5,43,ids,lse,out,12,4608,k.shape[1],72,
                               128,64,64,HIGH_PRECISION_AV=high_precision,HEAD_MAJOR_KV=major,
                               num_warps=4,num_stages=2)
            return out,lse
        graph, value = capture(call)
        graphs.append((graph, kk, vv))
        values.append(value)
    samples = [[],[],[]]
    for rep in range(32):
        for i in ((rep+j)%3 for j in range(3)):
            a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(10):
                graphs[i][0].replay()
            b.record()
            b.synchronize()
            if rep>=2:
                samples[i].append(a.elapsed_time(b)/10)
    equal=[torch.equal(a,b) for a,b in zip(values[0],values[1])]
    delta=values[2][0]-values[0][0]
    old,new,single=map(statistics.median,samples)
    report=dict(single_ms=single,single_speedup=old/single,single_nmse=(delta.square().sum()/values[0][0].square().sum()).item(),single_max=delta.abs().max().item(),single_lse_equal=torch.equal(values[0][1],values[2][1]),equal=equal,old_ms=old,new_ms=new,speedup=old/new,samples_ms=samples)
    dest=folder/'exact_head_major_v1'
    dest.mkdir(exist_ok=True)
    (dest/'module.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!="samples_ms"})
if __name__=='__main__':
    main()
