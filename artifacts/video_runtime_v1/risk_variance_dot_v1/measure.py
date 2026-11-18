import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
from variance_dot import _partial as dot
from qk_tile import _partial as unrolled
from qk_loop import _partial as looped
from local_experiments.video_runtime_v1.risk_partial_mass import _partial, _finish
from local_experiments.video_runtime_v1.risk_bound import bound
from local_experiments.video_runtime_v1.profile_route import capture


def measure(experiment):
    original = (_partial,128,4,3,())
    cases_by_name = {
        'variance': ('module.json', [original, (_partial,64,4,3,()), (_partial,128,8,3,()),
                                     (dot,64,4,3,()), (dot,128,4,3,()), (dot,128,8,3,())]),
        'unroll': ('qk_module.json', [original] + [(unrolled,128,4,3,(bk,fast))
                                                   for bk in (16,32,64,128) for fast in (False,True)]),
        'pipeline': ('qk_loop.json', [original] + [(looped,128,4,stages,(bk,False))
                                                   for bk in (16,32,64) for stages in (1,2,3,4)
                                                   if bk!=64 or stages<=2]),
        'layout': ('integrated_layout.json', [original, (_partial,128,4,3,(True,))]),
    }
    output, configs = cases_by_name[experiment]
    rows = []
    for path in sorted(Path('/tmp/jzk_route_mass_inputs').glob('*.pt')):
        event = torch.load(path, weights_only=True)
        q, k, st, theta = [event[key].cuda().contiguous() for key in ('q','keys','stats','theta')]
        h,t = q.shape[:2]
        p = st.shape[1]
        u = q.float().square().reshape(h,t,16,8,16).sum(-1)/128
        norm = q.float().norm(dim=-1)/(128**.5)
        c = bound(st).contiguous()
        def run(kernel,width,warps,stages,extra):
            blocks = triton.cdiv(p,width)
            v = u.new_empty((h*t,blocks,16,width))
            n = u.new_empty((h*t,blocks,16,2))
            out = u.new_empty((h,t,p))
            stats = st.flatten(1,2).transpose(1,2).contiguous() if kernel is _partial and extra==(True,) else st
            kernel[(h*t,blocks)](q,k,u,norm,stats,theta,c,v,n,t,p,width,*extra,num_stages=stages,num_warps=warps,enable_fp_fusion=False)
            _finish[(h*t,blocks)](v,n,out,p,width,triton.next_power_of_2(blocks),num_warps=4,enable_fp_fusion=False)
            return out
        cases = [capture(lambda: run(*cfg)) for cfg in configs]
        reference = cases[0][1]
        own = event['own'].cuda()
        def ids(x):
            return x.scatter(-1,own[...,None],float('inf')).topk(p//10).indices
        reference_ids = ids(reference)
        timings = [[] for _ in cases]
        for rep in range(27):
            for j in range(len(cases)):
                i = (j+rep)%len(cases)
                a,b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                for _ in range(12):
                    cases[i][0].replay()
                b.record()
                b.synchronize()
                if rep>=3:
                    timings[i].append(a.elapsed_time(b)/12)
        row = dict(event=path.stem, ms=list(map(statistics.median,timings)), max_rel=[float(((out-reference).abs()/reference.abs().clamp_min(1.e-12)).max()) for _,out in cases], ordered_id_differences=[int((ids(out)!=reference_ids).sum()) for _,out in cases],samples_ms=timings)
        rows.append(row)
        print(path.stem,row['ms'],flush=True)
        Path(__file__).with_name(output).write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment',choices=['variance','unroll','pipeline','layout'])
    measure(parser.parse_args().experiment)
