import ctypes
import json
import statistics
from pathlib import Path

import torch
import triton
from local_experiments.video_runtime_v1 import risk_recheck as repair
from local_experiments.video_runtime_v1.fixed_page_select import select
from local_experiments.video_runtime_v1.profile_route import capture
from local_experiments.video_runtime_v1.risk_partial_mass import _score
from local_experiments.video_runtime_v1.risk_bound import bound
from local_experiments.video_runtime_v1 import risk_partial_mass

lib=ctypes.CDLL('/tmp/jzk_radix_select.so')
lib.select_pages.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*3+[ctypes.c_bool,ctypes.c_void_p]
lib.select_pages.restype=None


def radix(risk,own,threads,flags):
    h,t,p=risk.shape
    ids=torch.empty((h,t,p//10),dtype=torch.int,device=risk.device)
    mask=torch.empty_like(risk,dtype=torch.bool)
    rows=ids.new_empty(h*t)
    count=ids.new_empty(())
    lib.select_pages(*(v.data_ptr() for v in (risk,own,ids,mask,rows,count)),h*t,p,threads,flags,
                     torch.cuda.current_stream().cuda_stream)
    return ids,mask,rows,count


def timing(funcs):
    cases=[capture(f) for f in funcs]
    times=[[] for _ in funcs]
    for rep in range(30):
        for j in range(len(funcs)):
            i=(j+rep)%len(funcs)
            a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            for _ in range(12):
                cases[i][0].replay()
            b.record()
            b.synchronize()
            if rep>=5:times[i].append(a.elapsed_time(b)/12)
    return list(map(statistics.median,times)),[x[1] for x in cases],times


def measure():
    risk_partial_mass.CHANNEL_MAJOR_STATS=True
    results=[]
    for path in sorted(Path('/tmp/jzk_route_mass_inputs').glob('*.pt')):
        data=torch.load(path,weights_only=True)
        q,k,st,theta,own=[data[n].cuda().contiguous() for n in ('q','keys','stats','theta','own')]
        h,t=q.shape[:2]
        p=st.shape[1]
        u=q.float().square().reshape(h,t,16,8,16).sum(-1)/128
        qn=q.float().norm(dim=-1)/128**.5
        c=bound(st)
        def run(threads):
            risk=_score(q,k,u,qn,st,theta,c)
            if threads==0:
                repair.repair(q,k,u,qn,st,theta,c,own,risk)
                return (risk,*select(risk,own))
            ids,mask,rows,count=radix(risk,own,threads,True)
            z=risk.new_empty((h*t,16,4*p))
            norms=risk.new_empty((h*t,16,2))
            repair._qk[(32,triton.cdiv(p*4,64))](q,k,z,rows,count,t,p,32,num_warps=4)
            repair._norm[(32,16)](z,norms,rows,count,p,triton.next_power_of_2(4*p),32,num_warps=8 if p<=256 else 16,enable_fp_fusion=False)
            repair._recheck[(32,triton.cdiv(p,16))](u,qn,z,norms,st,theta,c,risk,rows,count,t,p,32,num_warps=8,enable_fp_fusion=False)
            repair._reselect[(32,)](risk,own,ids,mask,rows,count,p,p//10,triton.next_power_of_2(p),32,num_warps=4)
            return risk,ids,mask
        ms,out,samples=timing([lambda:run(0),lambda:run(64),lambda:run(128)])
        equal=[[torch.equal(a,b) for a,b in zip(out[0],other)] for other in out[1:]]
        tie_tests=[]
        for x in (torch.randint(-4,4,(h,t,p),device=q.device).float(),torch.full((h,t,p),-0.,device=q.device)):
            x[:,:,::3]=0.
            ref=select(x,own)
            tie_tests.append([all(torch.equal(a,b) for a,b in zip(radix(x,own,w,False),ref)) for w in (64,128)])
        assert all(map(all,equal)) and all(map(all,tie_tests))
        results.append(dict(event=path.stem,ms=ms,speedups=[ms[0]/x for x in ms[1:]],equal=equal,ties=tie_tests,samples_ms=samples))
        Path(__file__).with_name('module.json').write_text(json.dumps(results,indent=2))
        print(path.stem,ms,flush=True)

if __name__=='__main__':measure()
