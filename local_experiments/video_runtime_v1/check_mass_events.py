"""Measure fast and rechecked routes on inputs captured from the current deployment."""
import json
import statistics
from pathlib import Path
import torch
from .profile_route import capture
from .weights import ROOT
from .risk_partial_mass import score, select_scores
from .risk_online import bounded_risk
from .tensor_bmm import scaled_route_qk
from . import tensor_bmm
from .fixed_page_select import select
from .selected_id_order import tie_order
from ..review_v1.ops import Summary


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch._dynamo.config.cache_size_limit=64
    tensor_bmm.QK_TILE64=True
    folder=Path('/tmp/jzk_route_mass_inputs')
    rows=[]
    def compute(q,k,st,theta,own,mode):
        p=st.shape[1]
        s=Summary(k,None,None,st)
        if mode==3:
            risk,lut,_=select_scores(q,s,theta,own)
            if p//10<=32:lut=tie_order(risk,lut,p//10)
            return risk,lut
        if mode==0:
            z=scaled_route_qk(q.flatten(1,2),k.flatten(1,2).transpose(1,2)).reshape(12,72,16,p,4)
            risk=bounded_risk(q.float(),z,s,theta)
        else:risk=score(q,s,theta,own if mode in (2,4) else None,cuda_flags=mode==4)
        lut,_=select(risk,own)
        if p//10<=32:lut=tie_order(risk,lut,p//10)
        return risk,lut
    fn=torch.compile(compute,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    for f in sorted(folder.glob('*.pt')):
        data=torch.load(f,weights_only=True)
        args=[data[n].cuda() for n in ('q','keys','stats','theta','own')]
        cases=[capture(lambda:fn(*args,mode)) for mode in (0,1,2,3,4)]
        original=[data[k].cuda() for k in ('risk','ids')]
        p=args[2].shape[1]
        ranked=cases[1][1][0].scatter(-1,args[-1][...,None],float('inf')).topk(p//10+1).values
        flagged=((ranked[:,:,1:-1]-ranked[:,:,2:])<=4.e-5*ranked[:,:,1:-1]).any(-1)
        times=[[] for _ in cases]
        for rep in range(32):
            for j in range(len(cases)):
                i=(j+rep)%len(cases)
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(8):cases[i][0].replay()
                b.record();b.synchronize()
                if rep>=2:times[i].append(a.elapsed_time(b)/8)
        ms=list(map(statistics.median,times))
        row=dict(event=f.stem,flagged=int(flagged.sum()),ms=ms,speedup=ms[0]/ms[4],
                 old_matches_capture=[torch.equal(a,b) for a,b in zip(original,cases[0][1])],
                 restored_flagged_scores=torch.equal(original[0][flagged],cases[4][1][0][flagged]),
                 lut_equal=[torch.equal(original[1],c[1][1]) for c in cases])
        print(row,flush=True)
        rows.append(dict(**row,samples_ms=times))
    assert len(rows)==7 and all(all(r['old_matches_capture']) for r in rows)
    (ROOT/'artifacts/video_runtime_v1/logit_layout_v1/cuda_recheck_events.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
