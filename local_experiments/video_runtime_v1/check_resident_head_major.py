"""Time both physical resident KV layouts with real prefix strides and the shared dense path."""
import json
import statistics
import torch
from .weights import ROOT
from .profile_route import capture
from .attention import Workspace
from .legacy_components import attention
from ..review_v1.ops import Summary

@torch.no_grad()
def main():
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
    q,k,v=(data[key].cuda().contiguous() for key in ('q','k','v'))
    packed=[y.transpose(1,2).contiguous().transpose(1,2) for y in (k,v)]
    basis=data['basis'].cuda()
    rows=[]
    for p in (216,288,360,432):
        ids=(data['ids'][...,:p//10]%p).cuda().contiguous()
        summary=Summary(*(data['summary'][name][:,:p].cuda().contiguous() for name in ('keys','mean','detail','stats')))
        for mode in ('sparse','dense'):
            graphs=[]
            for kk,vv in ((k,v),packed):
                ws=Workspace();ws.tail_merge=True
                kk,vv=kk[:,:p*64],vv[:,:p*64]
                def run():
                    if mode=='sparse':return attention(q,kk,vv,summary,basis,ids,ws)
                    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),kk.transpose(1,2),vv.transpose(1,2))
                graphs.append((*capture(run),ws))
            times=[[],[]]
            for r in range(16):
                for i in (r%2,1-r%2):
                    a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                    a.record()
                    for _ in range(6):graphs[i][0].replay()
                    b.record();b.synchronize()
                    if r>=2:times[i].append(a.elapsed_time(b)/6)
            x,y=[g[1] for g in graphs]
            ms=list(map(statistics.median,times))
            row=dict(pages=p,mode=mode,ms=ms,speedup=ms[0]/ms[1],equal=torch.equal(x,y),max_abs=(x.float()-y.float()).abs().max().item())
            rows.append(dict(**row,samples_ms=times));print(row,flush=True)
    path=root/'resident_head_major_v1';path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
