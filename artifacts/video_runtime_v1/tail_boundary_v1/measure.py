import json,statistics,sys,os,importlib
from pathlib import Path
import torch,triton
sys.path.insert(0,os.getcwd())
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.profile_route import capture
from local_experiments.video_runtime_v1.weights import ROOT
mode=sys.argv[1]
src=Path(_tail.fn.__code__.co_filename).read_text()
src=src[:src.index('@tr.jit\ndef _merge')].replace('from ..','from local_experiments.')
from boundary import variant_for
variant=variant_for(src,mode)
root=ROOT/'artifacts/video_runtime_v1'
data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
q,k,v=(data[key].cuda().contiguous() for key in ('q','k','v'))
keys,mean,det=(data['summary'][key].cuda().contiguous() for key in ('keys','mean','detail'))
basis=data['basis'].cuda().bfloat16().contiguous()
rows=[]
for p in (432,360,288,216):
    count=p//10
    lut=(data['ids'][...,:count].cuda()%p).contiguous()
    bits=None
    if mode in ('readmask','wordpair'):
        from local_experiments.high_sparse_runtime import _budget_mask
        bits=torch.empty(12,72,16,dtype=torch.uint32,device=q.device)
        _budget_mask[(72,12)](lut,bits,72,count,triton.next_power_of_2(count),num_warps=4)
    scratch=torch.empty_like(q,dtype=torch.float32) if mode.startswith("spill") else None
    variants=[]
    for ker in (_tail,variant):
        out=torch.empty_like(q)
        def run():
            return ker[(72,12)](q,keys,mean,det,bits,scratch,None,None,p,4608,12,128,HEAD_PAGES=432,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=lut,EXACT_COUNT=count,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
        comp=run()
        variants.append((capture(run)[0],out,dict(regs=comp.n_regs,spills=comp.n_spills,shared=comp.metadata.shared)))
    for _ in range(400):variants[0][0].replay()
    torch.cuda.synchronize()
    times=[[],[]]
    for r in range(18):
        for i in (r%2,1-r%2):
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
            for _ in range(8):variants[i][0].replay()
            b.record();b.synchronize()
            if r>=2:times[i].append(a.elapsed_time(b)/8)
    ms=list(map(statistics.median,times))
    row=dict(pages=p,ms=ms,speedup=statistics.median(a/b for a,b in zip(*times)),equal=torch.equal(variants[0][1],variants[1][1]),resources=[v[2] for v in variants])
    print(row,flush=True); rows.append(dict(**row,samples_ms=times))
folder=root/'tail_boundary_v1';folder.mkdir(exist_ok=True)
(folder/((sys.argv[2] if len(sys.argv)>2 else mode)+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
