import json,statistics,sys,os,importlib
from pathlib import Path
import torch,triton
sys.path.insert(0,os.getcwd())
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.profile_route import capture
from local_experiments.video_runtime_v1.weights import ROOT
import subprocess
from shared import _tail as variant
folder=Path(__file__).parent
root=ROOT/'artifacts/video_runtime_v1'
data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
q,k,v=(data[key].cuda().contiguous() for key in ('q','k','v'))
keys,mean,det=(data['summary'][key].cuda().contiguous() for key in ('keys','mean','detail'))
basis=data['basis'].cuda().bfloat16().contiguous()
rows=[]
for p in map(int,sys.argv[2:]):
    count=p//10
    lut=(data['ids'][...,:count].cuda()%p).contiguous()
    variants=[]
    for ker in (_tail,variant):
        out=torch.empty_like(q)
        def run():
            return ker[(72,12)](q,keys,mean,det,None,None,None,None,p,4608,12,128,HEAD_PAGES=432,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=lut,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=count,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
        comp=run()
        if p==432 and sys.argv[1]=='primary':
            path=folder/('baseline' if ker is _tail else 'shared')
            for ext in ('ttgir','ptx'):
                path.with_suffix('.'+ext).write_text(comp.asm[ext])
            path.with_suffix('.cubin').write_bytes(comp.asm['cubin'])
            for option,ext in [('--dump-sass','.sass'),('--dump-resource-usage','.resources')]:
                path.with_suffix(ext).write_bytes(subprocess.check_output(['cuobjdump',option,str(path.with_suffix('.cubin'))]))
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
    ref,actual=[v[1].float() for v in variants]
    delta=actual-ref
    row.update(max_abs=delta.abs().max().item(),nmse=(delta.square().sum()/ref.square().sum()).item(),changed=(ref!=actual).sum().item())
    print(row,flush=True); rows.append(dict(**row,samples_ms=times))
(folder/(sys.argv[1]+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
