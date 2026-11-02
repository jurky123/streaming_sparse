"""Interleaved full-head timing to expose GPU clock/load variability."""
import json
from pathlib import Path
import statistics
import time
import torch
from .cache_grouping import balanced_order,history_order,reorder
from .cache_sparse_prototype import sparse_forward


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/cache_research_20261029');folder=root/'capture_fullhead/qkv'
    hist=torch.load(folder/'c20_l14_history.pt',weights_only=True);cur=torch.load(folder/'c20_l14_s2.pt',weights_only=True)
    kh,vh,kc,vc,q=[x.cuda() for x in [hist['k'],hist['v'],cur['k'],cur['v'],cur['q']]]
    def blhd(x):return x.transpose(0,1).unsqueeze(0).contiguous()
    qb=blhd(q);kb=blhd(torch.cat((kh,kc),1));vb=blhd(torch.cat((vh,vc),1))
    funcs={'dense':lambda:torch.nn.functional.scaled_dot_product_attention(qb.transpose(1,2),kb.transpose(1,2),vb.transpose(1,2)),
           'spatial':lambda:sparse_forward(qb,kb,vb,None,False)}
    for name,weight in [('k',0.),('kv1',1.)]:
        order=history_order(kh,vh,weight)
        kp=blhd(torch.cat((reorder(kh,order),kc),1));vp=blhd(torch.cat((reorder(vh,order),vc),1))
        funcs[name+'_preprocess']=lambda w=weight:balanced_order(kh[:,:4608],vh[:,:4608],w)
        for tail in [False,True]:
            suffix='_tail' if tail else ''
            funcs[name+suffix]=lambda o=order,t=tail:sparse_forward(qb,kb,vb,o,t)
            funcs[name+suffix+'_preordered']=lambda k=kp,v=vp,t=tail:sparse_forward(qb,k,v,None,t)
    for _ in range(3):
        for fn in funcs.values():fn()
    names=list(funcs);records={k:[] for k in names}
    for trial in range(24):
        for name in names[trial%len(names):]+names[:trial%len(names)]:
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            tick=time.perf_counter();start.record();funcs[name]();end.record();torch.cuda.synchronize()
            records[name].append(dict(wall_ms=(time.perf_counter()-tick)*1000,cuda_ms=start.elapsed_time(end),trial=trial))
    stats={name:{key:statistics.median(r[key] for r in rows) for key in ['wall_ms','cuda_ms']} for name,rows in records.items()}
    for name,rows in records.items():
        stats[name].update(min_wall_ms=min(r['wall_ms'] for r in rows),max_wall_ms=max(r['wall_ms'] for r in rows))
    result=dict(shape=list(qb.shape),history_tokens=kh.shape[1],repeats=24,records=records,summary=stats,
        note='Real 12-head layer, rotated method order, synchronized wall and CUDA-event time; still shared GPU, not end-to-end speedup.',
        amortization='Full five-chunk history consumes 5 * preprocess_one_chunk / 20 per attention call in steady state.')
    (root/'microbenchmark_interleaved.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
