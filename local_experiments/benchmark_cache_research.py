"""Observed latency on actual three-head tensors; no full-model extrapolation."""
import json
import math
import argparse
from pathlib import Path
import statistics
import time
import torch
from .cache_grouping import history_order,balanced_order
from .cache_sparse_prototype import sparse_forward


def measure(fn,repeats=12):
    for _ in range(3):fn()
    samples=[]
    for _ in range(repeats):
        torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize()
        samples.append((time.perf_counter()-t)*1000)
    return dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),samples_ms=samples)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--full-heads',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/cache_research_20261029');folder=root/('capture_fullhead/qkv' if a.full_heads else 'capture/sailboat/qkv')
    hist=torch.load(folder/'c20_l14_history.pt',weights_only=True);curr=torch.load(folder/'c20_l14_s2.pt',weights_only=True)
    kh,vh,kc,vc,q=[x.cuda() for x in [hist['k'],hist['v'],curr['k'],curr['v'],curr['q']]]
    def blhd(x):return x.transpose(0,1).unsqueeze(0).contiguous()
    kb=blhd(torch.cat((kh,kc),1));vb=blhd(torch.cat((vh,vc),1));qb=blhd(q)
    # Four dimensions are REQUIRED for CUDA fused SDPA. Three-dimensional HLD
    # inputs silently fall back to the math backend and invalidate speed claims.
    results={'dense':measure(lambda:torch.nn.functional.scaled_dot_product_attention(
        qb.transpose(1,2),kb.transpose(1,2),vb.transpose(1,2)))}
    for name,weight,tail in [('spatial',None,False),('k',0.,False),('kv1',1.,False),('k_tail',0.,True),('kv1_tail',1.,True)]:
        order=None if weight is None else history_order(kh,vh,weight)
        online=measure(lambda:sparse_forward(qb,kb,vb,order,tail))
        pre=None if weight is None else measure(lambda:balanced_order(kh[:,:4608],vh[:,:4608],weight))
        preordered=None
        if order is not None:
            from .cache_grouping import reorder
            kp=blhd(torch.cat((reorder(kh,order),kc),1));vp=blhd(torch.cat((reorder(vh,order),vc),1))
            preordered=measure(lambda:sparse_forward(qb,kp,vp,None,tail))
        saving=results['dense']['median_ms']-online['median_ms']
        results[name]=dict(online=online,online_with_kv_already_reordered=preordered,preprocess_one_clean_chunk=pre,
            amortized_preprocess_per_20_uses_ms=pre['median_ms']/20 if pre else 0,
            amortized_five_chunk_window_preprocess_ms=5*pre['median_ms']/20 if pre else 0,
            # Five chunks are consumed by each attention call, hence five
            # per-chunk preprocessing charges must be amortized, not just one.
            five_chunk_window_break_even_uses=math.ceil(5*pre['median_ms']/saving) if pre and saving>0 else None,
            online_faster_than_dense=saving>0,
            permutation_bytes=0 if order is None else order.numel()*4)
    result=dict(shape=dict(heads=q.shape[0],q_tokens=4608,history_tokens=23040,d=128),results=results,
        limitations=['Shared GPU, Python prototype, synchronized wall time, 12 repeats; not paper speedup.',
          'Actual captured heads; operator timing only, no extrapolation to full model.',
          'Already-reordered variant is an operator-only experiment; video prototype still caches permutations and gathers online.',
          'Dense SDPA is fused; sparse prototype includes Python routing, gather, and transient statistics.'])
    (root/('microbenchmark_fullhead.json' if a.full_heads else 'microbenchmark.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
