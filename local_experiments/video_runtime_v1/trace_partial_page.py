"""Locate summary fields affected by combining the four representative moments."""
import argparse
import json
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .summary_moments import build
from .weights import ROOT
from ..broad_common import Engine
from ..rollout_suite import generate_chunk

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--feature-page',action='store_true')
    ap.add_argument('--cuda-pool',action='store_true')
    args=ap.parse_args()
    cfg=config('intersection',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    records={}
    first={}
    def summary(m,k,v,cheap=False):
        old=build(m,k,v,cheap)
        alt=build(m,k,v,cheap,cuda_pool=args.cuda_pool,partial_page=not (args.feature_page or args.cuda_pool),feature_page=args.feature_page)
        for name in ('keys','mean','detail','stats'):
            a,b=getattr(old,name),getattr(alt,name)
            delta=(a.float()-b.float()).abs()
            counts=(delta!=0).reshape(-1,a.shape[-1]).sum(0)
            key=f'{int(cheap)}_{name}'
            if key not in records:records[key]=torch.zeros_like(counts)
            records[key]+=counts
            if key not in first and counts.any():
                pos=(delta!=0).nonzero()[0].tolist()
                first[key]=dict(chunk=engine.current['chunk'],layer=engine.current['layer'],step=engine.current['step'],
                                index=pos,old=float(a[tuple(pos)]),new=float(b[tuple(pos)]),max_abs=float(delta.max()))
        return old
    rt.summary_graph=summary
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    for c in range(9):
        generate_chunk(engine,emb,701,c)
    rt.close()
    out=dict(first=first,counts={k:v.cpu().tolist() for k,v in records.items()})
    (ROOT/'artifacts/video_runtime_v1'/('cuda_pool_v1' if args.cuda_pool else 'feature_page_v1' if args.feature_page else 'partial_page_v1')/'trace.json').write_text(json.dumps(out,indent=2))
    print(first,flush=True)
if __name__=='__main__':main()
