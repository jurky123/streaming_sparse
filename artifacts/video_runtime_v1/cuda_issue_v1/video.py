"""Same-process video acceptance for a fused kernel candidate."""
import argparse
import json
from pathlib import Path
import time

import torch
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime
from local_experiments.video_runtime_v1.weights import ROOT
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',type=int,default=8)
    parser.add_argument("--save-pair",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)

    import ctypes
    native=ctypes.CDLL('/tmp/jzk_precarry_release.so').run
    native.argtypes=rt.packed_attention.kernel.argtypes
    alt=ctypes.CDLL('/tmp/jzk_carry.so').run
    alt.argtypes=native.argtypes
    dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float()
    cases=[('city_walk',701)]*args.pairs
    emb={p:torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{p}.pt',weights_only=True).cuda() for p in dict(cases)}
    path=Path(__file__).parent/'release_pairs/base.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    try:
        for pair in range(-1,len(cases)):
            p,seed=cases[max(0,pair)]
            row=dict(prompt=p,seed=seed)
            outputs={}
            counts={}
            for idx in ((1,0) if pair%2 else (0,1)):
                rt.packed_attention.kernel=alt if idx else native
                torch.cuda.synchronize()
                start=time.perf_counter()
                rt.reset()
                z=torch.cat([generate_chunk(engine,emb[p],seed,c) for c in range(9)],dim=1)
                torch.cuda.synchronize()
                row[('old_s','new_s')[idx]]=time.perf_counter()-start
                if rt.packed_attention is not None:
                    assert rt.packed_attention.calls==812
                    assert rt.packed_attention.direct_commits==29*9
                outputs[idx]=z.cpu()
                counts[idx]=rt.counters.tolist()
            x,y=outputs[0],outputs[1]
            row.update(bitwise=torch.equal(x.view(torch.uint8),y.view(torch.uint8)),max_abs=float((x.float()-y.float()).abs().max()),counters=counts)
            assert (p,seed)==('city_walk',701)
            row['old_dense_nmse']=float((x.float()-dense).square().sum()/dense.square().sum())
            row['new_dense_nmse']=float((y.float()-dense).square().sum()/dense.square().sum())
            delta=(x.float()-y.float())
            row.update(nmse=float(delta.square().sum()/x.float().square().sum()),mae=float(delta.abs().mean()),chunk_nmse=[float(d.square().sum()/ref.float().square().sum()) for d,ref in zip(delta.chunk(9,dim=1),x.chunk(9,dim=1))])
            print(pair,row,flush=True)
            if pair>=0:
                results.append(row)
                if pair==0 and args.save_pair:torch.save(outputs,args.save_pair)
                path.write_text(json.dumps(results,indent=2))

    finally:
        rt.close()
if __name__=='__main__':main()
