"""Same-process video acceptance for a fused kernel candidate."""
import argparse
import json
from pathlib import Path
import time

import torch
from local_experiments.video_runtime_v1 import risk_partial_mass
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime
from local_experiments.video_runtime_v1.weights import ROOT
from local_experiments.video_runtime_v1.validate_single_av import PROMPTS
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from measure import generate
import tempfile

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tier',default='base',choices=('base','step','intersection'))
    parser.add_argument('--pairs',type=int,default=8)
    parser.add_argument('--multi-video',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config(args.tier,local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    baseline=risk_partial_mass._partial
    temp=tempfile.TemporaryDirectory()
    specialized=generate(Path(temp.name)/'partial_remainder.py',False)
    class PartialByWindow:
        def __getitem__(self, grid):
            def launch(*args, **options):
                kernel=specialized if args[10]==288 else baseline
                return kernel[grid](*args,**options)
            return launch
    candidate=PartialByWindow()
    graphs=[{},{}]
    cases=([(p,s) for s in (703,704) for p in PROMPTS] if args.multi_video else [('city_walk',701)]*args.pairs)
    emb={p:torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{p}.pt',weights_only=True).cuda() for p in dict(cases)}
    path=Path(__file__).parent/('multi_video' if args.multi_video else 'single_video')/(args.tier+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    try:
        for pair in range(-1,len(cases)):
            p,seed=cases[max(0,pair)]
            row=dict(prompt=p,seed=seed)
            outputs={}
            counts={}
            for idx in ((1,0) if pair%2 else (0,1)):
                risk_partial_mass._partial=(baseline,candidate)[idx]
                rt.batch_graph.entries=graphs[idx]
                torch.cuda.synchronize()
                start=time.perf_counter()
                rt.reset()
                z=torch.cat([generate_chunk(engine,emb[p],seed,c) for c in range(9)],dim=1)
                torch.cuda.synchronize()
                row[('old_s','new_s')[idx]]=time.perf_counter()-start
                outputs[idx]=z.cpu()
                counts[idx]=rt.counters.tolist()
            x,y=outputs[0],outputs[1]
            row.update(bitwise=torch.equal(x.view(torch.uint8),y.view(torch.uint8)),max_abs=float((x.float()-y.float()).abs().max()),counters=counts)
            row['nmse']=float((x.float()-y.float()).square().sum()/x.float().square().sum())
            print(pair,row,flush=True)
            if pair>=0:
                results.append(row)
                path.write_text(json.dumps(results,indent=2))
    finally:
        risk_partial_mass._partial=baseline
        rt.close()
if __name__=='__main__':main()
