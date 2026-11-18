"""Compare selected-page fetch against the deployed base on paired videos."""
import argparse,ctypes,json,time
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime
from local_experiments.video_runtime_v1.weights import ROOT
from local_experiments.video_runtime_v1.validate_single_av import PROMPTS
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk

@torch.no_grad()
def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--multi',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    installed=rt.packed_attention.kernel
    reference=ctypes.CDLL("/tmp/jzk_delivery_control.so").run
    reference.argtypes=installed.argtypes
    alternative=installed if args.name=='installed' else ctypes.CDLL('/tmp/jzk_delivery_'+args.name+'.so').run
    alternative.argtypes=reference.argtypes
    cases=[(p,s) for s in (703,704) for p in PROMPTS] if args.multi else [('city_walk',701)]*8
    embeddings={p:torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{p}.pt',weights_only=True).cuda() for p in dict(cases)}
    dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float() if args.name in ('fused_center','anchor_center','floor_center') and not args.multi else None
    records=[]
    try:
        for i in range(-1,len(cases)):
            prompt,seed=cases[max(0,i)]
            outputs={}
            row=dict(prompt=prompt,seed=seed)
            for j in ((1,0) if i%2 else (0,1)):
                rt.packed_attention.kernel=(reference,alternative)[j]
                torch.cuda.synchronize()
                start=time.perf_counter()
                rt.reset()
                z=torch.cat([generate_chunk(engine,embeddings[prompt],seed,chunk) for chunk in range(9)],dim=1)
                torch.cuda.synchronize()
                row[('baseline_s','candidate_s')[j]]=time.perf_counter()-start
                outputs[j]=z.cpu()
                assert rt.packed_attention.calls==812 and rt.packed_attention.direct_commits==261
            row['bitwise']=torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8))
            if args.name in ('fused_center','anchor_center','floor_center'):
                if i==0 and not args.multi:
                    torch.save(outputs,f'/tmp/jzk_{args.name}_pair.pt')
                old,new=outputs[0].float(),outputs[1].float()
                row['nmse']=float((new-old).square().sum()/old.square().sum())
                if dense is not None:
                    row['old_dense_nmse']=float((old-dense).square().sum()/dense.square().sum())
                    row['new_dense_nmse']=float((new-dense).square().sum()/dense.square().sum())
            else:
                assert row['bitwise']
            if i>=0:
                records.append(row)
                path=Path(__file__).parent/(args.name+('_multi' if args.multi else '_video')+'.json')
                path.write_text(json.dumps(records,indent=2))
                print(i,1000*(row['baseline_s']-row['candidate_s']),flush=True)
    finally:
        rt.close()
if __name__=='__main__':run()
