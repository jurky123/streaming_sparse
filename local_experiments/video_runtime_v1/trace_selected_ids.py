"""Find the first changed LUT using the exact same live routing inputs."""
import json
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .weights import ROOT
from .profile_route import capture
from ..broad_common import Engine
from ..rollout_suite import generate_chunk

@torch.no_grad()
def main():
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    model,reader,phi,b,_=load_weights(cfg)
    rt=make_runtime(engine,model,reader,phi,b,cfg)
    batch=rt.batch_graph
    original=batch.replay
    variants={}
    tested=0
    mismatches=[]
    def trace(e,q,own):
        nonlocal tested
        out=original(e,q,own)
        if mismatches:return out
        key=id(e)
        if key not in variants:
            batch.reuse_selected_ids=True
            variants[key]=capture(lambda:batch.compute(e,False))
            batch.reuse_selected_ids=False
        graph,alt=variants[key]
        graph.replay()
        ids,mask,risk,stats=out
        diff=(ids!=alt[0]).any(-1)
        tested+=1
        if diff.any():
            h,t=diff.nonzero()[0].tolist()
            ranks=risk[h,t].clone();ranks[own[h,t]]=torch.inf
            original_ids=ids[h,t].long()
            new_ids=alt[0][h,t].long()
            mismatches.append(dict(layer=engine.current['layer'],chunk=engine.current['chunk'],step=engine.current['step'],
                                   differing_rows=int(diff.sum()),risk_equal=torch.equal(risk,alt[3]),
                                   mask_equal=torch.equal(mask,alt[2]),head=h,tile=t,
                                   old_ids=original_ids.tolist(),new_ids=new_ids.tolist(),
                                   old_risk=ranks[original_ids].tolist(),new_risk=ranks[new_ids].tolist()))
        return out
    batch.replay=trace
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    for c in range(9):
        generate_chunk(engine,emb,701,c)
        if mismatches:break
    report=dict(calls=tested,mismatches=mismatches)
    (ROOT/'artifacts/video_runtime_v1/reuse_selected_ids_v1/trace.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)
    rt.close()
if __name__=='__main__':main()
