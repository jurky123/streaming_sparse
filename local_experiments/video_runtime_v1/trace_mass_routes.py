"""Compare partial mass and deployed routing on each call of the same latent trajectory."""
import argparse
import json
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .risk_partial_mass import score
from .selected_id_order import tie_order
from .fixed_page_select import select
from .weights import ROOT
from ..broad_common import Engine
from ..rollout_suite import generate_chunk


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--recheck',action='store_true')
    parser.add_argument('--tier',choices=['base','intersection','step'],default='base')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=config(args.tier,rechecked_mass=False,local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    graphs=rt.batch_graph
    replay=graphs.replay
    fast=torch.compile(score,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    records=[]
    thresholds=(1.e-6,4.e-6,1.e-5,4.e-5,1.e-4)

    def audit(e,q,own):
        ids,mask,risk,stats=replay(e,q,own)
        if e['full_pset']:return ids,mask,risk,stats
        new=fast(q,e['s'],e['theta'],own) if args.recheck else fast(q,e['s'],e['theta'])
        lut,newmask=select(new,own)
        k=ids.shape[-1]
        if k<=32:lut=tie_order(new,lut,k)
        wrong=(ids!=lut).any(-1)
        set_change=(mask!=newmask).any(-1)
        ranked=new.scatter(-1,own[...,None],float('inf')).topk(k+1).values
        rel_gap=((ranked[:,:,1:-1]-ranked[:,:,2:])/ranked[:,:,1:-1].clamp_min(1.e-30)).amin(-1)
        rel=(new-risk).abs()/risk.clamp_min(1.e-30)
        counts=[wrong.sum(),set_change.sum()]
        for threshold in thresholds:
            flag=rel_gap<=threshold
            counts.extend((flag.sum(),(wrong&~flag).sum()))
        if args.capture and engine.current['layer']==15 and engine.current['step']==0:
            from pathlib import Path
            folder=Path('/tmp/jzk_route_mass_inputs')
            folder.mkdir(exist_ok=True)
            p=risk.shape[-1]
            file=folder/f'p{p}_c{engine.current["chunk"]}.pt'
            torch.save(dict(q=q.cpu(),keys=e['s'].keys.cpu(),stats=e['s'].stats.cpu(),theta=e['theta'].cpu(),
                            own=own.cpu(),risk=risk.cpu(),ids=ids.cpu()),file)
        records.append((k,torch.stack(counts),rel.max(),(new-risk).abs().max()))
        return ids,mask,risk,stats

    graphs.replay=audit
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    for c in range(9):generate_chunk(engine,emb,701,c)
    rt.close()
    rows=[]
    for k in sorted({r[0] for r in records}):
        group=[r for r in records if r[0]==k]
        counts=torch.stack([r[1] for r in group]).sum(0).tolist()
        rows.append(dict(k=k,events=len(group),row_count=len(group)*864,wrong_order=counts[0],changed_sets=counts[1],
                         max_relative=float(torch.stack([r[2] for r in group]).max()),
                         max_abs=float(torch.stack([r[3] for r in group]).max()),
                         thresholds=[dict(gap=t,flagged=f,missed=m) for t,f,m in zip(thresholds,counts[2::2],counts[3::2])]))
    dest=ROOT/'artifacts/video_runtime_v1/partial_mass_video_v1'
    dest.mkdir(exist_ok=True)
    (dest/(f'recheck_{args.tier}.json' if args.recheck else 'trace.json' if args.tier=='base' else f'trace_{args.tier}.json')).write_text(json.dumps(rows,indent=2))
    print(rows,flush=True)


if __name__=='__main__':
    main()
