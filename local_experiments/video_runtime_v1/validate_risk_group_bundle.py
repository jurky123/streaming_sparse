"""Verify the combined runtime against the same-card original frozen trajectories."""
import argparse
import json
import gc
import torch
from .validate_single_av import PROMPTS, TIERS, CACHE
from .weights import ROOT
from .accepted_tiers import config
from .deployment import configure, load_weights, make_runtime
from ..broad_common import Engine
from ..rollout_suite import generate_chunk

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--overlap-tail',action='store_true')
    parser.add_argument('--tie-direct',action='store_true')
    parser.add_argument('--contiguous-commit',action='store_true')
    parser.add_argument('--ffn-group-commit',action='store_true')
    parser.add_argument('--norm-raw',action='store_true')
    args=parser.parse_args()
    if args.norm_raw:args.ffn_group_commit=True
    if args.ffn_group_commit:args.contiguous_commit=True
    if args.contiguous_commit:args.tie_direct=True
    if args.tie_direct:args.overlap_tail=True
    shard=args.shard
    cases=[(p,s) for p in PROMPTS for s in range(703,707)][shard::8]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    configure(config('base',local_assets='/tmp/streaming_sparse-0/runtime'))
    out=ROOT/'artifacts/video_runtime_v1'/('norm_raw_v1' if args.norm_raw else 'ffn_group_commit_v1' if args.ffn_group_commit else 'contiguous_commit_v1' if args.contiguous_commit else 'tie_direct_v1' if args.tie_direct else 'tail_merge_overlap_v1' if args.overlap_tail else 'risk_group_bundle_v1')/'validation'
    out.mkdir(parents=True,exist_ok=True)
    for tier in TIERS:
        cfg=config(tier,layernorm_mod=args.norm_raw,exact_group_direction='raw' if args.norm_raw else args.ffn_group_commit,rounded_ffn=args.ffn_group_commit,contiguous_commit=args.contiguous_commit,selected_tie_order=args.tie_direct,direct_output=args.tie_direct,inline_risk_projection=True,group_gathers=True,clean_stream_width=4 if args.overlap_tail else 0,tail_merge=args.overlap_tail)
        model,reader,phi,basis,_=load_weights(cfg)
        engine=Engine()
        engine.wrapper.requires_grad_(False)
        rt=make_runtime(engine,model,reader,phi,basis,cfg)
        if args.contiguous_commit:
            for store in rt.stores.values():store.contiguous_commit=True
        if args.tie_direct:
            rt.batch_graph.reuse_selected_ids='ties'
            rt.direct_output=True
        for prompt,seed in cases:
            emb=torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{prompt}.pt',weights_only=True).cuda()
            rt.reset()
            z=torch.cat([generate_chunk(engine,emb,seed,c) for c in range(9)],dim=1).cpu()
            old=torch.load(CACHE/f'{prompt}_{seed}/{tier}_old.pt',weights_only=True)
            delta=z.float()-old.float()
            row=dict(prompt=prompt,seed=seed,tier=tier,gpu=shard,bitwise=torch.equal(z.view(torch.uint8),old.view(torch.uint8)),max_abs=delta.abs().max().item(),nmse=(delta.square().sum()/old.float().square().sum()).item())
            (out/f'{tier}_{prompt}_{seed}.json').write_text(json.dumps(row))
        rt.close()
        del rt,engine,model,reader,phi,basis
        gc.collect()
        torch.cuda.empty_cache()
if __name__=='__main__':main()
