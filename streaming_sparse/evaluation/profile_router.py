"""Synchronized per-module latency, including actual KV gathering and attention."""
import argparse
import json
import time
from pathlib import Path
import torch
from ..capture.dataset_io import AttentionDataset
from ..router.model import load_router
from ..router.clustering import soft_kmeans
from ..train.train_router import to_device


def measure(function, device, repeats):
    for _ in range(2):
        function()
    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        sync()
        begin = time.perf_counter()
        function()
        sync()
        times.append((time.perf_counter()-begin)*1000)
    return sum(times)/len(times)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--keep',type=float,default=.3)
    p.add_argument('--repeats',type=int,default=10)
    args=p.parse_args()
    router=load_router(args.checkpoint,args.device).eval()
    dataset=AttentionDataset(args.data)
    s=to_device(dataset[0],args.device)
    k,v,pos=s['k_hist'],s['v_hist'],s['token_positions']
    q=s['q'][:router.config.query_tile]
    state=router.precompute(k,v,pos,s['head_id'],s['layer_id'])
    g,content=router.features(k,v,pos,s['head_id'],s['layer_id'])
    z,empty=router.aggregator(content,state['assignment'],state['hard'],pos)
    route=router.route(q,state,args.keep,relaxation=False)
    mask=(state['hard']@route['hard_mask'])>0
    kc,vc=s['k_current'],s['v_current']
    def sparse():
        selected_k=torch.cat((k[mask],kc))
        selected_v=torch.cat((v[mask],vc))
        return torch.nn.functional.scaled_dot_product_attention(q[None,None],selected_k[None,None],selected_v[None,None])
    def dense():
        return torch.nn.functional.scaled_dot_product_attention(q[None,None],torch.cat((k,kc))[None,None],torch.cat((v,vc))[None,None])
    tasks=dict(feature=lambda:router.features(k,v,pos,s['head_id'],s['layer_id']),
        aggregation=lambda:router.aggregator(content,state['assignment'],state['hard'],pos),
        encoder=lambda:router.encoder(z,empty),
        precompute_total=lambda:router.precompute(k,v,pos,s['head_id'],s['layer_id']),
        online=lambda:router.route(q,state,args.keep,relaxation=False),gather_and_attention=sparse,dense_attention=dense)
    if router.config.grouping=='learned':
        c=router.config
        tasks['cluster']=lambda:soft_kmeans(g,temperature=c.tau_cluster,iterations=c.cluster_iterations,
            target_size=c.target_size,max_blocks=c.max_blocks,max_size_factor=c.max_size_factor)
    result={name:measure(fn,args.device,args.repeats) for name,fn in tasks.items()}
    same_history=[r for r in dataset.records if r['cache_version']==s['cache_version']]
    # On this adapter's supported path current KV length equals full Q length.
    reuse=sum((r.get('total_queries',len(kc))+router.config.query_tile-1)//router.config.query_tile
              for r in same_history)
    result.update(units='ms',history_tokens=len(k),actual_keep=float(mask.float().mean()),
                  full_query_tiles_sharing_history=reuse,
                  amortized_precompute_ms=result['precompute_total']/reuse,
                  note='Per-head per-tile reference PyTorch implementation; includes gather. No speedup claim.')
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
