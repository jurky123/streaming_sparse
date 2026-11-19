"""Single-video experiment: pin the first frame and attend exactly to the current chunk."""
import argparse
from pathlib import Path
from types import MethodType

import torch
from local_experiments import rollout_suite
from local_experiments.broad_common import Engine
from local_experiments.stream_eval_common import decode, native_vae, save_video
from .accepted_tiers import config
from .deployment import configure, load_weights, make_runtime

FRAME = 24  # pages of 64 tokens
CHUNK = 3 * FRAME


def commit(store, k, v, summary, order, start):
    # Page slots 0..23 hold the clean first latent frame throughout generation.
    capacity = store.config.history_pages
    slots = torch.arange(CHUNK, device=k.device)
    if start:
        slots = FRAME + (slots + store.write - FRAME) % (capacity - FRAME)
    tokens = (slots[:, None]*64 + torch.arange(64, device=k.device)).flatten()
    for key, values in [('k', k), ('v', v)]:
        store.cache[key][0].index_copy_(0, tokens, values.flatten(1, 2).transpose(0, 1))
    for key in ('keys', 'mean', 'detail', 'stats'):
        getattr(store.summary, key).index_copy_(1, slots, getattr(summary, key))
    store.token_ids.index_copy_(1, slots, (order+start).reshape(12, CHUNK, 64))
    store.write = FRAME + (store.write + CHUNK - FRAME) % (capacity - FRAME)
    store.n = min(capacity, store.n+CHUNK)
    store.commits += 1


def install(rt, mode, fixed_percent=10):
    engine = rt.engine
    anchor_only = mode in ("sink", "anchor-extra", "anchor-fixed")
    # Native layer 0: sink + last history frame + three current frames.
    layer0 = engine.wrapper.model.blocks[0].self_attn
    if mode != 'current':
        layer0.sink_size = 1
    if mode == 'combined':
        layer0.max_attention_size = 5*1536
        for key in ('k','v'):
            engine.cache[0][key] = engine.cache[0][key][:,:5*1536]
    if mode != 'current':
        for store in rt.stores.values():
            store.commit = MethodType(commit,store)
        original_commit = rt.commit_group
        def frame_commit(layer,model,k,v,order,start,enabled):
            # Preserve the learned ordering within each frame, then rebuild summaries.
            if anchor_only and start:
                return original_commit(layer,model,k,v,order,start,enabled)
            assert not enabled
            perm = torch.argsort(order.reshape(12,-1)//1536, dim=1, stable=True)
            gather = perm[...,None].expand(-1,-1,128)
            original_commit(layer,model,k.flatten(1,2).gather(1,gather).reshape_as(k),
                            v.flatten(1,2).gather(1,gather).reshape_as(v),
                            order.reshape(12,-1).gather(1,perm),start,enabled)
        rt.commit_group = frame_commit
    original_replay = rt.batch_graph.replay
    checked = set()
    def exact_current(*args):
        ids, mask, risk, stats = original_replay(*args)
        store = rt.stores[engine.current['layer']]
        p = store.n
        frame = torch.arange(FRAME,device=risk.device)
        page = torch.arange(CHUNK, device=risk.device)
        own = p + page[:, None]
        cur = p + (page[:, None] + page) % CHUNK
        scores = risk.clone()
        parts = []
        if anchor_only:
            parts.append(own)
            scores.scatter_(2, own.expand(12, -1, -1), -torch.inf)
        else:
            parts.append(cur)
            scores = scores[:, :, :p]
        if mode != 'current':
            parts.append(frame.expand(CHUNK, -1))
            scores[:, :, frame] = -torch.inf
        if mode == 'combined':
            recent = FRAME + (store.write-2*FRAME+frame) % (store.config.history_pages-FRAME)
            parts.append(recent.expand(CHUNK, -1))
            scores[:, :, recent] = -torch.inf
        forced = torch.cat(parts, dim=1)
        if mode == 'anchor-extra':
            extra = (p + CHUNK) // 10 - 1
        elif mode == 'anchor-fixed':
            # Reserve all 24 sink pages and the query's own page at P216.
            extra = max(0, (p + CHUNK) * fixed_percent // 100 - FRAME - 1)
        elif mode == 'sink':
            # Own page is already part of the ordinary base routing budget.
            extra = (p + CHUNK - FRAME) // 10 - 1
        else:
            extra = (p + CHUNK - forced.shape[1]) // 10
        selected = scores.topk(extra, dim=-1).indices
        ids = torch.cat([forced.expand(12,-1,-1),selected],dim=-1).int()
        if p not in checked:
            ordered = ids.sort(dim=-1).values
            assert (ordered.diff() > 0).all() and ordered.min() >= 0 and ordered.max() < p + CHUNK
            assert torch.equal(ids[..., :1], own.expand(12, -1, -1))
            if anchor_only:
                assert torch.equal(ids[..., 1:1+FRAME],frame.expand_as(ids[..., 1:1+FRAME]))
            print(f'{mode}: {ids.shape[-1]} exact pages / {p+CHUNK}', flush=True)
            checked.add(p)
        return ids,mask,risk,stats  # base fused attention consumes only the page IDs
    rt.batch_graph.replay = exact_current


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('inputs',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--mode',choices=('combined','sink','current','anchor-extra','anchor-fixed'),default='combined')
    ap.add_argument('--fixed-percent', type=int, choices=(10, 15), default=10,
                    help='Total exact-page percentage for anchor-fixed, including the anchor.')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('base',first_frame_fixed15=False,local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    data={k:v.cuda() for k,v in torch.load(args.inputs,weights_only=True).items()}
    def saved_noise(seed,chunk,step):
        return (data['initial_noise'][:,3*chunk:3*chunk+3] if step==0
                else data['transition_noise'][chunk,step-1])
    original_noise=rollout_suite.noise
    try:
        install(rt, args.mode, args.fixed_percent)
        rollout_suite.noise=saved_noise
        rt.reset()
        zs=[]
        for chunk in range(41):
            zs.append(rollout_suite.generate_chunk(engine,data['prompt_embeds'],2027,chunk))
            for store in (rt.stores.values() if args.mode != "current" else ()):
                sink=store.token_ids[:,:FRAME].flatten(1)
                assert torch.equal(sink.sort(1).values,torch.arange(1536,device=sink.device).expand_as(sink))
            if chunk%10==0: print('completed chunk',chunk,flush=True)
        z=torch.cat(zs,dim=1)
        assert torch.isfinite(z).all()
    finally:
        rollout_suite.noise=original_noise
        rt.close()
    save_video(args.output,decode(native_vae(),z,480))

if __name__=='__main__':
    main()
