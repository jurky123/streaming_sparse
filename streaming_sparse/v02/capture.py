"""Versioned capture adapter with deterministic spatiotemporal stratified Q tiles."""
import argparse
import hashlib
import math
import sys
import torch
from ..capture.hook_attention import AttentionCapture as V01Capture
from ..attention.block_statistics import exact_attention
from ..attention.reconstruction_loss import normalized_attention_mse


class StratifiedCapture(V01Capture):
    sampling_seed=2026

    @torch.no_grad()
    def __call__(self,context,q,k,v,output):
        if context['layer_id'] not in self.layers or context['phase']!='denoise' or not context['history_tokens']:
            return
        tile=self.query_tile
        positions=context['query_positions'].cpu()
        starts=torch.arange(0,q.shape[1]-tile+1,tile)
        centers=positions[starts+tile//2]
        height=int(positions[:,1].max())+1
        width=int(positions[:,2].max())+1
        buckets={}
        for start,pos in zip(starts.tolist(),centers.tolist()):
            cell=(pos[0],min(1,pos[1]*2//height),min(1,pos[2]*2//width))
            buckets.setdefault(cell,[]).append(start)
        # No step/layer/head in the seed: corresponding snapshot events use the
        # same positions, and all heads/layers observe consistent spatial queries.
        key=(context['prompt_id'],context.get('seed'),context['chunk_id'],self.sampling_seed)
        seed=int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator().manual_seed(seed)
        times=sorted({cell[0] for cell in buckets})
        cells={t:[c for c in sorted(buckets) if c[0]==t] for t in times}
        for t in times:
            order=torch.randperm(len(cells[t]),generator=generator).tolist()
            cells[t]=[cells[t][i] for i in order]
        selected=[]
        count=min(len(starts),self.tiles_per_event) if self.tiles_per_event>0 else len(starts)
        for round_id in range(math.ceil(count/len(times))):
            for t in times:
                cell=cells[t][round_id%len(cells[t])]
                candidates=[s for s in buckets[cell] if s not in selected]
                if not candidates:
                    candidates=[s for s in starts.tolist() if s not in selected]
                index=int(torch.randint(len(candidates),(1,),generator=generator))
                selected.append(candidates[index])
                if len(selected)==count:
                    break
            if len(selected)==count:
                break
        selected.sort()
        indices=torch.cat([torch.arange(s,s+tile,device=q.device) for s in selected])
        nh=context['history_tokens']
        for head in self.heads:
            queries=q[0,indices,head]
            dense=exact_attention(queries,k[0,:,head],v[0,:,head],tile)
            actual=output[0,indices,head].float()
            error=float(normalized_attention_mse(dense,actual))
            self.errors.append(error)
            meta={key:context[key] for key in ('prompt_id','layer_id','chunk_id','step_id','timestep','phase','seed','run_id')}
            meta.update(head_id=head,history_tokens=nh,query_tile=tile,total_queries=q.shape[1],
                captured_queries=len(indices),sampling='temporal_round_robin_spatial_2x2',
                sampling_seed=self.sampling_seed,query_tile_starts=selected,
                dense_fp32_vs_backend_error=error)
            history=dict(k_hist=k[0,:nh,head].cpu(),v_hist=v[0,:nh,head].cpu(),token_positions=context['positions'][:nh].cpu())
            event=dict(q=queries.cpu(),k_current=k[0,nh:,head].cpu(),v_current=v[0,nh:,head].cpu(),
                o_dense=dense.cpu(),o_backend=actual.cpu(),query_positions=context['query_positions'][indices].cpu(),
                query_indices=indices.cpu(),current_positions=context['positions'][nh:].cpu())
            self.writer.write(history,event,meta)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sampling-seed',type=int,default=2026)
    known,remaining=parser.parse_known_args()
    StratifiedCapture.sampling_seed=known.sampling_seed
    from ..capture import hook_attention,capture_runner
    original=hook_attention.AttentionCapture
    argv=sys.argv
    try:
        hook_attention.AttentionCapture=StratifiedCapture
        sys.argv=[argv[0],*remaining]
        capture_runner.main()
    finally:
        hook_attention.AttentionCapture=original
        sys.argv=argv


if __name__=='__main__':
    main()
