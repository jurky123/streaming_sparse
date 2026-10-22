"""Capture only the actual cached self-attention call; no backbone edits."""
import torch
from .dataset_io import CaptureWriter
from ..attention.block_statistics import exact_attention
from ..attention.reconstruction_loss import normalized_attention_mse


class AttentionCapture:
    def __init__(self, output, layers=(4, 14, 24), heads=(0, 3, 6, 9),
                 query_tile=16, tiles_per_event=4):
        self.writer = CaptureWriter(output)
        self.layers, self.heads = set(layers), tuple(heads)
        self.query_tile, self.tiles_per_event = query_tile, tiles_per_event
        self.errors = []

    @torch.no_grad()
    def __call__(self, context, q, k, v, output):
        if context['layer_id'] not in self.layers or context['phase'] != 'denoise':
            return
        nh = context['history_tokens']
        if nh == 0:
            return
        # Each saved Q tile keeps the ENTIRE current KV (not just corresponding K/V).
        starts = list(range(0, q.shape[1], self.query_tile))
        if self.tiles_per_event > 0 and len(starts) > self.tiles_per_event:
            starts = [starts[i] for i in torch.linspace(0, len(starts)-1, self.tiles_per_event).long().tolist()]
        indices = torch.cat([torch.arange(s, min(s + self.query_tile, q.shape[1]), device=q.device) for s in starts])
        for head in self.heads:
            if head >= q.shape[2]:
                raise ValueError(f'Invalid head {head}')
            queries = q[0, indices, head]
            dense = exact_attention(queries, k[0, :, head], v[0, :, head], self.query_tile)
            actual = output[0, indices, head].float()
            error = float(normalized_attention_mse(dense, actual))
            self.errors.append(error)
            meta = {key: context[key] for key in ('prompt_id','layer_id','chunk_id','step_id','timestep','phase')}
            meta.update({key: context[key] for key in ('seed','run_id') if key in context})
            meta.update(head_id=head, history_tokens=nh, query_tile=self.query_tile,
                        total_queries=q.shape[1], captured_queries=len(indices),
                        dense_fp32_vs_backend_error=error)
            history = dict(k_hist=k[0, :nh, head].cpu(), v_hist=v[0, :nh, head].cpu(),
                           token_positions=context['positions'][:nh].cpu())
            event = dict(q=queries.cpu(), k_current=k[0, nh:, head].cpu(), v_current=v[0, nh:, head].cpu(),
                         o_dense=dense.cpu(), o_backend=actual.cpu(),
                         query_positions=context['query_positions'][indices].cpu(),
                         query_indices=indices.cpu(), current_positions=context['positions'][nh:].cpu())
            self.writer.write(history, event, meta)
