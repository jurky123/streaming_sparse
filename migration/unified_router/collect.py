"""Format3 raw Q/H-KV/current-KV capture; no true labels in online sketches."""
import argparse
import os
from pathlib import Path
import time
import torch

from streaming_sparse.v08.replay import ReplayDataset
from streaming_sparse.v08.backbone import FunctionalBackbone, load_teacher
from streaming_sparse.capture.capture_runner import sdpa_backend
from .common import config, event_key, history_key, save, atomic_json, paired_seed, update, sha


def cpu(x):
    return x.detach().cpu().contiguous().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    root, m = config(args.directory)
    paired_seed(0)
    torch.cuda.set_device(0)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    rows = [r for r in m['rows'] if not args.pilot or r['sample_id'] in m['pilot_sample_ids']]
    trajectories = sorted({r['trajectory_id'] for r in rows})
    mine = set(trajectories[rank::world])
    rows = [r for r in rows if r['trajectory_id'] in mine]
    cache = Path(m['cache'])
    phase = 'pilot_capture' if args.pilot else 'capture'
    datasets = {s: ReplayDataset(s, require_complete=True) for s in ['train', 'val']}
    row = None
    start_frame = None

    class Capture(FunctionalBackbone):
        def _block(self, *a, **kw):
            block, layer, x, e0, grid, freqs = a[:6]
            hk, hv, start = a[7:10]
            if start == start_frame and layer in m['layers']:
                path = cache / 'events' / event_key(row) / f'layer{layer:02d}.pt'
                if not path.exists():
                    from wan.modules.causal_model import causal_rope_apply
                    frames = e0.shape[1]
                    tokens = x.shape[1] // frames
                    e = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
                    mod = (block.norm1(x).unflatten(1, (frames, tokens)) * (1+e[1]) + e[0]).flatten(1, 2)
                    att = block.self_attn
                    q = att.norm_q(att.q(mod)).view(1, mod.shape[1], 12, 128)
                    k = att.norm_k(att.k(mod)).view_as(q)
                    v = att.v(mod).view_as(q)
                    q = causal_rope_apply(q, grid, freqs, start_frame=start).to(v.dtype)[0]
                    k = causal_rope_apply(k, grid, freqs, start_frame=start).to(v.dtype)[0]
                    hp = cache / 'history' / history_key(row) / f'layer{layer:02d}.pt'
                    if not hp.exists():
                        save(hp, dict(kh=cpu(hk[0].transpose(0, 1)), vh=cpu(hv[0].transpose(0, 1)),
                            layer=layer, history_tokens=row['history_tokens'], trajectory_id=row['trajectory_id'],
                            teacher_sha256=m['teacher_sha256'], cache_format=3))
                    save(path, dict(q=cpu(q[row['query_indices']].transpose(0, 1)),
                        kc=cpu(k.transpose(0, 1)), vc=cpu(v[0].transpose(0, 1)),
                        history_file=str(hp), query_indices=row['query_indices'],
                        sample_id=row['sample_id'], split=row['split'], layer=layer, step=row['step'],
                        history_tokens=row['history_tokens'], grid=grid.tolist(), start_frame=start,
                        visibility='all original history and whole current chunk; verified chunkwise model path',
                        current_summary_reuse='forbidden across layers or denoising steps',
                        teacher_sha256=m['teacher_sha256'], cache_format=3))
            return super()._block(*a, **kw)

    status_path = root / phase / f'rank{rank}.json'
    update(status_path, status='loading', rank=rank, completed=0, total=len(rows))
    net = Capture(load_teacher('cuda'), None, layers=m['layers']).eval()
    versions = tuple(p._version for p in net.model.parameters())
    begin = time.monotonic()
    with torch.no_grad(), sdpa_backend():
        for index, row in enumerate(rows, 1):
            if (root / 'STOP').exists():
                raise RuntimeError('Stop requested')
            receipt = cache / 'receipts' / (event_key(row) + '.json')
            if not receipt.exists():
                ds = datasets[row['split']]
                sample = ds.load(row['replay_index'])
                if row['chunk'] == 0:
                    payload = ds._payload
                    sample = dict(row, history=payload['latents'][:, :0],
                        noisy=payload['noisy'][0, row['step']], target=payload['teacher_x0'][0, row['step']],
                        prompt_embeds=payload['prompt_embeds'], context_noise=payload['context_noise'])
                assert sample['sample_id'] == row['sample_id']
                start_frame = sample['history'].shape[1]
                result = net.replay(sample, sparse=False, checkpoint=False, collect_aux=False, recent_gradient=False)
                if row['chunk'] == 0:
                    torch.testing.assert_close(result['x0'].float(), sample['target'].cuda().float(), atol=3e-3, rtol=3e-3)
                del result
                files = [cache / 'events' / event_key(row) / f'layer{layer:02d}.pt' for layer in m['layers']]
                assert all(p.is_file() for p in files)
                atomic_json(receipt, dict(cache_format=3, sample_id=row['sample_id'],
                    query_indices=row['query_indices'], layers=m['layers'], teacher_sha256=m['teacher_sha256'],
                    files=[dict(path=str(p), bytes=p.stat().st_size) for p in files]))
            update(status_path, status='collecting', rank=rank, completed=index, total=len(rows),
                elapsed_seconds=time.monotonic()-begin, sample_id=row['sample_id'])
    assert tuple(p._version for p in net.model.parameters()) == versions
    update(status_path, status='completed', rank=rank, completed=len(rows), total=len(rows),
        elapsed_seconds=time.monotonic()-begin, backbone_unchanged=True, collector_sha256=sha(__file__))


if __name__ == '__main__':
    main()
