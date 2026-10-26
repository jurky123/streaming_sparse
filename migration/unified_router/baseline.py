"""P1 true-AV/total-token frontier: Dense, historical, current, unified."""
import argparse
import os
from pathlib import Path
import time
import torch

from .common import config, load_raw, atomic_json, update, paired_seed, sha
from .core import build, scores, initial_mask, LabelStore


def metrics(truth, mask, sketch):
    output = truth.output(mask)
    sse = float((output-truth.target).square().sum())
    energy = float(truth.target.square().sum())
    costs = sketch.costs[None, None]
    retained = mask*costs
    query_count = mask.shape[0]*mask.shape[1]
    h = sketch.origins == 0
    keep_h = float(retained[..., h].sum())/(query_count*sketch.history_tokens) if sketch.history_tokens else None
    keep_c = float(retained[..., ~h].sum())/(query_count*sketch.current_tokens)
    return dict(nmse=sse/(energy+1e-8), sse=sse, energy=energy,
        keep=float(retained.sum())/(query_count*(sketch.history_tokens+sketch.current_tokens)),
        keep_H=keep_h, keep_C=keep_c, kept_query_tokens=float(retained.sum()),
        total_query_tokens=query_count*(sketch.history_tokens+sketch.current_tokens),
        raw_queries=query_count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    root, m = config(args.directory)
    assert (root/'P0_math.json').is_file()
    paired_seed(21)
    torch.cuda.set_device(0)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    rows = [r for r in m['rows'] if r['split'] == 'val' and
            (not args.pilot or r['sample_id'] in m['pilot_sample_ids'])]
    phase = 'P0_real' if args.pilot else 'P1_baseline'
    for layer in m['layers'][rank::world]:
        parameters = torch.load(root/f'parameters/layer{layer:02d}.pt', map_location='cpu', weights_only=True)['parameters']
        parameters = {k: v.cuda() for k, v in parameters.items()}
        output_rows = []
        start = time.monotonic()
        for index, row in enumerate(rows, 1):
            if (root/'STOP').exists():
                raise RuntimeError('Stop requested')
            data = load_raw(root, row, layer)
            sketch = build(data['kh'], data['vh'], data['kc'], data['vc'], parameters, need_value=False)
            truth = LabelStore(**data)
            # Independent raw-token reference, including the last partial pages.
            keys = torch.cat((data['kh'], data['kc']), 1)
            vals = torch.cat((data['vh'], data['vc']), 1)
            logits = torch.bmm(data['q'][:, :2].float(), keys.float().transpose(1, 2))/128**.5
            direct = torch.bmm(logits.double().softmax(-1), vals.double())
            torch.testing.assert_close(truth.target[:, :2], direct, atol=5e-5, rtol=5e-5)
            for dim in [64, 128]:
                score, _ = scores(data['q'], sketch, key_dim=dim, content=None)
                combinations = [('Dense', 0.)] + [(mode, beta) for mode in m.get('baseline_modes',['H-Sol','C-Sol','U-Sol']) if mode!='Dense' for beta in m['beta_grid']]
                for mode, beta in combinations:
                    mask = initial_mask(score, sketch, beta, mode)
                    result = metrics(truth, mask, sketch)
                    if row['history_tokens'] == 0:
                        assert mask.all() and result['nmse'] < 1e-20
                    output_rows.append(dict(sample_id=row['sample_id'], prompt_id=row['prompt_id'],
                        generation_seed=row['generation_seed'], history_tokens=row['history_tokens'],
                        step=row['step'], mode=mode, beta=beta, key_dim=dim, layer=layer, **result))
            update(root/phase/f'rank{rank}.json', status='evaluating', layer=layer, completed=index,
                total=len(rows), elapsed_seconds=time.monotonic()-start)
            del data, sketch, truth, keys, vals, logits, direct
        atomic_json(root/phase/f'layer{layer:02d}.json', dict(passed=True, layer=layer,
            scope='sampled128 original Queries, local Dense-input attention; not closed-loop quality',
            key64_status='candidate projection with full128 reference; no noninferiority claim',
            first_chunk_dense=True, first_chunk_anchor=False, current_prototypes=16,
            rows=output_rows, execution_sha256=sha(__file__)))
    update(root/phase/f'rank{rank}.json', status='completed', layers=m['layers'][rank::world])


if __name__ == '__main__':
    main()
