"""Export jointly trained routing parameters and head checkpoints after all workers."""
import json
from pathlib import Path
import torch
from ..common import atomic_json, save, sha


def finalize(root, m):
    summary = {}
    for variant in m['variants']:
        layers, records = {}, []
        for layer in m['layers']:
            folder = root / f'training/layer{layer:02d}/{variant}'
            state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            best = json.loads((folder / 'best_validation.json').read_text())
            initial = json.loads((folder / 'initial_validation.json').read_text())
            assert state['best_value'] == best['nmse']
            assert best['keep'] <= state['max_best_keep'] + 1e-10
            layers[layer] = dict(head=state['head'], parameters=state['parameters'], updates=state['updates'])
            records.append(dict(layer=layer, updates=state['updates'], initial_nmse=initial['nmse'],
                                best_nmse=best['nmse'], keep=best['keep'], checkpoint_sha256=sha(folder / 'best.pt')))
        save(root / 'exports' / f'{variant}.joint_best.pt', dict(layers=layers, variant=variant,
             best_metric='actual_deployment_attention_nmse', backbone_included=False, protocol=m))
        summary[variant] = dict(initial_nmse=sum(r['initial_nmse'] for r in records)/len(records),
            best_nmse=sum(r['best_nmse'] for r in records)/len(records),
            keep=sum(r['keep'] for r in records)/len(records), rows=records)
    frozen = json.loads((root / 'parent_lock.json').read_text())
    for path, digest in frozen.items():
        assert sha(path) == digest, path
    atomic_json(root / 'training_summary.json', dict(variants=summary, parent_unchanged=True,
                complete=True, e2e_video_evaluation_pending=True))
    from .plot import plot
    plot(root)
