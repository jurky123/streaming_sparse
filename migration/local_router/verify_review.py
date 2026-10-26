"""Verify the review bundle using metric records, without weights or tensors."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def scientific_identity(source):
    here = source / 'streaming_sparse/v08'
    paths = list(here.glob('*.py')) + [here / 'configs/protocol.json']
    paths += [source / name for name in (
        'utils/wan_wrapper.py', 'utils/scheduler.py', 'wan/modules/causal_model.py',
        'wan/modules/model.py', 'wan/modules/attention.py',
        'streaming_sparse/v05/primitives.py', 'streaming_sparse/capture/capture_runner.py',
        'configs/default_config.yaml', 'configs/self_forcing_dmd.yaml',
        'wan_models/Wan2.1-T2V-1.3B/config.json')]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(source)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def verify(root):
    root = Path(root).resolve()
    entries = []
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        entries.append(name)
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert files == set(entries) | {'SHA256SUMS'}, 'Unlisted or missing files'
    banned = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.pkl', '.pickle',
              '.npy', '.npz', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif', '.zip'}
    for name in files:
        path = Path(name)
        assert path.suffix.lower() not in banned, name
        assert not {'data', 'checkpoints', '__pycache__', '.venv'} & set(path.parts), name
        if path.suffix.lower() in {'.png', '.pdf'}:
            assert path.stem == 'end_to_end_comparison', name
    config = json.loads((root / 'metadata/experiment_config.json').read_text())
    assert scientific_identity(root / 'code') == config['source_sha256']
    assert 'rows' not in config
    assert config['layers'] == list(range(1, 30))
    summary = json.loads((root / 'results/quality/end_to_end_assessment.json').read_text())
    manifest = summary['manifest']
    assert manifest['dense_layers'] == [0] and manifest['sparse_layers'] == list(range(1, 30))
    assert manifest['checkpoints']['Klast']['router_sha256'] == manifest['checkpoints']['Kbest']['router_sha256']
    expected = {(p, s) for p in manifest['prompt_ids'] for s in manifest['generation_seeds']}
    cases = [r for i in range(8) for r in json.loads(
        (root / f'results/quality/rank{i}/video_rows.json').read_text())]
    assert len(cases) == 16 and {(r['prompt_id'], r['generation_seed']) for r in cases} == expected
    for mode in ['Dense', 'mass', 'Klast', 'Vlast', 'Kbest', 'Vbest']:
        rows = [r for i in range(8) for r in json.loads(
            (root / f'results/quality/rank{i}/{mode}_offline.json').read_text())['rows']]
        assert len(rows) == 192
        assert {(r['prompt_id'], r['generation_seed'], r['history_tokens'], r['step']) for r in rows} == {
            (p, s, h, t) for p, s in expected for h in (4680, 9360, 14040) for t in range(4)}
        for row in rows:
            close(row['nmse'], row['sse'] / (row['energy'] + 1e-8))
        prompt_values = [statistics.mean(r['nmse'] for r in rows if r['prompt_id'] == p)
                         for p in manifest['prompt_ids']]
        close(statistics.mean(prompt_values), summary['offline'][mode]['prompt_equal_error'])
        if mode == 'Dense':
            assert all(r['nmse'] == 0 for r in rows)
            continue
        values = []
        for case in cases:
            data = case['metrics'][mode]
            frames = data['quality']['per_frame']
            assert len(frames) == 45 and all(f['identical'] for f in frames[:9])
            assert {(c['history_tokens'], c['step']) for c in data['sparsity']} == {
                (h, t) for h in (4680, 9360, 14040) for t in range(4)}
            assert all(c['routed_queries'] == 29 * 12 * 4680 and c['keep'] <= .2 + 1e-9
                       for c in data['sparsity'])
            for scope, part in [('full', frames), ('sparse_frames_only', frames[9:])]:
                stats = data['quality'][scope]
                close(statistics.mean(f['mse_rgb01'] for f in part), stats['mse_rgb01'])
                close(statistics.mean(f['lpips'] for f in part), stats['lpips'])
            values.append(data['quality']['sparse_frames_only'])
        mse = statistics.mean(v['mse_rgb01'] for v in values)
        close(-10 * math.log10(mse), summary['video'][mode]['sparse_frames_only']['psnr_db'])
        close(statistics.mean(v['lpips'] for v in values), summary['video'][mode]['sparse_frames_only']['lpips'])
    with (root / 'results/training/layer_summary.csv').open() as stream:
        layers = list(csv.DictReader(stream))
    assert len(layers) == 29 and all(r['done'] == 'True' for r in layers)
    assert sorted(int(r['layer']) for r in layers) == list(range(1, 30))
    bootstrap = 'not checked (install numpy to reproduce the exact bootstrap draws)'
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        for name, k, v in [('last_V_minus_K', 'Klast', 'Vlast'), ('best_V_minus_K', 'Kbest', 'Vbest')]:
            pids = sorted(manifest['prompt_ids'])
            delta = np.array([summary['video'][v]['prompts'][p] - summary['video'][k]['prompts'][p] for p in pids])
            draw = np.random.default_rng(20261019).integers(0, len(pids), size=(5000, len(pids)))
            ci = np.quantile(delta[draw].mean(-1), [.025, .975])
            expected_ci = summary['paired_lpips'][name]['prompt_paired_95CI']
            for a, b in zip(ci, expected_ci):
                close(float(a), b)
            close(float(delta.mean()), summary['paired_lpips'][name]['delta_V_minus_K'])
        bootstrap = 'passed (numpy; 5000 paired prompt bootstrap draws)'
    result = dict(passed=True, files_checked=len(entries), scientific_source_identity='matched',
                  excluded_payload_check='passed', offline_observations=6 * 192,
                  video_cases=16, frames_per_case=45, trained_layers=29, bootstrap=bootstrap)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent)
