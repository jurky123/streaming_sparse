"""Strict v0.3 views over the existing immutable QKV format."""
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
from ..capture.dataset_io import AttentionDataset, query_slices
from ..v02.data import SnapshotData as V02Data

HERE = Path(__file__).parent


class MissingQKVError(FileNotFoundError):
    pass


class SnapshotData:
    def __init__(self, root, split, heads=(0, 3, 6, 9), layers=(4, 14, 24),
                 require_complete=True, check_files=True):
        self.root = Path(root).resolve()
        self.split, self.heads, self.layers = split, tuple(heads), tuple(layers)
        protocol = json.loads((HERE / 'configs/protocol.json').read_text())
        splits = json.loads((self.root / 'splits.json').read_text())
        if splits != protocol['splits']:
            raise ValueError('v0.3 must preserve the original 16/4/4 Prompt split')
        self.dataset = AttentionDataset(self.root, split)
        self.dataset.records = [r for r in self.dataset.records
                                if r['head_id'] in heads and r['layer_id'] in layers]
        self.groups, self.strata = defaultdict(list), defaultdict(list)
        cells = set()
        for r in self.dataset.records:
            cell = tuple(r[k] for k in ('prompt_id', 'seed', 'layer_id', 'head_id', 'history_tokens', 'step_id'))
            if cell in cells:
                raise ValueError(f'Duplicate/conflicting QKV cell: {cell}')
            cells.add(cell)
            if r['query_tile'] != 16 or r['step_id'] not in range(4) or r['seed'] not in (0, 1):
                raise ValueError('QKV query tile, step, or generation seed violates protocol')
            self.groups[r['cache_version']].append(r)
        if not self.groups:
            raise ValueError(f'Empty {split} split for the requested layer/head IDs')
        for version, rows in self.groups.items():
            first = rows[0]
            if len(rows) != 4 or {r['step_id'] for r in rows} != set(range(4)):
                raise ValueError(f'Every snapshot requires exactly four diffusion steps: {version}')
            scope = ('prompt_id', 'seed', 'layer_id', 'head_id', 'history_tokens', 'history')
            if any(any(r[k] != first[k] for k in scope) for r in rows):
                raise ValueError(f'Snapshot metadata/history mismatch: {version}')
            if any(r.get('query_tile_starts') != first.get('query_tile_starts') for r in rows):
                raise ValueError('The four diffusion steps must retain aligned query windows')
            self.strata[(first['layer_id'], first['head_id'])].append(version)
        if require_complete:
            expected = set(itertools.product(splits[split], (0, 1), layers, heads,
                                             (4680, 9360, 14040), range(4)))
            if cells != expected:
                raise ValueError(f'Incomplete v0.3 {split} coverage: missing={len(expected-cells)}, unexpected={len(cells-expected)}')
            for r in self.dataset.records:
                if r.get('captured_queries') != 128 or len(r.get('query_tile_starts', [])) != 8:
                    raise ValueError('Official capture requires eight 16-token tiles per event')
                if not 0 <= r.get('dense_fp32_vs_backend_error', float('inf')) <= 1e-4:
                    raise ValueError('Capture backend dense-attention error exceeds the v0.2 acceptance bound')
        self.require_complete = require_complete
        if check_files:
            self.check_files()

    def check_files(self):
        missing = {kind: sorted({str(self.root/r[kind]) for r in self.dataset.records
                                 if not (self.root/r[kind]).is_file()}) for kind in ('history', 'event')}
        if any(missing.values()):
            raise MissingQKVError(f'Missing real QKV tensors: {len(missing["history"])} history files, '
                                  f'{len(missing["event"])} query files in {self.split}; '
                                  'events.jsonl is only an index. Restore tensors before training/evaluation.')
        return True

    def load(self, version, device):
        events = V02Data.load(self, version, device)
        for event, row in zip(events, sorted(self.groups[version], key=lambda r: r['step_id'])):
            for key in ('cache_version', 'prompt_id', 'seed', 'layer_id', 'head_id', 'step_id'):
                if event[key] != row[key]:
                    raise ValueError(f'Tensor/manifest mismatch: {key}, {version}')
            if event['k_hist'].shape != event['v_hist'].shape or event['k_hist'].shape != (row['history_tokens'], 128):
                raise ValueError('Invalid history KV shape')
            if event['q'].shape != event['o_dense'].shape or event['q'].shape[-1] != 128:
                raise ValueError('Invalid captured Q/dense output shape')
            if len(event['query_indices']) != len(event['q']) or len(event['q']) != row['captured_queries']:
                raise ValueError('Query indices/capture count mismatch')
            if len(event['k_current']) == 0 or event['k_current'].shape != event['v_current'].shape:
                raise ValueError('Current KV must be nonempty and retained in full')
            if any(s.stop-s.start != 16 for s in query_slices(event, 16)):
                raise ValueError('v0.3 requires complete 16-query tiles')
        return events

    def sample_versions(self, rng, count=4):
        keys = sorted(self.strata)
        rng.shuffle(keys)
        selected = []
        for i in range(count):
            choices = [v for v in self.strata[keys[i % len(keys)]] if v not in selected]
            if not choices:
                choices = self.strata[keys[i % len(keys)]]
            selected.append(rng.choice(sorted(choices)))
        return selected

    def validation_versions(self):
        """One fixed seed/history snapshot per Prompt x Layer x Head."""
        groups = defaultdict(list)
        for version, rows in self.groups.items():
            r = rows[0]
            groups[(r['prompt_id'], r['layer_id'], r['head_id'])].append(version)
        return [min(groups[k], key=lambda v: (self.groups[v][0]['seed'],
                    self.groups[v][0]['history_tokens'], v)) for k in sorted(groups)]

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in ('events.jsonl', 'splits.json'):
            digest.update((self.root/name).read_bytes())
        for path in sorted({self.root/r[k] for r in self.dataset.records for k in ('history', 'event')}):
            stat = path.stat()
            digest.update(str((str(path), stat.st_size, stat.st_mtime_ns)).encode())
        return digest.hexdigest()


def main():
    import argparse
    p = argparse.ArgumentParser(description='Check coverage and real QKV references without running an experiment')
    p.add_argument('--data', required=True)
    p.add_argument('--heads', type=int, nargs='+', default=[0, 3, 6, 9])
    args = p.parse_args()
    for split in ('train', 'val', 'test'):
        data = SnapshotData(args.data, split, heads=args.heads)
        print(json.dumps(dict(split=split, snapshots=len(data.groups), events=len(data.dataset.records))))


if __name__ == '__main__':
    main()
