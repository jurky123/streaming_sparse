"""Non-destructive, content-addressed QKV merge; no tensor collection."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import torch
from .paths import protect_output
from ..capture.dataset_io import tensor_digest
from ..v02.evaluation_journal import atomic_json

CELL = ('prompt_id', 'seed', 'layer_id', 'head_id', 'history_tokens', 'step_id')


def payload_digest(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    digest = hashlib.sha256()
    for name in sorted(value):
        if isinstance(value[name], torch.Tensor):
            digest.update(name.encode())
            digest.update(tensor_digest(value[name]).encode())
    return digest.hexdigest(), value


def merge(sources, output, heads=tuple(range(12)), require_complete=True):
    output = protect_output(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Merge requires a fresh output; never overwrites an existing capture')
    protocol = json.loads((Path(__file__).parent/'configs/protocol.json').read_text())
    selected, digests, identities, duplicate_cells = {}, {}, {}, 0
    for root in map(Path, sources):
        for line in (root/'events.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['head_id'] not in heads:
                continue
            paths = [root/r[k] for k in ('history', 'event')]
            hashes = []
            for kind, path in zip(('history', 'event'), paths):
                key = str(path.resolve())
                if key not in digests:
                    digest, value = payload_digest(path)
                    if value['cache_version'] != r['cache_version']:
                        raise ValueError('Tensor/manifest cache identity mismatch')
                    digests[key] = digest
                    identities[key] = {k: v for k, v in value.items() if not isinstance(v, torch.Tensor)}
                    if kind == 'history':
                        scope = (r['prompt_id'], r['layer_id'], r['head_id'])
                        prefix = hashlib.sha256(str(scope).encode()).hexdigest()[:16]
                        expected_version = prefix+'-'+tensor_digest(value['k_hist'], value['v_hist'], value['token_positions'])
                        if expected_version != r['cache_version']:
                            raise ValueError('History content does not match its cache_version')
                        identities[key]['scope'] = scope
                identity_fields = identities[key]
                if identity_fields['cache_version'] != r['cache_version']:
                    raise ValueError('Reused tensor path has conflicting cache metadata')
                if kind == 'history':
                    if identity_fields['scope'] != (r['prompt_id'], r['layer_id'], r['head_id']):
                        raise ValueError('History path reused across incompatible layer/head/prompt scopes')
                elif any(identity_fields.get(field) != r[field] for field in CELL+('chunk_id', 'query_tile', 'captured_queries')):
                    raise ValueError('Event tensor/manifest metadata mismatch')
                hashes.append(digests[key])
            cell = tuple(r[k] for k in CELL)
            if cell in selected:
                if selected[cell]['digests'] != hashes:
                    raise ValueError(f'Conflicting QKV tensors for cell {cell}; no output has been written')
                duplicate_cells += 1
                continue
            selected[cell] = dict(record=r, paths=paths, digests=hashes)
    if require_complete:
        prompts = [p['prompt_id'] for p in protocol['prompts']]
        expected = set(itertools.product(prompts, (0, 1), (4, 14, 24), heads,
                                         (4680, 9360, 14040), range(4)))
        if set(selected) != expected:
            raise ValueError(f'Merge coverage mismatch: missing={len(expected-set(selected))}, unexpected={len(set(selected)-expected)}')
    output.mkdir(parents=True, exist_ok=True)
    rows, copied = [], set()
    for cell, item in sorted(selected.items()):
        r = dict(item['record'])
        for kind, source, digest in zip(('history', 'event'), item['paths'], item['digests']):
            relative = Path('history' if kind == 'history' else 'queries')/f'{digest}.pt'
            target = output/relative
            if relative not in copied:
                target.parent.mkdir(exist_ok=True)
                temp = target.with_suffix('.pt.tmp')
                shutil.copyfile(source, temp)
                temp.replace(target)
                copied.add(relative)
            r[kind] = str(relative)
        rows.append(r)
    temporary = output/'events.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    temporary.replace(output/'events.jsonl')
    atomic_json(output/'splits.json', protocol['splits'])
    report = dict(events=len(rows), snapshots=len({r['cache_version'] for r in rows}),
                  duplicate_identical_cells=duplicate_cells, heads=list(heads),
                  sources=[str(Path(p).resolve()) for p in sources], conflict_policy='Reject before writing any output')
    atomic_json(output/'merge_audit.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--heads', nargs='+', type=int, default=list(range(12)))
    args = p.parse_args()
    print(json.dumps(merge(args.sources, args.output, tuple(args.heads)), indent=2))


if __name__ == '__main__':
    main()
