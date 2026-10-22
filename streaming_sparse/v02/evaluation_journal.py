"""Durable snapshot boundaries for exact, duplicate-free evaluation recovery."""
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import torch


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        path = path.resolve()
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(model, data, versions, keeps, baselines, oracle_tiles, save_masks):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    value = dict(model_sha256=digest.hexdigest(), config=asdict(model.config),
                 records=[data.groups[v] for v in versions], versions=versions,
                 keeps=list(keeps), baselines=baselines, oracle_tiles=oracle_tiles,
                 save_masks=save_masks, metric_version=2)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def row_key(row):
    return row['step_id'], row['tile_index'], row['method'], row['requested_keep']


def keys_match(rows, expected):
    try:
        return len(rows) == len(expected) and {row_key(r) for r in rows} == expected
    except (KeyError, TypeError):
        return False


def json_lines(path, limit=None):
    if not path.exists():
        return
    with path.open('rb') as stream:
        while limit is None or stream.tell() < limit:
            line = stream.readline()
            if not line or not line.endswith(b'\n'):
                return
            try:
                row = json.loads(line)
            except ValueError:
                return
            yield row, stream.tell()


class EvaluationJournal:
    def __init__(self, path, signature, versions, expected, oracle_versions, keeps,
                 adopt_legacy=False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.versions = versions
        self.done = 0
        self.records, self.oracle_rows = [], []
        checkpoint = self.path / 'resume.json'
        raw = self.path / 'per_tile.jsonl'
        oracle = self.path / 'oracle.jsonl'
        raw_bytes = oracle_bytes = 0
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['signature'] != signature:
                raise ValueError('Evaluation resume identity mismatch: model/data/settings changed')
            self.done = saved['snapshots_done']
            raw_bytes, oracle_bytes = saved['raw_bytes'], saved['oracle_bytes']
            for source, size in ((raw, raw_bytes), (oracle, oracle_bytes)):
                if size and (not source.exists() or source.stat().st_size < size):
                    raise ValueError(f'Evaluation journal data shorter than committed offset: {source}')
            self.records = [r for r, _ in json_lines(raw, raw_bytes)]
            self.oracle_rows = [r for r, _ in json_lines(oracle, oracle_bytes)]
            self.validate(expected, oracle_versions, keeps)
        elif raw.exists() and raw.stat().st_size:
            if not adopt_legacy:
                raise ValueError('Legacy partial evaluation exists; use --resume-existing to validate and adopt it')
            archive = self.path.parent / 'interrupted' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) / self.path.name
            archive.mkdir(parents=True, exist_ok=False)
            for name in ('per_tile.jsonl', 'oracle.jsonl', 'progress.json'):
                source = self.path / name
                if source.exists():
                    shutil.copy2(source, archive / name)
            progress = json.loads((self.path / 'progress.json').read_text())
            maximum = min(progress['snapshots_done'], len(versions))
            all_oracle = list(json_lines(oracle))
            for version, items in itertools.groupby(json_lines(raw), key=lambda item: item[0]['cache_version']):
                if self.done >= maximum or version != versions[self.done]:
                    break
                items = list(items)
                rows = [r for r, _ in items]
                if not keys_match(rows, expected[version]):
                    break
                orows = [(r, offset) for r, offset in all_oracle if r['cache_version'] == version]
                required = set(keeps) if version in oracle_versions else set()
                if len(orows) != len(required) or {r['requested_keep'] for r, _ in orows} != required:
                    break
                self.records.extend(rows)
                self.oracle_rows.extend(r for r, _ in orows)
                raw_bytes = items[-1][1]
                if orows:
                    oracle_bytes = orows[-1][1]
                self.done += 1
            self.validate(expected, oracle_versions, keeps)
            atomic_json(self.path / 'legacy_adoption.json', dict(archive=str(archive),
                        validated_snapshots=self.done, raw_bytes=raw_bytes, oracle_bytes=oracle_bytes))
        for source, size in ((raw, raw_bytes), (oracle, oracle_bytes)):
            if source.exists():
                with source.open('r+b') as stream:
                    stream.truncate(size)
        self.raw = raw.open('a')
        self.oracle = oracle.open('a') if oracle_versions else None
        self.commit(self.done)
        print(json.dumps(dict(event='evaluation_resume', snapshots_done=self.done,
                              total_snapshots=len(versions), identity=signature)), flush=True)

    def validate(self, expected, oracle_versions, keeps):
        groups = {v: [] for v in self.versions[:self.done]}
        for row in self.records:
            if row['cache_version'] not in groups:
                raise ValueError('Unexpected snapshot in committed evaluation records')
            groups[row['cache_version']].append(row)
        for version, rows in groups.items():
            if not keys_match(rows, expected[version]):
                raise ValueError(f'Incomplete/duplicate committed evaluation rows: {version}')
        expected_oracle = {(v, keep) for v in groups if v in oracle_versions for keep in keeps}
        actual = {(r['cache_version'], r['requested_keep']) for r in self.oracle_rows}
        if actual != expected_oracle or len(actual) != len(self.oracle_rows):
            raise ValueError('Incomplete/duplicate committed Oracle records')

    def commit(self, done):
        for stream in (self.raw, self.oracle):
            if stream:
                stream.flush()
                os.fsync(stream.fileno())
        self.done = done
        atomic_json(self.path / 'resume.json', dict(signature=self.signature, snapshots_done=done,
                    total_snapshots=len(self.versions), raw_bytes=self.raw.tell(),
                    oracle_bytes=self.oracle.tell() if self.oracle else 0))
        atomic_json(self.path / 'progress.json', dict(snapshots_done=done, total_snapshots=len(self.versions)))

    def close(self):
        self.raw.close()
        if self.oracle:
            self.oracle.close()
