"""Verify an extracted migration bundle without allocating a GPU."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    entries = json.loads((root / 'MIGRATION_INVENTORY.json').read_text())
    for index, item in enumerate(entries, 1):
        name = item['name']
        if not name.startswith('Self-Forcing/') or '..' in Path(name).parts:
            raise ValueError('Unsafe inventory path')
        path = root / name.removeprefix('Self-Forcing/')
        if path.is_symlink() or path.stat().st_size != item['size']:
            raise ValueError('Missing file, symlink or size mismatch: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError('SHA-256 mismatch: ' + name)
        if index % 1000 == 0:
            print(f'Verified {index}/{len(entries)} files', flush=True)
    print(f'Verified all {len(entries)} files. No GPU work was started.')
    print('The explicit supervisor.stop marker keeps the experiment paused.')


if __name__ == '__main__':
    main()
