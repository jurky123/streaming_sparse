"""CPU-only checkout/source/checkpoint verification; no model load or GPU use."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    manifest = json.loads((ROOT / 'migration/inventory.json').read_text())
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'File hash mismatch: {path}')
    from streaming_sparse.v08.common import source_identity
    from streaming_sparse.v08.execution.accelerate import identity
    from streaming_sparse.v08.launcher.worker import launcher_identity
    import torch
    assert source_identity() == manifest['source_sha256'], 'Scientific source identity changed'
    assert identity() == manifest['execution_sha256'], 'Execution source identity changed'
    assert launcher_identity() == manifest['launcher_sha256'], 'Launcher source identity changed'
    checkpoint = torch.load(ROOT / manifest['recovery_path'], map_location='cpu', weights_only=True)
    assert checkpoint['source_sha256'] == manifest['source_sha256']
    assert checkpoint['updates'] == manifest['recovery_updates']
    print(json.dumps(dict(verified_files=len(manifest['files']), source_sha256=source_identity(),
        group=checkpoint['group'], seed=checkpoint['seed'], updates=checkpoint['updates'],
        torch=str(torch.__version__), cuda=torch.version.cuda, data_included=False), indent=2))


if __name__ == '__main__':
    main()
