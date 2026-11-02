"""Verify the local installation without changing the upstream migration manifest."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import torch

from streaming_sparse.v08.common import source_identity
from streaming_sparse.v08.execution.accelerate import identity
from streaming_sparse.v08.launcher.worker import launcher_identity

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'migration/inventory.json').read_text())
mismatches = []
for row in manifest['files']:
    path = ROOT / row['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        mismatches.append(row['path'])
assert set(mismatches) <= {'.gitignore', 'README.md', 'REBUILD.md'}, mismatches
assert source_identity() == manifest['source_sha256']
assert identity() == manifest['execution_sha256']
assert launcher_identity() == manifest['launcher_sha256']
checkpoint = torch.load(ROOT / manifest['recovery_path'], map_location='cpu', weights_only=True)
assert checkpoint['source_sha256'] == manifest['source_sha256']
assert checkpoint['updates'] == manifest['recovery_updates']
assert str(torch.__version__) == '2.6.0+cu124'
for requirement in (ROOT / 'migration/requirements.lock.txt').read_text().splitlines():
    if '==' in requirement:
        name, expected = requirement.split('==')
        assert importlib.metadata.version(name) == expected, requirement
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
assert torch.cuda.is_available(), 'CUDA is unavailable'
result = dict(passed=True, torch=str(torch.__version__), cuda=torch.version.cuda,
              verified_files=len(manifest['files']) - len(mismatches),
              upstream_stale_manifest_entries=mismatches,
              scientific_source_matches=True, execution_source_matches=True,
              launcher_matches=True, checkpoint_source_matches=True,
              checkpoint_updates=checkpoint['updates'], data_included=False)
output = ROOT / 'local_setup/results'
output.mkdir(parents=True, exist_ok=True)
(output / 'environment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
