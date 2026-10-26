"""Evaluate eight Val prompts independently on eight GPUs."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from ..common import config, atomic_json, sha
from .status import ROOT


def main():
    root, m = config(ROOT)
    assert all((root / f'training/layer{i:02d}/completed.json').exists() for i in m['layers'])
    assert not any(json.loads((root / 'training/exit_codes.json').read_text()))
    old = json.loads((root/'source_manifest.json').read_text())
    for f, digest in old.items():
        assert sha(f) == digest, f
    weights = {}
    for layer in m['layers']:
        for variant in m['variants']:
            for f in [root / f'training/layer{layer:02d}/{variant}/best.pt',
                      Path(m['parent_opt11']) / f'strong_training/layer{layer:02d}/{variant}/best.pt']:
                weights[str(f)] = sha(f)
    out = root / 'quality'
    out.mkdir(exist_ok=True)
    atomic_json(out/'evaluation_lock.json', dict(training_source=old, weights=weights,
                code={str(f.resolve()):sha(f) for f in Path(__file__).parent.glob('*.py')}))
    procs = []
    for rank in range(8):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(rank), RANK=str(rank), WORLD_SIZE='8', OMP_NUM_THREADS='4',
                   TORCHINDUCTOR_COMPILE_THREADS='2',
                   TORCHINDUCTOR_CACHE_DIR=f'/tmp/slimav_opt12_quality/rank{rank}/inductor',
                   TRITON_CACHE_DIR=f'/tmp/slimav_opt12_quality/rank{rank}/triton')
        f = open(out / f'rank{rank}.log', 'a')
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'migration.runtime.bootstrap',
                  'migration.unified_router.opt12.quality', '--directory', str(root)],
                  env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((proc, f))
    atomic_json(out/'launcher.json', dict(pid=os.getpid(), started_at=time.time(), workers=[p.pid for p,_ in procs]))
    codes = [p.wait() for p,_ in procs]
    for _, f in procs:
        f.close()
    atomic_json(out/'exit_codes.json', codes)
    print('quality exit codes:', codes, flush=True)
    raise SystemExit(int(any(codes)))


if __name__ == '__main__':
    main()
