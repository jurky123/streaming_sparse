"""Persistent supervisor: eight independent layer workers; record child failures."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from ..common import config, atomic_json, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', default='streaming_sparse/v10/experiments/opt12_joint_20261025')
    a = p.parse_args()
    root, m = config(a.directory)
    assert json.loads((root / 'preflight.json').read_text())['passed']
    out = root / 'training'
    out.mkdir(exist_ok=True)
    lock = json.loads((root / 'parent_lock.json').read_text())
    for f, digest in lock.items():
        assert sha(f) == digest, f
    atomic_json(root / 'source_manifest.json', {str(f.resolve()): sha(f) for f in Path(__file__).parent.glob('*.py')})
    pending = [i for i in m['layers'] if not (out / f'layer{i:02d}/completed.json').exists()]
    assignments = {rank: pending[rank::8] for rank in range(8)}
    procs = []
    for rank in range(8):
        if not assignments[rank]:
            atomic_json(out / f'rank{rank}.json', dict(status='completed', layers=[]))
            continue
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(rank), RANK=str(rank), WORLD_SIZE='8', OMP_NUM_THREADS='4',
                   TORCHINDUCTOR_COMPILE_THREADS='2',
                   TORCHINDUCTOR_CACHE_DIR=f'/tmp/slimav_opt12/rank{rank}/inductor',
                   TRITON_CACHE_DIR=f'/tmp/slimav_opt12/rank{rank}/triton')
        f = open(out / f'rank{rank}.log', 'a', buffering=1)
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'migration.runtime.bootstrap',
                                  'migration.unified_router.opt12.train', '--directory', str(root), '--layers', *map(str, assignments[rank])],
                                 env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((rank, proc, f))
    atomic_json(out / 'launcher.json', dict(pid=os.getpid(), started_at=time.time(),
                 revision='fp32_all_surrogate_backward_v3', workers=[dict(rank=i, pid=p.pid, layers=assignments[i]) for i, p, _ in procs]))
    codes = [0] * 8
    for rank, p, _ in procs:
        codes[rank] = p.wait()
    for _, _, f in procs:
        f.close()
    atomic_json(out / 'exit_codes.json', codes)
    if not any(codes) and all((out / f'layer{i:02d}/completed.json').exists() for i in m['layers']):
        from .finalize import finalize
        finalize(root, m)
    print('worker exit codes:', codes, flush=True)
    raise SystemExit(int(any(codes)))


if __name__ == '__main__':
    main()
