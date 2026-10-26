"""Retire the single-GPU worker after its initial validation checkpoint."""
import json
import os
from pathlib import Path
import signal
import time
import torch
from streaming_sparse.v08.common import atomic_json,sha


def main():
    run=Path('streaming_sparse/v08/runs/seed0/F-K')
    prep=Path('migration/restoration/20261022_133350')
    worker=242836;controller=187615
    assert b'streaming_sparse.v08.train'in Path(f'/proc/{worker}/cmdline').read_bytes()
    assert b'start_approved_continuation.py'in Path(f'/proc/{controller}/cmdline').read_bytes()
    while not(run/'latest.pt').exists():time.sleep(2)
    checkpoint=torch.load(run/'latest.pt',map_location='cpu',weights_only=True)
    assert checkpoint['updates']==0,'Handle later updates explicitly instead of discarding work'
    assert len(checkpoint['curve'])==1 and checkpoint['curve'][0]['inputs']==192
    os.kill(worker,signal.SIGSTOP)
    assert not(run/'metrics.jsonl').exists(),'An optimizer update completed; preserve it before handoff'
    os.kill(controller,signal.SIGTERM)
    os.kill(worker,signal.SIGTERM);os.kill(worker,signal.SIGCONT)
    state=json.loads((prep/'continuation_status.json').read_text())
    atomic_json(prep/'single_gpu_retired_status.json',state)
    atomic_json(prep/'handoff.json',dict(status='ready_for_multigpu',resume=str((run/'latest.pt').resolve()),
        sha256=sha(run/'latest.pt'),completed_updates=0,initial_validation_inputs=192,
        retired_worker=worker,retired_controller=controller,reason='User requested sample-parallel training and router optimization'))
    state.update(status='switching_to_multigpu',worker_pid=None,controller_pid=None,initial_validation=dict(completed=192,total=192,split='val'))
    atomic_json(prep/'continuation_status.json',state)


if __name__=='__main__':main()
