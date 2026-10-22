"""Run the fixed four-prompt / two-generation-seed held-out experiment.

Run capture first using the command recorded in protocol.json. This script
audits coverage, writes prompt-disjoint splits, trains from scratch, and runs
train/validation comparisons using the fixed final (1000-step) checkpoints.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parent
DATA = ROOT / 'streaming_sparse/data/validation_4p2s'


def run(arguments, log):
    command = [sys.executable, '-m', *arguments]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['PYTHONUNBUFFERED'] = '1'
    tick = time.time()
    print('START', ' '.join(command), flush=True)
    with (EXPERIMENT / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(command=command, started=tick))+'\n')
    with (EXPERIMENT / log).open('w') as f:
        subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print('DONE', log, f'{time.time()-tick:.1f}s', flush=True)


def main():
    prompts = (ROOT/'prompts/MovieGenVideoBench_extended.txt').read_text().splitlines()[:4]
    ids = [hashlib.sha256(p.encode()).hexdigest()[:20] for p in prompts]
    if len(set(ids)) != 4:
        raise ValueError('The four prompts must be distinct')
    splits = dict(train=ids[:2], val=ids[2:], test=[])
    records = [json.loads(x) for x in (DATA/'events.jsonl').read_text().splitlines()]
    expected = {(pid, seed) for pid in ids for seed in (0,1)}
    pairs = Counter((r['prompt_id'],r['seed']) for r in records)
    if set(pairs) != expected or any(n != 12 for n in pairs.values()):
        raise ValueError(f'Expected eight prompt/seed pairs, each with 12 events: {pairs}')
    for pid, seed in expected:
        cells = {(r['history_tokens'],r['step_id']) for r in records
                 if (r['prompt_id'],r['seed']) == (pid,seed)}
        if cells != {(n,s) for n in (4680,9360,14040) for s in range(4)}:
            raise ValueError(f'Incomplete history/step grid: {pid}, {seed}')
    (DATA/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    protocol = dict(prompt_file='prompts/MovieGenVideoBench_extended.txt',
        prompts=[dict(index=i,prompt_id=pid,text=text,split='train' if i<2 else 'val')
                 for i,(pid,text) in enumerate(zip(ids,prompts))],
        generation_seeds=[0,1], router_seed=0, train_steps=1000,
        train_keep=[.3,.2], evaluation_keep=[.3,.2,.1],
        checkpoint_selection='Fixed final 1000-step checkpoint, no selection on validation',
        layers=[4],heads=[0],latent_frames=12,query_tile=16,capture_window=32,
        capture_windows_per_event=4,splits=splits,
        capture_command=[sys.executable,'-m','streaming_sparse.capture.capture_runner',
            '--mode','capture','--frames','12','--limit','4','--seeds','0','1',
            '--layers','4','--heads','0','--query-tile','32','--tiles-per-event','4',
            '--backend','sdpa','--output',str(DATA)])
    (EXPERIMENT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    coverage = dict(events=len(records), snapshots=len({r['cache_version'] for r in records}),
        prompt_seed_pairs=[dict(prompt_id=pid,seed=seed,events=pairs[(pid,seed)],
            split='train' if pid in splits['train'] else 'val') for pid,seed in sorted(expected)],
        train_events=sum(r['prompt_id'] in splits['train'] for r in records),
        val_events=sum(r['prompt_id'] in splits['val'] for r in records),
        max_dense_backend_relative_error=max(r['dense_fp32_vs_backend_error'] for r in records))
    if coverage['max_dense_backend_relative_error']>1e-4:
        raise ValueError('Dense/backend QKV mismatch')
    (EXPERIMENT/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    cfg = json.loads((ROOT/'streaming_sparse/train/configs/p1.json').read_text())
    cfg.update(steps=1000,save_every=200,log_every=50)
    for name in ('ours','b3'):
        current = json.loads(json.dumps(cfg))
        if name == 'b3':
            current['router']['grouping']='spatial'
            current['collapse_weight']=0.
        path = EXPERIMENT/f'{name}_config.json'
        path.write_text(json.dumps(current,indent=2)+'\n')
        run(['streaming_sparse.train.train_router','--config',str(path),'--data',str(DATA),
             '--output',str(EXPERIMENT/name)],f'train_{name}.log')
    for split in ('train','val'):
        run(['streaming_sparse.evaluation.eval_attention','--data',str(DATA),
             '--checkpoint',str(EXPERIMENT/'ours/router.pt'),
             '--b3-checkpoint',str(EXPERIMENT/'b3/router.pt'),
             '--split',split,'--keep','0.3','0.2','0.1',
             '--output',str(EXPERIMENT/f'eval_{split}')],f'eval_{split}.log')
    run(['streaming_sparse.evaluation.eval_attention','--data',str(DATA),
         '--checkpoint',str(EXPERIMENT/'ours/initial_router.pt'),'--split','val',
         '--keep','0.3','0.2','0.1','--output',str(EXPERIMENT/'eval_val_initial')],
         'eval_val_initial.log')
    (EXPERIMENT/'completed.json').write_text(json.dumps(dict(completed_at=time.time()),indent=2))


if __name__=='__main__':
    main()
