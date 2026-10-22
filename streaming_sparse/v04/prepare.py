"""Read-only inventory and command plan. Never captures QKV or starts training."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import sys
from .identity import source_identity
from .train import configuration
from ..v02.evaluation_journal import atomic_json
from .paths import protect_output

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
EXISTING=ROOT/'streaming_sparse/v03/rounds/20261016_165725/data/stage_a'
CELL=('prompt_id','seed','layer_id','head_id','history_tokens','step_id')


def plan(existing,output):
    protocol=json.loads((HERE/'configs/protocol.json').read_text())
    existing=Path(existing).resolve()
    rows=[json.loads(line) for line in (existing/'events.jsonl').read_text().splitlines()]
    if json.loads((existing/'splits.json').read_text())!=protocol['splits']:
        raise ValueError('Capture source prompt split mismatch')
    cells={tuple(r[k] for k in CELL) for r in rows}
    if len(cells)!=len(rows):raise ValueError('Duplicate capture cells')
    expected=set(itertools.product([p['prompt_id'] for p in protocol['prompts']],(0,1),(4,14,24),
                                   range(12),(4680,9360,14040),range(4)))
    if cells-expected:raise ValueError('Unexpected capture cells')
    files={kind:{existing/r[kind] for r in rows} for kind in ('history','event')}
    missing={kind:[str(p) for p in paths if not p.is_file()] for kind,paths in files.items()}
    sizes=sum(p.stat().st_size for paths in files.values() for p in paths if p.is_file())
    heads=sorted({r['head_id'] for r in rows})
    expected_existing=set(itertools.product([p['prompt_id'] for p in protocol['prompts']],(0,1),(4,14,24),
                                            heads,(4680,9360,14040),range(4)))
    source_complete=cells==expected_existing and not any(missing.values())
    if not source_complete:raise ValueError('Existing head subset is not complete/readable')
    needed=sorted(set(range(12))-set(heads))
    full=HERE/'data/full'
    env=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    commands=[]
    def add(identifier,stage,argv,gpu=None,depends=(),**extra):
        commands.append(dict(id=identifier,stage=stage,argv=argv,gpu_physical_index=gpu,
            environment={'CUDA_VISIBLE_DEVICES':str(gpu) if gpu is not None else ''},
            depends_on=list(depends),**extra))
    sources=[str(existing)]
    capture_ids=[]
    if needed:
        for gpu,generation_seed in enumerate((0,1)):
            supplemental=HERE/f'data/supplemental_generation_seed{generation_seed}'
            argv=[sys.executable,'-m','streaming_sparse.v02.capture','--mode','capture',
                '--prompts',str(HERE/'configs/capture_prompts.txt'),'--limit','24','--frames','12',
                '--seeds',str(generation_seed),'--layers','4','14','24','--heads',*map(str,needed),
                '--query-tile','16','--tiles-per-event','8','--sampling-seed','2026',
                '--backend','sdpa','--output',str(supplemental)]
            identifier=f'capture_generation_seed{generation_seed}'
            add(identifier,'capture_missing_heads',argv,gpu=gpu,generation_seed=generation_seed)
            sources.append(str(supplemental));capture_ids.append(identifier)
    add('merge','merge_immutable_sources',[sys.executable,'-m','streaming_sparse.v03.merge',
                    '--sources',*sources,'--output',str(full)],depends=capture_ids)
    last={}
    for gpu in (0,1):
        sanity=HERE/f'reports/real_full12_cuda_gpu{gpu}.json'
        identifier=f'full12_correctness_gpu{gpu}'
        add(identifier,'full12_correctness_zero_updates',[sys.executable,'-m','streaming_sparse.v04.sanity',
            '--data',str(full),'--device','cuda','--full-head-coverage','--output',str(sanity)],gpu=gpu,depends=['merge'])
        last[gpu]=identifier
    run_index=0
    for keep in (.3,.2):
        for seed in (0,1,2):
            gpu=run_index%2
            run_index+=1
            run=HERE/f'runs/keep{int(keep*100)}/seed{seed}'
            sanity=HERE/f'reports/real_full12_cuda_gpu{gpu}.json'
            identifier=f'train_keep{int(keep*100)}_seed{seed}'
            add(identifier,'paired_train',[sys.executable,'-u','-m','streaming_sparse.v04.train',
                '--data',str(full),'--output',str(run),'--keep',str(keep),'--seed',str(seed),
                '--sanity-gate',str(sanity)],gpu=gpu,depends=[last[gpu]],train_keep=keep,router_seed=seed)
            last[gpu]=identifier
            for split in ('train','val','test'):
                identifier=f'eval_keep{int(keep*100)}_seed{seed}_{split}'
                add(identifier,'paired_evaluate_all_three_keeps',[sys.executable,'-u','-m','streaming_sparse.v04.evaluate',
                    '--data',str(full),'--checkpoint',str(run/'final.pt'),'--split',split,
                    '--output',str(run/f'evaluation_{split}')],gpu=gpu,depends=[last[gpu]],split=split)
                last[gpu]=identifier
    add('report','aggregate_three_router_seeds',[sys.executable,'-m','streaming_sparse.v04.report',
        '--runs',str(HERE/'runs'),'--output',str(HERE/'reports/final_three_seed_summary.json')],depends=list(last.values()))
    audit=dict(status='prepared_not_started',source_sha256=source_identity(),existing_data=str(existing),
        existing_heads=heads,existing_events=len(rows),existing_snapshots=len({r['cache_version'] for r in rows}),
        missing_heads=needed,missing_events=len(expected-cells),target_events=20736,target_snapshots=5184,
        target_query_tiles=165888,existing_reference_missing={k:len(v) for k,v in missing.items()},
        can_reuse_existing_heads=source_complete,data_coverage_complete=not needed,ready_for_full_training=False,
        full12_real_correctness_required=True,compute_started=False,commands_queued=False,
        paired_training_runs=6,arm_results=12,formal_updates_per_pair=6000,optimizer_updates_total=36000,
        train_snapshots=3456,snapshot_draws_per_run=24000,
        maximum_unique_training_tile_fraction=24000/(3456*8),
        storage=dict(existing_raw_bytes=sizes,estimated_full_raw_bytes=sizes*12/len(heads),
                     estimated_additional_with_copy_merge_bytes=sizes*(24/len(heads)-1),
                     estimate_definition='Linear head-count extrapolation; merge copies all tensors to a new root',
                     free_bytes=shutil.disk_usage(ROOT).free),
        execution=dict(cwd=str(ROOT),environment=env,clear_proxy_variables=[
            'http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'],
            physical_gpu_workers=[0,1],max_simultaneous_gpu_jobs=2,max_jobs_per_gpu=1,
            placement='Entire shared frontend and A/B branches stay together on one GPU',commands=commands))
    output=protect_output(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='execution'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing-data',default=str(EXISTING))
    p.add_argument('--output',default=str(HERE/'reports/preparation.json'))
    args=p.parse_args();plan(args.existing_data,args.output)


if __name__=='__main__':main()
