import json,os
from pathlib import Path
from ..common import atomic_json,sha,event_key
ROOT=Path('streaming_sparse/v10/experiments/opt20_all29_tile_finetune_20261025').resolve()
PARENT=Path('streaming_sparse/v10/experiments/opt18_continue_frontier_20261025').resolve()
KERNELS=Path('streaming_sparse/v10/experiments/opt19_tile_kernels_20261025').resolve()
PILOTS=(1,14,24,29)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'experiment.json').exists():print(ROOT);return
    m=json.loads((PARENT/'experiment.json').read_text())
    m.update(parent_opt18=str(PARENT),parent_opt19=str(KERNELS),layers=list(range(1,30)),
        variants={name:dict(candidate_fraction=.35,representatives=8,tile_budget=budget) for name,budget in [('Tile15',.15),('Tile20',.20)]},
        full_query_cache=str(ROOT/'full_queries'),max_updates=2000,min_updates=1000,
        learning_rate_router=3e-5,warmup_updates=50,validate_every=100,checkpoint_every=25,
        batch_events=2,tree_refresh_every=0,gate_temperature=.5,patience_validations=6,
        initialization='Opt18 Fast35 best for pilot layers1/14/24/29; original per-layer prototypes and Train-only grouping calibration for other25 layers; both tile budgets share the same initial weights',
        trainable=['all current/history prototype weights and biases','key projection first64'],
        frozen=['backbone','value basis','Q/K grouping trees','tile budgets','INT8 QK / BF16 PV attention'],
        training_objective='actual Opt19 shared32 tile-kernel attention NMSE against Dense; full4680Q forward; original128 sampled loss queries',
        backward='biased STE through shared tile probability aggregation and coarse candidate gate; dense page-attention Jacobian only in backward',
        best_metric='actual_deployment_attention_nmse',parallelism='8 GPU workers, dynamic queue of29layers x2tile budgets',
        evaluation_scope='all layers1..29 sparse; layer0 and first chunk Dense; full closed-loop videos before and after fine-tuning',
        validation_extra_candidate_fractions=[])
    atomic_json(ROOT/'experiment.json',m)
    locks=json.loads((KERNELS/'final_source_manifest.json').read_text())
    reused=0
    for layer in PILOTS:
        f=PARENT/f'training/layer{layer:02d}/Fast35/best.pt';locks[str(f)]=sha(f)
    for row in m['rows']:
        for layer in m['layers']:
            raw=Path(m['cache'])/'events'/event_key(row)/f'layer{layer:02d}.pt'
            assert raw.exists(),raw
            if layer in PILOTS:
                source=Path(json.loads((PARENT/'experiment.json').read_text())['full_query_cache'])/event_key(row)/f'layer{layer:02d}.pt'
                assert source.exists(),source
                dest=ROOT/'full_queries'/event_key(row)/f'layer{layer:02d}.pt';dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.symlink_to(source)
                reused+=1
    for f,digest in locks.items():assert sha(f)==digest,f
    atomic_json(ROOT/'parent_lock.json',locks)
    atomic_json(ROOT/'run_status.json',dict(status='prepared',layers=29,training_jobs=58,reused_full_query_files=reused,new_full_query_files=25*len(m['rows']),video_e2e='pending'))
    print(ROOT)

if __name__=='__main__':main()
