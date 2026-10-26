"""Behavior-supervised grouping, mean-initialized bounded local Q pooling."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt30_behavior_grouping_20261026'
PREVIOUS=ROOT.parent/'opt29_query_aggregation_20261025'
PARENT=ROOT.parent/'opt20_all29_tile_finetune_20261025'
PILOTS=[1,14,24,29]
def option(g=64,k=128,mode='mean',r=8,grouping='none'):
 return dict(g=g,k=k,warp_q=16,mode=mode,r=r,refresh='fresh',candidate=.35,trees=False,budget=.2,grouping=grouping)
VARIANTS={
 'q32_full':option(32,64,'full',32),
 'q64_full':option(mode='full',r=64),
 'q64_mean8':option(),
 'q64_q_mean8':option(grouping='q'),
 'q64_k_mean8':option(grouping='k'),
 'q64_qk_mean8':option(grouping='qk'),
 'q64_rawqk_mean8':option(grouping='rawqk'),
 'q64_local8':option(mode='local'),
 'q64_qk_local8':option(mode='local',grouping='qk'),
 'q64_qk_mean16':option(r=16,grouping='qk'),
 'q64_qk_local16':option(mode='local',r=16,grouping='qk'),
 'q32_qk_local16':option(32,64,'local',16,'qk'),
}
def source_step(step,refresh):
 assert refresh=='fresh'
 return step
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'experiment.json').exists():return
 m=json.loads((PREVIOUS/'experiment.json').read_text())
 m.update(opt30_variants=VARIANTS,initialization=str(PREVIOUS),trainable=['KV prototypes and key/query projection','bounded within-subgroup Q pooling weights (zero initialization)','Q/K tree planes via offline Train-only behavior distillation'],grouping_training='Dense attention response profiles, balanced target-space partitions, linear classifiers distilled into existing depth6 tree; then held fixed during native-output finetuning',backbone_frozen=True,max_pilot_updates=2000,max_final_updates=6000)
 (ROOT/'experiment.json').write_text(json.dumps(m,indent=2))
 (ROOT/'reserved_gpus.json').write_text('[]')
 print(ROOT)
if __name__=='__main__':main()
