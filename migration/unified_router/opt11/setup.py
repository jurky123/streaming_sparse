import json,hashlib
from pathlib import Path
from ..common import atomic_json,event_key,sha
BASE=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024').resolve()
ROOT=Path('streaming_sparse/v10/experiments/opt11_finetune_20261025').resolve()
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'experiment.json').exists():print(ROOT);return
 m=json.loads((BASE/'experiment.json').read_text());rows=[]
 for source in m['rows']:
  if not source['history_tokens']:continue
  if source['split']=='train':
   offset=(m['splits']['train'].index(source['prompt_id'])+source['generation_seed']+source['chunk'])%4
   if source['step']not in [offset,(offset+2)%4]:continue
  row=dict(source);blocks=[]
  for i in row['query_indices']:
   block=i//128
   if block<36 and block not in blocks:blocks.append(block)
   if len(blocks)==2:break
  blocks.sort();ids=[i for block in blocks for i in range(block*128,(block+1)*128)]
  targets=[]
  for group in range(8):
   candidates=[i for i in row['query_indices']if i in ids[group*32:(group+1)*32]]
   original=candidates[0]if candidates else ids[group*32+(source['step']*7+group)%32]
   targets.append(ids.index(original))
  row.update(group_query_indices=ids,target_query_positions=targets);rows.append(row)
 m.update(parent_experiment=str(BASE),rows=rows,groups=['K32-Set'],training_seeds=[0],variants=['Compact','Aggressive'],training_objective='expected_actual_deployment_attention_nmse',best_metric='actual_deployment_attention_nmse',learning_rate=1e-4,max_updates=200,validate_every=50,batch_events=8,initialization='parent seed0 best head; all head weights fine-tuned; normalization, prototypes and backbone frozen',group_cache=str(ROOT/'group_queries'),feature_cache=str(ROOT/'features'),counterfactual='one target query per original group32 changes action; other31 keep initial pretrained deployment masks; peer masks are refreshed under current head during actual validation',aggressive=dict(skip_log2_threshold=0.,pv_dtype='float16',qk='int8',route='Opt10 Compact'),monitor_events='all96 Val, all256 captured queries per event',capture_query_design='two original complete 128-query tiles preserving canonical candidate hash and eight original group32 quantization/skip units; final incomplete tile covered by end-to-end evaluation',test_used=False)
 atomic_json(ROOT/'experiment.json',m)
 old=json.loads((BASE/'optimization10/manifest.json').read_text());frozen={**old['code'],**old['prior_unchanged']}
 for f,h in frozen.items():assert sha(f)==h,f
 files={}
 for layer in m['layers']:
  for f in [BASE/f'action_basis/layer{layer:02d}.pt',BASE/f'P3_training/layer{layer:02d}/K32-Set/seed0/best.pt']:files[str(f)]=sha(f)
 atomic_json(ROOT/'parent_lock.json',dict(source=frozen,parameters=files));print(ROOT, 'train',sum(r['split']=='train'for r in rows),'val',sum(r['split']=='val'for r in rows))
if __name__=='__main__':main()
