from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt29_query_aggregation_20261025'
PARENT=ROOT.parent/'opt20_all29_tile_finetune_20261025'
PILOTS=[1,14,24,29]
def option(g=64,k=128,mode='learned',r=8,refresh='fresh',candidate=.35,trees=False):
 return dict(g=g,k=k,warp_q=16,mode=mode,r=r,refresh=refresh,candidate=candidate,trees=trees,budget=.2)
VARIANTS={
 'q32_full':option(32,64,'full',32), 'q64_full':option(64,128,'full',64),
 'q32_a8':option(32,64,r=8), 'q32_a16':option(32,64,r=16),
 'q64_a4':option(r=4),'q64_a8':option(r=8),'q64_a16':option(r=16),
 'q64_mean8':option(mode='mean',r=8),'q64_sample8':option(mode='sample',r=8),
 'q64_a8_pair':option(r=8,refresh='pair'),'q64_a16_pair':option(r=16,refresh='pair'),
 'q64_a8_early':option(r=8,refresh='early'),'q64_a16_c25':option(r=16,candidate=.25),
 'q64_a8_tree':option(r=8,trees=True),'q64k64_a8':option(64,64,r=8)}
def source_step(step,refresh):
 if refresh=='fresh':return step
 if refresh=='pair':return step-step%2
 if refresh=='early':return min(step,1)
 raise ValueError(refresh)
def main():
 from ..common import atomic_json,event_key,sha
 from streaming_sparse.v08.replay import ReplayDataset
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'experiment.json').exists():print(ROOT);return
 m=json.loads((PARENT/'experiment.json').read_text());m.update(opt29_variants=VARIANTS,pilots=PILOTS,query_cache=str(ROOT/'full_queries'),max_pilot_updates=2000,max_final_updates=6000,validate_every=100,checkpoint_every=50,learning_rate_router=3e-5,learning_rate_query=1e-4,batch_events=2,minimum_updates=1000,patience=8,trainable=['Q assignment weights/biases','KV prototype weights/biases','64D key/query projection','optional alternating Q/K trees (Train proposals only)'],frozen=['backbone','value basis'],best_metric='actual_native_attention_nmse_on_96_Val_events',budgets=dict(actual_attention=.2),training_forward='Opt28 native INT8 QK FP16 PV with new routing; hard output NMSE; FP32 biased gate STE backward',test_used=False)
 original={r['sample_id']:r for r in m['rows']};capture=[];allrows=[]
 for split in ['train','val']:
  ds=ReplayDataset(split,require_complete=True);allowed=set(m['splits'][split])
  for i,row in enumerate(ds.rows):
   if row['prompt_id']not in allowed or row['history_tokens']==0:continue
   # Same prompt/seed/chunk owners as the original training/validation dataset.
   if not any(x['trajectory_id']==row['trajectory_id']and x['chunk']==row['chunk']for x in m['rows']):continue
   row=dict(row,replay_index=i);allrows.append(row)
   if row['sample_id']not in original:capture.append(row)
   else:
    for layer in range(1,30):
     src=PARENT/'full_queries'/event_key(row)/f'layer{layer:02d}.pt';assert src.exists()
     dst=ROOT/'full_queries'/event_key(row)/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
     if not dst.exists():dst.symlink_to(src)
 m['sequence_rows']=allrows;atomic_json(ROOT/'experiment.json',m)
 cm=dict(m,rows=capture,layers=list(range(1,30)));atomic_json(ROOT/'capture_manifest.json',cm)
 atomic_json(ROOT/'source_lock.json',json.loads((PARENT/'chosen_checkpoints.json').read_text()))
 atomic_json(ROOT/'plan.json',dict(variants=VARIANTS,pilot_jobs=len(VARIANTS)*4,pilot_layers=PILOTS,pilot_updates=2000,final_updates=6000,automatic_promotion=['q32_full control always','best fresh learned route','best reused-weight route if valid'],selection='96-event Val actual output NMSE and measured full-module speed; do not select on training proxy',video='All29 sparse,layer0/firstchunk Dense; Dense and frozen Q32K64 controls; 480p/720p, decoded comparisons',missing_query_events=len(capture),existing_events=len(original),training_updates='pilot min1000,max2000; finalists up to6000,plateau stop after min2000',failure_policy='one attempt per job; fail visibly and continue independent jobs, no automatic retry loop'))
 print(ROOT,'capture',len(capture),'variants',len(VARIANTS))
if __name__=='__main__':main()
