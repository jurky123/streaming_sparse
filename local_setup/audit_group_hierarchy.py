"""Strict audit of same-event inputs, six grouping conditions and five L2 layers."""
import json,math,itertools,hashlib,subprocess
from pathlib import Path
SRC=Path(__file__).resolve().parents[1];OUT=SRC/'artifacts/group_hierarchy_178f95a8';ROOT=Path('/tmp/group_hierarchy_results');GROUPS=['G-ORDER','G-K','G-K-joint','G-KVR','G-K3-V2','G-K3-V2-Kinner'];LAYERS=[1,8,15,22,29]
def load(p):return json.loads(p.read_text())
pre=load(ROOT/'preflight.json');assert pre['passed'] and pre['mixed_matches_composed_original_balanced'] and pre['inner_variant_preserves_page_membership']
checks=[]
for c,s in itertools.product(['horse_field','traffic_crossing'],[101,303]):
 key=f'{c}_s{s}';assert load(ROOT/'restoration'/f'{key}.json')['latent_bitwise_equal']
 for ch,l,step in itertools.product([2,5,8],LAYERS,[0,2]):
  name=f'c{ch}_l{l}_s{step}.json';x=load(ROOT/'local'/key/name);assert load((ROOT/'local'/key/name).with_suffix('.reuse'))['passed']
  t=(17*l+13*ch+7*step+s)%72;assert x['tiles']==[t,(t+36)%72] and x['heads']==12 and x['queries_per_tile']==64
  snap=OUT/'qkv'/key/Path(name).with_suffix('.pt');assert snap.exists();assert hashlib.sha256(snap.read_bytes()).hexdigest()==x['qkv_sha256']
  for g in GROUPS:
   r=x['rows'][g];i=r['identity'];assert i['token_bijection'] and i['token_kv_exact'] and i['partial_unchanged']
   assert i['history_pages']==(144 if ch==2 else 338) and i['regrouped_pages']==(144 if ch==2 else 288) and i['partial_pages']==(0 if ch==2 else 50)
   assert i['dense_nmse']<2e-10 and i['dense_max_abs']<1e-3;assert r['exact_pages_per_tile_head']==(21 if ch==2 else 41)
   for p in ['representation','mass','risk']:
    assert all(math.isfinite(r[p][n]) for n in ['nmse','page_log_denominator_mae','history_page_log_denominator_mae','unselected_page_log_denominator_mae','log_denominator_mae'])
  checks.append(dict(event=str(snap.relative_to(OUT)),sha256=x['qkv_sha256']))
assert len(list((ROOT/'local').glob('*/*.json')))==120
videos=[];weights=set();reused=0
for c,s,g,level in itertools.product(['horse_field','traffic_crossing'],[101,303],GROUPS,['Dense','L3']+['L2_'+str(l) for l in LAYERS]):
 p=ROOT/'video'/f'{c}_s{s}'/g/level;x=load(p/'result.json');assert x['status']=='complete' and x['frozen'] and x['history_pages']==338 and x['quality_window']==[21,101];weights.add(x['weights'])
 assert x['level']==level
 if 'reused_from' in x:
  reused+=1;assert g in ['G-ORDER','G-K','G-KVR'] and level in ['Dense','L3','L2_15']
  original=Path(x['reused_from']);assert hashlib.sha256((p/'latents.pt').read_bytes()).digest()==hashlib.sha256((original/'latents.pt').read_bytes()).digest()
 d=load(p/'diagnostics.json');assert len(d['identity_checks'])==261 and all(r['bijective'] and r['kv_matches_token_identity'] for r in d['identity_checks'])
 v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(p/'video.mp4')]))['streams'][0];assert (v['width'],v['height'],int(v['nb_frames']),v['r_frame_rate'])==(768,512,105,'16/1')
 if level!='Dense':assert all(math.isfinite(x['quality'][n]) for n in ['lpips','ssim','psnr']) and math.isfinite(x['latent_nmse'])
 videos.append(dict(case=c,seed=s,group=g,level=level,reused='reused_from' in x))
expected_weight=load(OUT/'protocol.json')['weights']['artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'];assert weights=={expected_weight} and reused==36
cost=load(ROOT/'paired_group_cost.json');assert cost['status']=='complete' and cost['parameters_unchanged'];assert len(cost['rows'])==18
for r in cost['rows']:
 assert len(r['trials'])==10
 for t in r['trials']:assert abs(t['commit_total_ms']-t['group_ms']-t['summary_ms']-t['write_ms'])<.02
result=dict(passed=True,restored_cases_bitwise=4,local_events=120,candidate_token_identity_checks=720,video_results=168,reused_videos=36,new_videos=132,L2_layers=LAYERS,cost_rows=18,no_training=True,qkv=checks,videos=videos)
(OUT/'final_audit.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ['qkv','videos']})
