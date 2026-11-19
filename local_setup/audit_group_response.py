"""Strict full-protocol acceptance; no partial or silently dropped events."""
import json,itertools,math,subprocess,hashlib
from pathlib import Path
src=Path(__file__).resolve().parents[1];out=src/'artifacts/group_response_178f95a8';raw=Path('/tmp/group_response_results');groups=['G-K','G-VR','G-KVR','G-ORDER'];cases=['horse_field','traffic_crossing'];seeds=[101,303]
protocol=json.loads((out/'protocol.json').read_text());sha=protocol['weights']['artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'];rows=[]
for case,seed,c,l,s in itertools.product(cases,seeds,[2,5,8],[1,8,15,22,29],[0,2]):
 p=raw/'local'/f'{case}_s{seed}'/f'c{c}_l{l}_s{s}.json';x=json.loads(p.read_text());t=(17*l+13*c+7*s+seed)%72
 assert x['tiles']==[t,(t+36)%72] and x['heads']==12 and x['queries_per_tile']==64
 for g in groups:
  r=x['rows'][g];i=r['identity'];assert i['token_bijection'] and i['token_kv_exact'] and i['partial_unchanged'];assert i['dense_nmse']<2e-10 and i['dense_max_abs']<1e-3
  assert i['history_pages']==(144 if c==2 else 338) and i['regrouped_pages']==(144 if c==2 else 288) and i['partial_pages']==(0 if c==2 else 50)
  assert r['exact_pages_per_tile_head']==(21 if c==2 else 41) and r['representation_exact_pages']==1
  for panel in ['representation','mass','risk','uniform_pool_diagnostic']:assert all(math.isfinite(r[panel][n]) for n in ['nmse','log_denominator_mae','denominator_relative_mae'])
 rows.append(str(p.relative_to(raw)))
assert len(list((raw/'local').glob('*/*.json')))==120
videos=[]
for case,seed,g,level in itertools.product(cases,seeds,groups,['Dense','L2','L3']):
 p=raw/'video'/f'{case}_s{seed}'/g/level;x=json.loads((p/'result.json').read_text());assert x['status']=='complete' and x['weights']==sha and x['history_pages']==338 and x['quality_window']==[21,101] and x['frozen']
 v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(p/'video.mp4')]))['streams'][0];assert (v['width'],v['height'],int(v['nb_frames']),v['r_frame_rate'])==(768,512,105,'16/1')
 d=json.loads((p/'diagnostics.json').read_text());assert len(d['identity_checks'])==261 and all(r['bijective'] and r['kv_matches_token_identity'] for r in d['identity_checks'])
 if level!='Dense':assert all(math.isfinite(x['quality'][n]) for n in ['lpips','ssim','psnr']) and math.isfinite(x['latent_nmse'])
 videos.append(dict(case=case,seed=seed,group=g,level=level,video_sha256=hashlib.sha256((p/'video.mp4').read_bytes()).hexdigest(),latent_sha256=hashlib.sha256((p/'latents.pt').read_bytes()).hexdigest()))
a=json.loads((out/'adaptation_decision.json').read_text());assert a['required'] is False,'Adaptation required and not yet completed'
assert json.loads((raw/'preflight.json').read_text())['passed'];assert json.loads((out/'baseline_source_audit.json').read_text())['passed'];assert json.loads((raw/'paired_group_cost.json').read_text())['status']=='complete'
result=dict(passed=True,local_events=120,candidate_identity_checks=480,closed_loop_runs=48,closed_loop_commit_identity_checks=48*261,all_expected_present=True,adaptation_status='not_triggered_not_run',videos=videos)
(out/'final_audit.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='videos'})
