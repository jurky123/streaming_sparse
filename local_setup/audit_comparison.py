"""Audit completed runs without executing GPU work or accepting partial results."""
import hashlib,json,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'artifacts/comparison_20261113'
METHODS=['dense','ours_latest_summary','ours_latest_summary_tail','ours_latest_rope','ours_latest_rope_tail','ours_latest_cached','ours_latest_cached_tail','ours_latest_stable','ours_latest_optimized','ours_latest_fused','ours_latest_fast','ours','dasa','pisa','solattn','svgear','lightforcing','forcingkv']
rows=[];missing=[];errors=[]
for case,gpu in [('horse_field','0'),('traffic_crossing','1')]:
 inputs=ROOT/case/'shared_inputs.pt'
 sha=hashlib.sha256(inputs.read_bytes()).hexdigest() if inputs.exists() else None
 checkpoint=None
 for method in METHODS:
  folder=ROOT/case/method;result=folder/'result.json'
  if not result.exists():missing.append(f'{case}/{method}');continue
  try:
   data=json.loads(result.read_text());assert data['status']=='complete'
   reps=data['repetitions'];assert len(reps)==2
   expected_gpu=gpu
   if data.get('paired_dense'):
    paired_path=ROOT/data['paired_dense'];paired=json.loads(paired_path.read_text())
    expected_gpu=paired['repetitions'][0]['physical_gpu']
    assert all(r['physical_gpu']==expected_gpu for r in paired['repetitions'])
    for ri,pr in enumerate(paired['repetitions']):
     assert pr['metadata']['inputs_sha256']==sha and pr['metadata']['checkpoint_sha256']==reps[0]['metadata']['checkpoint_sha256']
     ss=json.loads((paired_path.parent/f'rep{ri}.peers.json').read_text())['samples']
     assert len(ss)>=2 and all(not x.get('peers') and not x.get('error') for x in ss)
    assert json.loads((paired_path.parent/'cross_device_equivalence.json').read_text())['bitwise_equal']
   for index,r in enumerate(reps):
    assert r['rep']==index and r['physical_gpu']==expected_gpu
    meta=r['metadata'];assert meta['inputs_sha256']==sha
    assert (meta['height'],meta['width'],meta['chunks'])==(512,768,41)
    checkpoint=checkpoint or meta['checkpoint_sha256'];assert meta['checkpoint_sha256']==checkpoint
    if method in ['dense','dasa','lightforcing','forcingkv']:assert meta['config']['model_kwargs']['local_attn_size']==18
    elif method.startswith('ours_latest_'):
     implementations=dict(ours_latest_summary='SummaryLatestRuntime',ours_latest_summary_tail='SummaryTailLatestRuntime',ours_latest_fast='FastLatestRuntime',ours_latest_fused='FusedLatestRuntime',ours_latest_optimized='OptimizedLatestRuntime',ours_latest_stable='StableLatestRuntime',ours_latest_cached='CachedLatestRuntime',ours_latest_cached_tail='CachedTailLatestRuntime',ours_latest_rope='RopeLatestRuntime',ours_latest_rope_tail='RopeTailLatestRuntime')
     assert meta['history_pages']==360 and meta['implementation']==implementations[method]
     original=json.loads((ROOT/case/'ours/result.json').read_text())['repetitions'][0]['metadata']
     assert meta['sparse_checkpoint_sha256']==original['sparse_checkpoint_sha256']
     assert r['storage']['fast_attention_calls']>0 and r['storage']['route_graph_shapes' if method=='ours_latest_fast' else 'route_fused_shapes']==4
     if method not in ['ours_latest_fast','ours_latest_fused','ours_latest_summary','ours_latest_summary_tail']:assert r['storage']['summary_graph_shapes']==2 and r['storage']['summary_graph_calls']>0
     if method in ['ours_latest_cached','ours_latest_cached_tail','ours_latest_rope','ours_latest_rope_tail','ours_latest_summary','ours_latest_summary_tail']:assert r['storage']['group_graph_shapes']==1 and r['storage']['selection_graph_shapes']==4 and r['storage']['kv_reuse_calls']>0 and r['storage']['exact_kernel_stages']==2
     if method in ['ours_latest_rope','ours_latest_rope_tail','ours_latest_summary','ours_latest_summary_tail']:assert r['storage']['rope_calls']>0
     if method in ['ours_latest_summary','ours_latest_summary_tail']:assert r['storage']['fused_summary_calls']>0 and r['storage']['fused_summary_shapes']==2
    elif method=='ours':assert meta['history_pages']==360
    else:assert meta['history_latent_frames']==15 and meta['chunk_latent_frames']==3
    assert all(math.isfinite(r[k]) and r[k]>0 for k in ['generation_seconds','decode_seconds','total_generation_vae_seconds'])
    assert abs(r['total_generation_vae_seconds']-r['generation_seconds']-r['decode_seconds'])<1e-6
    samples=json.loads((folder/f'rep{index}.peers.json').read_text())['samples']
    assert len(samples)>=2 and all(not s.get('peers') and not s.get('error') for s in samples)
   video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(folder/'video.mp4')]))['streams'][0]
   assert (video['width'],video['height'],int(video['nb_frames']),video['r_frame_rate'])==(768,512,480,'16/1'),video
   if method!='dense':
    for window in ['early','middle','late']:
     q=data['quality']['windows'][window]
     assert all(math.isfinite(q[m]) for m in ['lpips','ssim','psnr'])
   repeats=[json.loads(p.read_text()) for p in sorted(folder.glob('rep*.repeatability.json'))]
   rows.append(dict(case=case,method=method,passed=True,video=video,repeatability=repeats))
  except Exception as exc:errors.append(dict(case=case,method=method,error=repr(exc)))
report=dict(passed=not missing and not errors,completed=len(rows),expected=2*len(METHODS),missing=missing,errors=errors,runs=rows)
(ROOT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
if not report['passed']:raise SystemExit(1)
