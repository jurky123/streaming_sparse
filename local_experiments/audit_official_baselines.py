"""Audit completed official-method videos, input identity and persistent KV bounds."""
import argparse,json,subprocess
from pathlib import Path
import torch
from .run_kv_history import sha,inspect_video,write_json
METHODS=['dense','lightforcing','dasa','forcingkv']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');a=p.parse_args();root=Path(a.experiment)
 manifest=json.loads((root/'manifest.json').read_text());assert manifest['methods']==METHODS
 cap=manifest['total_self_kv_storage_cap_bytes'];rows=[]
 for case in manifest['cases']:
  folder=root/case;inputs_sha=sha(folder/'shared_inputs.pt')
  for method in METHODS:
   result=json.loads((folder/f'{method}.result.json').read_text());assert result['completed']
   assert result['checkpoint_sha256']==manifest['checkpoints']
   assert result['shared_inputs_sha256']==inputs_sha
   assert result['chunks']==41 and result['actual_transition_noise_calls']==123
   video=inspect_video(folder/f'{method}.mp4');assert video['frames']==480 and video['duration']==30 and video['fps']==16 and video['width']==768 and video['height']==512
   trace=json.loads((folder/f'{method}.cache_trace.json').read_text())
   assert len(trace)==205 and set(r['chunk'] for r in trace)==set(range(41))
   assert all(r['self_kv_storage_bytes']<=cap for r in trace)
   calls=json.loads((folder/f'{method}.attention_trace.json').read_text())
   assert all(0<=r['layer']<30 and r['k_tokens']<=27648 for r in calls)
   assert set(r['layer'] for r in calls)==set(range(30))
   sparse=sum(r['sparse'] for r in calls)
   reduced=sum(r['timestep']>0 and r['q_tokens']<4608 for r in calls)
   if method=='lightforcing':assert sparse>0
   if method=='dasa':assert reduced>0
   rows.append(dict(case=case,method=method,video=video,cache_samples=len(trace),max_self_kv_bytes=max(r['self_kv_storage_bytes'] for r in trace),
                    sparse_calls=sparse,reduced_query_denoise_calls=reduced,checkpoint_sha256=result['checkpoint_sha256'],input_sha256=inputs_sha))
  comparison=inspect_video(folder/'comparison.mp4');assert comparison['frames']==480 and comparison['duration']==30
 preflight=Path('artifacts/official_baseline_preflight/dasa/dasa.latents.pt')
 before=torch.load(preflight,map_location='cpu',weights_only=True)
 after=torch.load(root/'sailboat/dasa.latents.pt',map_location='cpu',weights_only=True)[:,:before.shape[1]]
 difference=(before.float()-after.float()).abs()
 regression=dict(reference=str(preflight),chunks=int(before.shape[1]//3),exact_match=bool(torch.equal(before,after)),max_abs_difference=float(difference.max()))
 assert regression['exact_match'],regression
 source_patches={m:sha(root/f'{m}.compatibility.patch') for m in ['LightForcing','DASA','Forcing-KV']}
 write_json(root/'audit.json',dict(completed=True,rows=rows,dasa_no_overflow_prefix_regression=regression,
    physical_self_kv_cap_bytes=cap,source_patch_sha256=source_patches,
    assertions='12 videos and 3 comparisons have 480 frames / 30 seconds; shared model and input hashes; 123 saved transition draws; 205 forward samples per run obey persistent KV cap; sparse and full DASA region reuse activated; corrected DASA prefix matches unpatched official pipeline exactly.',
    scripts={p.name:sha(p) for p in Path('local_experiments').glob('*official*.py')}))
 print(json.dumps({'completed':True,'videos':len(rows),'dasa_prefix_regression':regression}))
if __name__=='__main__':main()
