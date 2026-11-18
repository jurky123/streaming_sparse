"""Collect same-device production runs, trajectory checks and CUDA timeline totals."""
import json
from pathlib import Path
from statistics import median
import torch

root=Path(__file__).parent
def load(path):return json.loads(path.read_text())
def tensor(path):return torch.load(path/'latents.pt',map_location='cpu',weights_only=True)
def nmse(a,b):return float((a.float()-b.float()).square().sum()/b.float().square().sum())
torch.set_num_threads(4)
dense_path=Path('/tmp/jzk_cuda_samecard_dense')
dense=tensor(dense_path)
dense_timing=load(dense_path/'result.json')
refs={'base':'cuda_tier_triton','intersection':'resroute_deploy_intersection','step':'resroute_deploy_step'}
result={'prompt':'city_walk','seed':701,'frames':105,'gpu':'A100-SXM4-40GB GPU 6','tiers':{}}
result['dense']=dict(seconds=dense_timing['seconds'],median_s=dense_timing['median_seconds'])
for tier,reference in refs.items():
    path=Path('/tmp/jzk_cuda_samecard_'+tier)
    r=load(path/'result.json');z=tensor(path);old=tensor(Path('/tmp/jzk_'+reference))
    stat=r['runtime_stats']
    result['tiers'][tier]=dict(seconds=r['seconds'],median_s=r['median_seconds'],
        speedup_vs_dense=dense_timing['median_seconds']/r['median_seconds'],exact_density=stat['exact_density'],cuda_calls=stat['packed_attention_calls'],
        direct_commits=stat['direct_layout_commits'],packed_kv_bytes=stat['packed_kv_bytes'],
        counters=stat['counters'],peak_allocated_bytes=r['peak_allocated_bytes'],
        reference=reference,reference_bitwise=torch.equal(z.view(torch.uint8),old.view(torch.uint8)),
        dense_nmse=nmse(z,dense),reference_dense_nmse=nmse(old,dense))
for name in ('paired_video','multi_video','release_pairs','triton_pairs'):
    rows=load(root/name/'base.json');d=[1000*(r['old_s']-r['new_s']) for r in rows]
    result[name]=dict(n=len(rows),faster=sum(x>0 for x in d),median_ms=median(d),
        range_ms=[min(d),max(d)],all_bitwise=all(r['bitwise'] for r in rows))
rows=load(root/'timeline.json')
labels={'main':'exact(','current_kv':'pack_range','summary':'encode','guard':'retry','clean_kv_commit':'commit_tile('}
t={label:sum(ms for row in rows for name,ms in row['by_name_ms'].items()
          if (name.startswith(prefix) or name.startswith('void '+prefix))) for label,prefix in labels.items()}
assert all(t.values()),t
t['attention_path_ms']=sum(t[k] for k in ('main','current_kv','summary','guard'))
result['device_ms']=t
(root/'release.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
