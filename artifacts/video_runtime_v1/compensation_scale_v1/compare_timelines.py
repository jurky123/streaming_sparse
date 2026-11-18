"""Estimate the direct attention cost using matched native and sparse device timelines."""
import json
from pathlib import Path
from local_experiments.video_runtime_v1.theoretical_cost import attention,sparse_main

d=Path(__file__).parent
base=json.loads((d/'base_timeline.json').read_text())
dense=json.loads((d/'dense_timeline.json').read_text())
assert len(base)==len(dense)==9

def flash(row):
    return sum(t for k,t in row['by_name_ms'].items() if 'pytorch_flash::flash_fwd_kernel' in k)

rows=[]
for c in range(2,9):
    assert c==base[c]['chunk']==dense[c]['chunk']
    p=min(c+1,6)*72
    ops=sparse_main(p)
    padded=((p+15)//16)*16
    # QK plus two AVs for exact pages; representative QK, two mean AVs
    # and two detail AVs padded from rank eight to sixteen.
    fused_ops=(1.5*ops['exact']+(ops['representative_qk']+2*ops['full_mean']+4*ops['detail_av'])*padded/p)*116
    dense_ops=attention(p)*116
    saved=flash(dense[c])-flash(base[c])
    fused=base[c]['by_name_ms']['_tail']
    at_dense_rate=saved*fused_ops/dense_ops
    rows.append(dict(chunk=c,pages=p,displaced_dense_ms=saved,fused_ms=fused,dense_flops=dense_ops,
                     main_fused_flops=fused_ops,dense_effective_tflops=dense_ops/saved/1e9,
                     fused_effective_tflops=fused_ops/fused/1e9,at_dense_rate_ms=at_dense_rate))
total={key:sum(row[key] for row in rows) for key in ('displaced_dense_ms','fused_ms','dense_flops','main_fused_flops','at_dense_rate_ms')}
total['estimated_difference_ms']=total['fused_ms']-total['at_dense_rate_ms']
(d/'attention_budget.json').write_text(json.dumps(dict(chunks=rows,total=total),indent=2))
print(total)
