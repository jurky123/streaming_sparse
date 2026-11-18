"""Compare the same 116 displaced attention calls per sparse chunk."""
import json
from pathlib import Path
from local_experiments.video_runtime_v1.theoretical_cost import attention, common, sparse_main

root = Path(__file__).parent
def read(name):
    return json.loads((root / name).read_text())

def flash_ms(chunk):
    return sum(ms for name, ms in chunk['by_name_ms'].items()
               if 'pytorch_flash::flash_fwd_kernel' in name)

base = read('base_timeline.json')
dense = read('dense_timeline.json')
rows = []
dense_total = 0
sparse_total = {route: 0 for route in (4,5)}
for index,(b,d) in enumerate(zip(base,dense,strict=True)):
    pages = 72 * min(index + 1,6)
    dense_ops = attention(pages)
    dense_total += 150 * (common + dense_ops)
    if index < 2:
        for route in sparse_total:
            sparse_total[route] += 150 * (common + dense_ops)
        continue
    count = pages * 15 // 100
    padded = (pages + 15) // 16 * 16
    parts = sparse_main(pages)
    exact = attention(count) * 1.5
    compensation = (parts['representative_qk'] + 2*parts['full_mean'] + 4*parts['detail_av']) * padded / pages
    ops = exact + compensation
    for route in sparse_total:
        sparse_total[route] += 150*common + 34*dense_ops + 116*(
            ops + parts['detail_weight'] + parts['detail_reconstruction'] + route*1e9)
    dense_ms = flash_ms(d) - flash_ms(b)
    rows.append(dict(chunk=index,pages=pages,count=count,calls=116,exact_gflop=exact/1e9,
                     compensation_gflop=compensation/1e9,fused_tflop=116*ops/1e12,
                     dense_tflop=116*dense_ops/1e12,dense_ms=dense_ms,
                     fused_ms=b['by_name_ms']['_tail'],reference_ms=dense_ms*ops/dense_ops))
total = {name: sum(r[name] for r in rows) for name in
         ('fused_tflop','dense_tflop','dense_ms','fused_ms','reference_ms')}
dense_s = read('dense_release.json')['median_seconds']
result = dict(rows=rows,total=total,whole_generation=[
    dict(route_gflop=r,speedup=dense_total/s,s=dense_s*s/dense_total) for r,s in sparse_total.items()])
(root/'budget.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['total'],result['whole_generation'])

# Keep the module names used in the previous profile, and retain unseen kernel names.
import csv
from collections import Counter
old_mapping = root.parents[2]/'docs/research/evidence/video_runtime_v1/PROFILE_KERNELS_105.csv'
with old_mapping.open() as handle:
    labels = {r['CUDA kernel / event']:r['模块'] for r in csv.DictReader(handle)}
def aggregate(timeline):
    times = Counter()
    for chunk in timeline:
        times.update(chunk['by_name_ms'])
    return times

a,b = aggregate(base),aggregate(dense)
modules = {}
with (root/'kernel_profile.csv').open('w') as handle:
    writer = csv.writer(handle)
    writer.writerow(['module','kernel','base_ms','dense_ms'])
    for k in sorted(a.keys() | b.keys(),key=lambda k:-a[k]):
        name = labels.get(k,'其他设备核（新增名称）')
        writer.writerow([name,k,a[k],b[k]])
        sums = modules.setdefault(name,[0,0])
        sums[0] += a[k]
        sums[1] += b[k]
with (root/'module_profile.csv').open('w') as handle:
    writer = csv.writer(handle)
    writer.writerow(['module','base_ms','dense_ms'])
    for name,times in sorted(modules.items(),key=lambda item:-item[1][0]):
        writer.writerow([name,*times])
