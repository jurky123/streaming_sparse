"""Use the original per-window Dense measurements and the executed MMA count."""
from pathlib import Path
import json
from statistics import median
import torch

root=Path(__file__).parent
def read(path):
    return json.loads(path.read_text())
summary={}
rows=[]
for capture in read(root.parent/'fixed15_baseline_v1/budget.json')['rows']:
    pages=capture['pages']
    count=capture['count']
    warp_mmas=192*count+102*((pages+15)//16-1)
    row={'pages':pages,'calls':capture['calls']}
    for name,ops in [('release',warp_mmas),('first',warp_mmas-32)]:
        work=ops*3456*4096*capture['calls']/1e12
        row[name]={'tflop':work,'reference_ms':work*capture['dense_ms']/capture['dense_tflop']}
    rows.append(row)
summary['by_chunk']=rows
for name in ('release','first'):
    trace=read(root/f'timeline_{name}.json')
    assert len(trace)==9
    times={prefix:sum(ms for row in trace for kernel,ms in row['by_name_ms'].items() if kernel.startswith(prefix))
           for prefix in ('exact(','prepare(','retry')}
    work=sum(row[name]['tflop'] for row in rows)
    ref=sum(row[name]['reference_ms'] for row in rows)
    main=times['exact(']
    summary[name]=dict(tflop=work,reference_ms=ref,main_ms=main,main_tflops=work*1000/main,
                       gap_ms=main-ref,prepare_ms=times['prepare('],retry_ms=times['retry'],full_ms=sum(times.values()))
for which,size in [('video',8),('multi',10)]:
    records=read(root/f'first_{which}.json')
    assert len(records)==size and all(row['bitwise'] for row in records)
    if which=='multi': assert len({(r['prompt'],r['seed']) for r in records})==10
    saves=[1000*(r['baseline_s']-r['candidate_s']) for r in records]
    summary[which]=dict(faster=sum(x>0 for x in saves),pairs=size,median_ms=median(saves),range_ms=[min(saves),max(saves)],
                        speedup=median(r['baseline_s']/r['candidate_s'] for r in records),bitwise=True)
summary['tiers']={}
dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float()
for name in ('base','intersection','step','formal_base'):
    path=Path('/tmp/jzk_anchorhalf_'+name)
    previous=Path('/tmp/jzk_lastpage_tier_'+name.replace('formal_',''))
    record=read(path/'result.json')
    z=torch.load(path/'latents.pt',weights_only=True)
    assert torch.equal(z.view(torch.uint8),torch.load(previous/'latents.pt',weights_only=True).view(torch.uint8))
    calls=record['runtime_stats']['packed_attention_calls']
    assert calls==(812 if name.endswith('base') else 0)
    summary['tiers'][name]=dict(seconds=record['seconds'],median_seconds=record['median_seconds'],calls=calls,
                                dense_nmse=float((z.float()-dense).square().sum()/dense.square().sum()),bitwise=True)
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
