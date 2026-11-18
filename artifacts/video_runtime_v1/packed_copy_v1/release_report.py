"""Check the released trajectories, clock samples and the QK/AV operation model."""
import csv,json
from pathlib import Path
from statistics import median
import torch

root=Path(__file__).parent
def load(path):return json.loads(path.read_text())
torch.set_num_threads(4)
pairs=torch.load('/tmp/jzk_carry_video_pair.pt',weights_only=True)
dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True)
result=dict(video='city_walk / 701',frames=105,device='A100-SXM4-40GB GPU 7',
            warmup=1,repetitions=5,tiers={})
for name in ('base','intersection','step'):
    dest=Path('/tmp/jzk_carry_prepared_final_'+name)
    r=load(dest/'result.json')
    z=torch.load(dest/'latents.pt',weights_only=True)
    reference=pairs[1] if name=='base' else torch.load(Path('/tmp/jzk_cuda_samecard_'+name)/'latents.pt',weights_only=True)
    assert torch.equal(z.view(torch.uint8),reference.view(torch.uint8))
    result['tiers'][name]=dict(median_s=r['median_seconds'],seconds=r['seconds'],reference_bitwise=True,
        dense_nmse=float((z.float()-dense.float()).square().sum()/dense.float().square().sum()),
        cuda_calls=r['runtime_stats']['packed_attention_calls'],packed_commits=r['runtime_stats']['direct_layout_commits'])
for key in ('release_pairs','release_multi','installed_pairs','combined_pairs'):
    rows=load(root/key/'base.json')
    diff=[1000*(r['old_s']-r['new_s']) for r in rows]
    result[key]=dict(n=len(rows),faster=sum(v>0 for v in diff),median_gain_ms=median(diff),
                     gain_range_ms=[min(diff),max(diff)],control_bitwise=all(r['bitwise'] for r in rows))
trace=load(root/'combined_timeline.json')
totals={pre:sum(v for r in trace for n,v in r['by_name_ms'].items() if n.startswith(pre)) for pre in ('exact(','prepare(','retry')}
result['device_ms']=totals
elapsed=sum(totals.values())
result['attention_path_ms']=elapsed
model=load(root.parent/'fixed15_baseline_v1/budget.json')
tf=ref=0;warp_count=864*4
for r in model['rows']:
    tiles=(r['pages']+15)//16-1
    exact_mma=r['count']*192
    comp_mma=tiles*102
    work=(exact_mma+comp_mma)*warp_count*4096*r['calls']/1e12
    tf+=work
    ref+=r['reference_ms']*work/r['fused_tflop']
tensor_count=[r for r in csv.reader((root/'combined_ncu.csv').open()) if r and r[-3:-2]==['smsp__inst_executed_pipe_tensor.sum']]
assert int(tensor_count[0][-1])==(64*192+26*102+32)*warp_count
result['work']=dict(current_qk_av_tflop=tf,matched_dense_ms=ref,remaining_ms=elapsed-ref,
                    main_tflops=1000*tf/totals['exact('],path_tflops=1000*tf/elapsed,
                    extra_tf32_projection_tflop=32*warp_count*2048*812/1e12,p432_tensor_instructions=int(tensor_count[0][-1]))
(root/'release.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
