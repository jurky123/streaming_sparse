"""CPU-only reproducible design checks and compiled-artifact evidence; no GPU work."""
import hashlib,json,math,shutil,subprocess
from pathlib import Path
ROOT=Path('artifacts/implementation_review_20261030')
def softmax(x):
    v=[math.exp(a-max(x)) for a in x];return [a/sum(v) for a in v]
def avg_attention(keys):
    probs=[softmax([q*k for k in keys]) for q in [-1.,1.]]
    return [sum(p[i] for p in probs)/2 for i in range(len(keys))]
def approx(keys,centered=False):
    center=sum(keys)/len(keys) if centered else 0.
    return softmax([.5*.1*(k-center)**2 for k in keys])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ROOT.mkdir(exist_ok=True)
    keys=[-2.,-.5,.1,1.];shift=[k+10 for k in keys]
    exact,translated=avg_attention(keys),avg_attention(shift)
    a,b=approx(keys),approx(shift);c,d=approx(keys,True),approx(shift,True)
    assert max(abs(x-y) for x,y in zip(exact,translated))<1e-14
    assert max(abs(x-y) for x,y in zip(c,d))<1e-14
    assert a.index(max(a))!=b.index(max(b))
    # Current mass=.5, value=0; pageA mass=.3,value=0; pageB mass=.2,value=10.
    target=2.;select_a=0.;select_b=2/.7
    example=dict(target=target,select_larger_mass_page_error=(select_a-target)**2,select_smaller_mass_page_error=(select_b-target)**2)
    assert example['select_smaller_mass_page_error']<example['select_larger_mass_page_error']
    raw_limit=30*27648*12*128*2*2
    pk=29*12*338*4*128*4
    counts=29*12*338*4*4;centers=29*12*128*4
    before=json.loads(Path('artifacts/rollout_acceptance_20261030/runtime_checks.json').read_text())['persistent_self_kv_and_aux_bytes']
    budget=dict(limit_bytes=raw_limit,existing_bytes=before,remaining_bytes=raw_limit-before,prototype_K_bytes=pk,
        page_local_logcount_bytes=counts,one_center_per_layer_head_bytes=centers,
        counts_plus_centers_fit_payload_budget=before+counts+centers<=raw_limit)
    matches=[]
    for meta in Path('/data/chenjiayu/.triton/cache').glob('*/_attn_fwd.json'):
        ir=meta.with_suffix('.ttir');ptx=meta.with_suffix('.ptx')
        if not ir.exists() or not ptx.exists():continue
        text=ir.read_text();m=json.loads(meta.read_text())
        if not all(t in text for t in ['external_baselines/LightForcing/wan/modules/kernel.py','dense<26240>','dense<4608>','%c144_i32']):continue
        if m['num_warps']!=4 or m['num_stages']!=3:continue
        assembly=ptx.read_text();binary=meta.with_suffix('.cubin')
        resources=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(binary)],text=True)
        target=ROOT/'compiled'/meta.parent.name;target.mkdir(parents=True,exist_ok=True)
        for source in [meta,ir,ptx]:shutil.copy2(source,target/source.name)
        (target/'resource_usage.txt').write_text(resources)
        matches.append(dict(path=str(meta.parent),metadata=m,ptx_sha256=sha(ptx),cubin_sha256=sha(binary),resource_usage=resources,
            has_wgmma='wgmma.mma_async' in assembly,has_tma='cp.async.bulk.tensor' in assembly,
            note='Matched source path and steady shape constants; compiler evidence, not hardware-counter measurement.'))
    result=dict(common_key_translation=dict(keys=keys,shift=10,alpha=.1,queries=[-1,1],exact=exact,exact_shifted=translated,
        approximate=a,approximate_shifted=b,centered=c,centered_shifted=d,note='Synthetic mathematical counterexample; not a video improvement experiment.'),
        value_selection_counterexample=example,budget=budget,compiled=matches,
        counts=dict(sparse_calls=29*39*4,clean_commits=29*41,balanced_sort_calls=29*41*7,bmm_calls_in_balanced=29*41*7*9,
            note='Static call counts for fixed 41-chunk rollout; framework calls, not counted GPU kernels. Excludes feature projection.'),
        pair_merge_expected_history_tokens=128*169*(1-266*265/(338*337)),
        script_sha256=sha(Path(__file__)))
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='compiled'},indent=2))
if __name__=='__main__':main()
