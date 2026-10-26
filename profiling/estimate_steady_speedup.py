"""Amdahl estimate from measured steady chunks; no router implementation is benchmarked."""
import argparse,csv,json,statistics
from pathlib import Path
from collections import defaultdict

def main():
 p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
 s=json.loads((root/'summary.json').read_text());m=s['metadata'];first=m['max_history_tokens']//m['query_tokens']+2
 chunks=list(csv.DictReader((root/'chunk_summary.csv').open()));steady=[x for x in chunks if int(x['chunk'])>=first]
 n=len(steady);baseline=statistics.mean(float(x['baseline_wall_ms_median']) for x in steady)
 vae=statistics.mean(float(x['vae_stream_decode_ms']) for x in steady)
 parts=list(csv.DictReader((root/'exclusive_kernels.csv').open()))
 def total(scope,stage,core_only=False):
  return sum(float(x['gpu_kernel_ms']) for x in parts if int(x['chunk'])>=first and (scope is None or int(x['layer']) in scope) and (stage is None or x['stage']==stage) and (x['part']=='self.sdpa_fused' if core_only else x['part'].startswith('self.')))/n
 selected={4,14,24};q=m['query_tokens'];h=m['max_history_tokens']
 baselines={'steady_chunk_generation_ms':baseline,'steady_vae_ms':vae,'selected3_denoise_self_module_ms':total(selected,'denoise'),'selected3_denoise_self_core_ms':total(selected,'denoise',True),'all30_denoise_self_module_ms':total(None,'denoise'),'all30_denoise_self_core_ms':total(None,'denoise',True),'all30_self_module_denoise_plus_context_ms':total(None,None)}
 rows=[]
 for interpretation in ['history_keep','history_delete']:
  for rate in [.1,.2]:
   keep=rate if interpretation=='history_keep' else 1-rate
   ratio=(q+keep*h)/(q+h)
   for scenario,scope in [('current3layers',selected),('all30layers_hypothetical',None)]:
    core=total(scope,'denoise',True);module=total(scope,'denoise');saved=core*(1-ratio)
    all_denoise=total(None,'denoise');all_five=total(None,None)
    rows.append(dict(interpretation=interpretation,rate=rate,history_keep=keep,scenario=scenario,remaining_core_work_fraction=ratio,eligible_core_speedup=1/ratio,eligible_full_self_attention_module_speedup=module/(module-saved),all30_denoise_self_attention_speedup=all_denoise/(all_denoise-saved),all30_self_attention_including_dense_context_speedup=all_five/(all_five-saved),steady_chunk_generation_speedup=baseline/(baseline-saved),steady_chunk_generation_ms=baseline-saved,steady_chunk_generation_plus_vae_speedup=(baseline+vae)/(baseline-saved+vae),steady_chunk_generation_plus_vae_ms=baseline-saved+vae,ideal_saving_ms_per_chunk=saved,additional_total_overhead_break_even_ms_per_chunk=saved))
 report={'definition':'3 historical chunks plus1 full current chunk; only denoising self-SDPA core is pruned. Router/sketch/gather/new sparse kernel inefficiency cost assumed zero; projected kernel time linear in visible KV count. QKV, RoPE, KV maintenance, O projection, cross-attention, FFN, dense clean-cache write and VAE are unchanged.','steady_chunks':[first,len(chunks)],'baselines_ms':baselines,'formula':{'core_work_fraction':'r=(Q+k*H)/(Q+H)=0.25+0.75*k','full_self_module':'S=M/(M-C*(1-r))','chunk':'S=T/(T-C*(1-r))','scope':'C includes exactly the selected layers and four denoising forwards; excludes the fifth dense context forward.'},'rows':rows,'measurement_note':'T is the mean of per-chunk median clean wall times for chunks5..27; M and C are exclusive CUDA kernel times from a separate profile with <1% overhead. Estimates inherit this cross-pass approximation.'}
 (root/'steady_speedup_estimates.json').write_text(json.dumps(report,indent=2)+'\n')
 with (root/'steady_speedup_estimates.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
