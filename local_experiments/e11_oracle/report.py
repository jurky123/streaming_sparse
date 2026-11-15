"""Conservative final audit: certificates, local diagnostics, rollout coverage."""
import json,csv,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/e11_results')
def dump_csv(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 certificates=[];local=[];d02=[];failures=[]
 for path in sorted((ROOT/'contributions').glob('*.npz')):
  name=path.stem;folder=ROOT/'pilot'/name;data=dict(np.load(path));result=folder/'result.json'
  if not result.exists():failures.append(dict(instance=name,scope='full_P',status='UNFINISHED',reason='no result.json'));continue
  r=json.loads(result.read_text());mask=np.zeros(data['Z'].shape[1],bool);mask[r['mask']]=True;warm=any(np.array_equal(mask,data[n]) for n in ('mass','risk'));status=r['status']
  if r.get('feasibility_residual') is not None and r['feasibility_residual']>1e-8:status='NUMERIC_UNRESOLVED'
  if r['LB']>r['UB']+max(1e-10,1e-6*abs(r['UB'])):status='NUMERIC_UNRESOLVED'
  if status=='BOUNDED_INCUMBENT' and warm and r['LB']<=0:status='NO_FEASIBLE_CERTIFICATE'
  (folder/'strict_certificate.json').write_text(json.dumps(dict(r,status=status,warm_start_only=warm),indent=2))
  fields=['P','K','Q','d','LB','UB','relative_gap','absolute_gap','solver_version','seed','runtime','build_seconds','wall_seconds','cpu_seconds','work','node_count','peak_memory_kib','feasibility_residual','integrality_residual','model_hash','denominator_condition_max']
  certificates.append(dict(instance=name,scope='full_P',status=status,warm_start_only=warm,**{n:r.get(n) for n in fields}))
  if status!='GLOBAL_TOL':failures.append(dict(instance=name,scope='full_P',status=status,solver_status=r['solver_status']))
  if not (folder/'L1.json').exists():failures.append(dict(instance=name,scope='L1',status='UNFINISHED'));continue
  vals=json.loads((folder/'L1.json').read_text());od=data['N'].sum(1)/data['Z'].sum(1)[:,None];cd=np.square(od).sum()+1e-12
  for method,met in vals.items():
   local.append(dict(instance=name,method=method,certificate=status,**{n:v for n,v in met.items() if n not in ('selected_pages','gap_to_optimum_interval')},gap_to_opt_lower=met['nmse']-r['UB'],gap_to_opt_upper=met['nmse']-r['LB']))
   a=dict(np.load(folder/(method+'_D02.npz')));selected=a['mask'];zm=np.where(selected[None],data['Z'],data['Zh']).sum(1)
   d02.append(dict(instance=name,method=method,A_mixed_normalized=float(np.sum(a['A']/zm**2)/cd),C_mixed_normalized=float(np.sum(a['C']/zm**2)/cd),actual_nmse=float(a['Jq'].sum()/cd),fixed_dense_denominator_nmse=float(a['fixed_dense_denominator_Jq'].sum()/cd),mean_kappa=float(a['kappa'].mean()),net_cancellation_query_fraction=float((a['kappa']<1).mean()),mean_denominator_ratio=float(a['denominator_ratio'].mean()),mean_abs_log_denominator_delta=float(np.abs(a['log_denominator_delta']).mean()),cosine_median=float(np.median(a['pair_cosines']))))
 dump_csv(ROOT/'full_P_certificates.csv',certificates);dump_csv(ROOT/'L1.csv',local);dump_csv(ROOT/'D02.csv',d02)
 small=[]
 for name in ('small_tests.json','real_small_tests.json'):
  if not (ROOT/name).exists():continue
  for r in json.loads((ROOT/name).read_text()):
   small.append(dict(scope=r['kind'],P=r['P'],K=r['K'],Q=r['Q'],d=r['d'],enumeration_status='EXHAUSTIVE_CHECKED',enumerated_sets=r['sets'],exhaustive_optimum=r['exhaustive_value'],solver_status=r['result']['status'],LB=r['result']['LB'],UB=r['result']['UB'],passed=r['passed']))
   if not r['passed']:failures.append(dict(instance=f"{r['kind']}_P{r['P']}_Q{r['Q']}_d{r['d']}",scope='reduced',status=r['result']['status']))
 dump_csv(ROOT/'reduced_certificates.csv',small)
 videos=[]
 for path in sorted((ROOT/'video').glob('*/*/*/result.json')):
  r=json.loads(path.read_text());videos.append(dict(case=r['case'],method=r['method'],level=r['level'],latent_nmse=r['latent_nmse'],lpips=r['quality']['lpips'],ssim=r['quality']['ssim'],psnr=r['quality']['psnr'],instances=r['instances'],fraction_of_full_sparse_headtiles=r['fraction_of_full_sparse_headtiles'],proof_fraction=r['proof_fraction'],run_label=r['run_label'],wall_seconds=r['wall_seconds']))
 dump_csv(ROOT/'L2.csv',[r for r in videos if r['level'].startswith('L2')]);dump_csv(ROOT/'L3.csv',[r for r in videos if r['level'].startswith('L3')]);(ROOT/'failures.json').write_text(json.dumps(failures,indent=2))
 summary=dict(pilot_results=len(certificates),pilot_expected=12,global_tol=sum(r['status']=='GLOBAL_TOL' for r in certificates),local_rows=len(local),video_results=len(videos),video_expected=36,small_checks=len(small),small_passed=sum(r['passed'] for r in small),failures=len(failures),full_coverage_not_run=True)
 (ROOT/'completion.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
