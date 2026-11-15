"""Paired full-query online comparison plus common-anchor oracle diagnostics."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from ..e11_oracle.solver import objective
ROOT=Path('/tmp/e11_small_tiles_groups');ARCH=Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_small_tiles');BASE=Path('/tmp/e11_small_tiles')
def read_cache(folder,size):
 cache=folder/'online_cache.json'
 if cache.exists():return json.loads(cache.read_text())
 rows=[]
 for ti in range(64//size):
  data=dict(np.load(folder/f't{ti}.npz'));od=data['N'].sum(1)/data['Z'].sum(1)[:,None]
  rows.append(dict(energy=float(np.square(od).sum()+1e-12),mass=objective(data,data['mass']),risk=objective(data,data['risk'])))
 cache.write_text(json.dumps(rows,indent=2));return rows

def main():
 rows=[];certs=[];anchors=[];old={}
 for layer in (1,15,29):
  path=Path('/tmp/e11_results/contributions')/f'horse_field_s101_c5_l{layer}_s0.npz';data=dict(np.load(path));folder=Path('/tmp/e11_results/pilot')/path.stem;r=json.loads((folder/'result.json').read_text());mask=np.zeros(data['Z'].shape[1],bool);mask[r['mask']]=True
  for size in (64,16,4):old[layer,size]=objective({n:data[n][:size] for n in ('N','Z','Nh','Zh')},mask)
 for folder in sorted(ROOT.glob('G-*/l*/q*')):
  if not (folder/'metadata.json').exists():continue
  meta=json.loads((folder/'metadata.json').read_text());group=meta['group'];layer=meta['layer'];size=meta['Q'];cache=read_cache(folder,size)
  meta['summary_note']='Frozen pooling formula; four complete historical chunks regrouped, oldest50 parent pages unchanged, current KV unchanged; each parent page then split to the requested smaller size.'
  tmp=folder/'metadata.report.tmp';tmp.write_text(json.dumps(meta,indent=2));tmp.replace(folder/'metadata.json')
  energy=sum(x['energy'] for x in cache)
  for method in ('mass','risk'):rows.append(dict(group=group,layer=layer,Q=size,K_page=size,method=method,nmse=sum(x[method]*x['energy'] for x in cache)/energy,queries=64,complete=True,global_tol=None))
  sse=opt_energy=0.;count=proof=0;anchor=dict(group=group,layer=layer,Q=size,K_page=size,query_range=f'[0,{size}) of original64',mass=cache[0]['mass'],risk=cache[0]['risk'],old_Q64_opt_incumbent_on_same_queries=old[layer,size],opt=None,LB=None,UB=None,status='NOT_COMPLETED')
  for ti,x in enumerate(cache):
   path=folder/f't{ti}/result.json'
   if not path.exists():continue
   r=json.loads(path.read_text());sse+=r['UB']*x['energy'];opt_energy+=x['energy'];count+=size;proof+=r['status']=='GLOBAL_TOL';certs.append(dict(group=group,layer=layer,Q=size,tile=ti,**{n:r.get(n) for n in ('status','LB','UB','absolute_gap','relative_gap','runtime','cpu_seconds','warm_start_only')}))
   if not (folder/f't{ti}/fp32_check.json').exists():
    from .verify import check
    check(folder,ti,dict(np.load(folder/f't{ti}.npz')),r)
   if ti==0:anchor.update(opt=r['UB'],LB=r['LB'],UB=r['UB'],status=r['status'])
  rows.append(dict(group=group,layer=layer,Q=size,K_page=size,method='opt',nmse=sse/opt_energy if opt_energy else None,queries=count,complete=count==64,global_tol=proof));anchors.append(anchor)
 if (BASE/'comparison.csv').exists():
  for r in csv.DictReader((BASE/'comparison.csv').open()):rows.append(dict(group='G-KVR',layer=int(r['layer']),Q=int(r['Q']),K_page=int(r['K_page']),method=r['method'],nmse=float(r['nmse']) if r['nmse'] else None,queries=int(r['queries']),complete=r['complete']=='True',global_tol=r['global_tol']))
 for layer in (1,15,29):
  for size in (64,16,4):
   folder=BASE/f'l{layer}'/f'q{size}_k{size}'
   if size==64:
    p=Path('/tmp/e11_results/pilot')/f'horse_field_s101_c5_l{layer}_s0';v=json.loads((p/'L1.json').read_text());r=json.loads((p/'strict_certificate.json').read_text());mass=v['O-MASS']['nmse'];risk=v['O-RISK']['nmse']
   else:
    x=read_cache(folder,size)[0];mass=x['mass'];risk=x['risk'];p=folder/'t0/result.json';r=json.loads(p.read_text()) if p.exists() else {}
   anchors.append(dict(group='G-KVR',layer=layer,Q=size,K_page=size,query_range=f'[0,{size}) of original64',mass=mass,risk=risk,old_Q64_opt_incumbent_on_same_queries=old[layer,size],opt=r.get('UB'),LB=r.get('LB'),UB=r.get('UB'),status=r.get('status','NOT_COMPLETED')))
 for name,data in [('group_comparison.csv',rows),('group_certificates.csv',certs),('matched_anchor_comparison.csv',anchors)]:
  if data:
   with (ROOT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 progress=dict(prepared_other_group_layer_granularities=len(list(ROOT.glob('G-*/l*/q*/metadata.json'))),expected_other_group_layer_granularities=54,other_group_completed_opt=len(certs),other_group_expected_anchor_opt=54,groups=['G-ORDER','G-K','G-K-joint','G-VR','G-KVR','G-K3-V2','G-K3-V2-Kinner'])
 (ROOT/'group_progress.json').write_text(json.dumps(progress,indent=2));ARCH.mkdir(exist_ok=True,parents=True)
 for path in ROOT.iterdir():
  if path.is_file() and path.suffix in ('.csv','.json','.txt','.py'):shutil.copy2(path,ARCH/path.name)
 print(progress,flush=True)
if __name__=='__main__':main()
