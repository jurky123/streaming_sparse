"""Immediate online baselines and incremental oracle results on identical 64 queries."""
import json,csv,math,shutil
from pathlib import Path
import numpy as np
from ..e11_oracle.solver import objective
ROOT=Path('/tmp/e11_small_tiles');ARCH=Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_small_tiles')
def main():
 rows=[];cert=[]
 for layer in (1,15,29):
  old=json.loads((Path('/tmp/e11_results/pilot')/f'horse_field_s101_c5_l{layer}_s0/L1.json').read_text())
  for method in ('mass','risk','opt'):
   rows.append(dict(layer=layer,Q=64,K_page=64,method=method,nmse=old['O-'+method.upper()]['nmse'],queries=64,complete=True,global_tol=None,instances=1))
  for size in (16,4):
   folder=ROOT/f'l{layer}'/f'q{size}_k{size}';metafile=folder/'metadata.json'
   if not metafile.exists():continue
   values={n:[0.,0.,0,0] for n in ('mass','risk','opt')}
   for ti in range(64//size):
    file=folder/f't{ti}.npz';data=dict(np.load(file));od=data['N'].sum(1)/data['Z'].sum(1)[:,None];energy=float(np.square(od).sum()+1e-12)
    for method in ('mass','risk'):
     v=values[method];v[0]+=objective(data,data[method])*energy;v[1]+=energy;v[2]+=size
    result=folder/f't{ti}/result.json'
    if result.exists():
     r=json.loads(result.read_text())
     if not (folder/f't{ti}'/'fp32_check.json').exists():
      from .verify import check
      check(folder,ti,data,r)
     v=values['opt'];v[0]+=r['UB']*energy;v[1]+=energy;v[2]+=size;v[3]+=int(r['status']=='GLOBAL_TOL')
     cert.append(dict(layer=layer,Q=size,K_page=size,tile=ti,**{n:r.get(n) for n in ('P','K','LB','UB','absolute_gap','relative_gap','status','runtime','build_seconds','cpu_seconds','warm_start_only','feasibility_residual')}))
   for method,(sse,energy,count,global_tol) in values.items():
    rows.append(dict(layer=layer,Q=size,K_page=size,method=method,nmse=sse/energy if energy else None,queries=count,complete=count==64,global_tol=global_tol if method=='opt' else None,instances=count//size))
 ROOT.mkdir(exist_ok=True,parents=True);ARCH.mkdir(exist_ok=True,parents=True)
 for name,data in [('comparison.csv',rows),('certificates.csv',cert)]:
  if data:
   with (ROOT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 progress=dict(expected_solves=60,completed_solves=len(cert),global_tol=sum(x['status']=='GLOBAL_TOL' for x in cert),prepared_groups=len(list(ROOT.glob('l*/q*/metadata.json'))),status='complete' if len(cert)==60 else 'running',scope='horse_field seed101 chunk5 step0; one registered head at layers1/15/29; all 64 original queries')
 (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
 lines=['E11 小Q/K tile：实测进度与结果','',json.dumps(progress,ensure_ascii=False),'','layer | Q/K | 方法 | NMSE | 已评价Query | 完整64Query']
 for r in rows:lines.append(f"{r['layer']} | {r['Q']}/{r['K_page']} | {r['method']} | {r['nmse']} | {r['queries']} | {r['complete']}")
 lines+=['','注意：opt未完成64Query时，不能与完整64Query均值直接比较。','所有粒度均使用同一26240个原始token，每Query精确2624个token。','R4/rank8冻结；小页重新计算BF16摘要，代表质量从16改为4/1。','Page4的Key代表数等于原始token数，不能将其精度收益全部归因于选页。','只有GLOBAL_TOL才能称该实例容差内全局最优；其余保留上下界。']
 (ROOT/'REPORT.zh.txt').write_text('\n'.join(lines)+'\n')
 for file in ROOT.iterdir():
  if file.is_file() and file.suffix in ('.json','.csv','.txt','.py'):shutil.copy2(file,ARCH/file.name)
 from .partial_comparison import main as partial_report
 partial_report()
 print(progress,flush=True)
if __name__=='__main__':main()
