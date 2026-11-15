"""Compare only identical query subsets already covered by oracle results."""
import json,csv,shutil
from pathlib import Path
import numpy as np
from ..e11_oracle.solver import objective
ROOT=Path('/tmp/e11_small_tiles');OTHER=Path('/tmp/e11_small_tiles_groups');ARCH=Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_small_tiles')
def main():
 rows=[]
 for layer in (1,15,29):
  old=dict(np.load(f'/tmp/e11_results/contributions/horse_field_s101_c5_l{layer}_s0.npz'));r=json.loads(Path(f'/tmp/e11_results/pilot/horse_field_s101_c5_l{layer}_s0/result.json').read_text());oldmask=np.zeros(410,bool);oldmask[r['mask']]=True
  folders=[('G-KVR',ROOT/f'l{layer}/q{s}_k{s}',s) for s in (16,4)]
  folders.extend((p.parent.parent.name,p,int(p.name.split('_')[0][1:])) for p in OTHER.glob(f'G-*/l{layer}/q*'))
  for group,folder,size in folders:
   sums=np.zeros(6);tiles=[];statuses={};new_masks=0
   for path in sorted(folder.glob('t*/result.json')):
    ti=int(path.parent.name[1:]);r=json.loads(path.read_text());data=dict(np.load(path.parent.with_suffix('.npz')));od=data['N'].sum(1)/data['Z'].sum(1)[:,None];energy=float(np.square(od).sum()+1e-12);sl=slice(ti*size,(ti+1)*size)
    old_error=objective({k:old[k][sl] for k in ('N','Z','Nh','Zh')},oldmask)
    sums+=energy*np.array([objective(data,data['mass']),objective(data,data['risk']),r['UB'],r['LB'],old_error,1.]);tiles.append(ti);statuses[r['status']]=statuses.get(r['status'],0)+1;new_masks+=not r['warm_start_only']
   if not tiles:continue
   rows.append(dict(group=group,layer=layer,Q=size,K_page=size,queries=len(tiles)*size,tiles=','.join(map(str,sorted(tiles))),mass_nmse=sums[0]/sums[5],risk_nmse=sums[1]/sums[5],new_opt_UB=sums[2]/sums[5],new_opt_LB=sums[3]/sums[5],old_Q64_incumbent_same_queries_nmse=sums[4]/sums[5],new_masks=new_masks,global_tol=statuses.get('GLOBAL_TOL',0),instances=len(tiles),statuses=json.dumps(statuses)))
 ARCH.mkdir(parents=True,exist_ok=True)
 if rows:
  with (ROOT/'partial_opt_comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'partial_opt_comparison.json').write_text(json.dumps(rows,indent=2))
 lines=['小粒度OPT部分结果：严格相同Query范围比较','', '每行只评价该组已经求解完的Query，mass/risk与旧Q64集合均在完全相同Query上重算。','old_Q64_incumbent为旧模型当前最好可行集合，未取得全局最优证书。','new_opt_UB是当前可行上界；GLOBAL_TOL才支持该局部目标容差内最优。','若new_masks=0且LB=0，仅保留warm-start上界，不表示OPT已验证最优。','不同queries/tiles的行不能直接当作相同样本均值横比。','', 'group | layer | Q/K | Queries | mass | risk | newOPT UB | newOPT LB | oldQ64 incumbent']
 for r in rows:lines.append(f"{r['group']} | {r['layer']} | {r['Q']}/{r['K_page']} | {r['queries']} | {r['mass_nmse']:.9g} | {r['risk_nmse']:.9g} | {r['new_opt_UB']:.9g} | {r['new_opt_LB']:.9g} | {r['old_Q64_incumbent_same_queries_nmse']:.9g}")
 (ROOT/'PARTIAL_OPT.zh.txt').write_text('\n'.join(lines)+'\n')
 for n in ('partial_opt_comparison.csv','partial_opt_comparison.json','PARTIAL_OPT.zh.txt'):
  if (ROOT/n).exists():shutil.copy2(ROOT/n,ARCH/n)
 print('partial comparison rows',len(rows))
if __name__=='__main__':main()
