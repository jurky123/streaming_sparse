import json,statistics
from .setup import ROOT,PILOTS,VARIANTS
from .io import atomic_json
def choose():
 rows=[]
 for name,o in VARIANTS.items():
  paths=[ROOT/f'bench/pilot/layer{l:02d}/{name}.json'for l in PILOTS]
  if not all(p.exists()for p in paths):continue
  ds=[json.loads(p.read_text())for p in paths]
  if not all(d['completed']for d in ds):continue
  rows.append(dict(variant=name,nmse=statistics.mean(d['best_validation']['nmse']for d in ds),ms=statistics.mean(d['ms']for d in ds),speedup=statistics.mean(d['baseline_ms']for d in ds)/statistics.mean(d['ms']for d in ds),options=o))
 selected=[];reasons={}
 def add(row,reason):
  if row and row['variant']not in selected:selected.append(row['variant']);reasons[row['variant']]=reason
 for name in ['q32_full','q64_mean8']:
  add(next((r for r in rows if r['variant']==name),None),'mandatory matched continued-training control')
 reorder=[r for r in rows if r['options']['grouping']!='none'and r['options']['mode']=='mean']
 add(min(reorder,key=lambda r:r['nmse'])if reorder else None,'best behavior regrouping with fixed mean pooling; exploratory until full-video PSNR and LPIPS acceptance')
 local=[r for r in rows if r['options']['mode']=='local'and r['options']['g']==64]
 add(min(local,key=lambda r:r['nmse'])if local else None,'best bounded local pooling Q64 candidate; no claim of accepted quality')
 add(next((r for r in rows if r['variant']=='q32_qk_local16'),None),'Q32 hardware comparison for learned grouping and local pooling')
 atomic_json(ROOT/'promotion.json',dict(selected=selected,reasons=reasons,pilots=rows,rule='Up to five routes: Q32/full and Q64/mean controls, best regrouped mean, best Q64 local pooling, Q32 regrouped local16. Rank native96-Val NMSE; report measured runtime, no speed/quality acceptance until video PSNR+LPIPS.'))
 return selected
