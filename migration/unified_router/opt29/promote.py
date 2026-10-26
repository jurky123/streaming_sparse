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
  nmse=statistics.mean(d['best_validation']['nmse']for d in ds);ms=statistics.mean(d['ms']for d in ds);base=statistics.mean(d['baseline_ms']for d in ds)
  rows.append(dict(variant=name,nmse=nmse,ms=ms,speedup=base/ms,options=o))
 baselines=[json.loads(p.read_text())for p in(ROOT/'baseline').glob('layer*.json')];frozen=statistics.mean(x['q32_nmse']for x in baselines)if len(baselines)==4 else None
 selected=[];reasons={}
 def add(row,reason):
  if row and row['variant']not in selected:selected.append(row['variant']);reasons[row['variant']]=reason
 control=next((r for r in rows if r['variant']=='q32_full'),None);add(control,'mandatory continued-training Q32K64 control')
 fresh=[r for r in rows if r['options']['refresh']=='fresh'and r['options']['mode']in ['learned','mean','sample']]
 fast=[r for r in fresh if r['speedup']>=1.0 and(frozen is None or r['nmse']<=frozen*1.35)]
 add(min(fast or fresh,key=lambda r:r['nmse'])if fresh else None,'best fresh aggregation route (learned/mean/sample); if no speed/quality-qualified route exists, exploratory quality-first fallback is labelled')
 main=[r for r in fresh if r['options']['g']==64 and r['options']['k']==128];eligible=[r for r in main if r['speedup']>=1.0 and(frozen is None or r['nmse']<=frozen*1.35)]
 add(min(eligible or main,key=lambda r:r['nmse'])if main else None,'best Q64K128 hardware-aligned aggregation route retained alongside Q32 comparison')
 reused=[r for r in rows if r['options']['refresh']!='fresh'and r['speedup']>=1.0 and(frozen is None or r['nmse']<=frozen*1.5)]
 add(min(reused,key=lambda r:r['nmse'])if reused else None,'best reused-assignment route, screened for finite native error, >=1x module speed and <=1.5x frozen Q32 NMSE')
 atomic_json(ROOT/'promotion.json',dict(selected=selected,reasons=reasons,frozen_q32_nmse=frozen,pilots=rows,rule='Predeclared pilot selection; Val used for development, no sealed Test use; finalists up to6000updates/all29layers; unqualified quality fallback must not be called accepted deployment'))
 return selected
