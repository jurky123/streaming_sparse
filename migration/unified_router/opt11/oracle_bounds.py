"""Train-only, conditional one-query oracle diagnostic; not a global optimum."""
import argparse,json,statistics
from pathlib import Path
import torch
from ..common import config,atomic_json
from .data import feature_path
p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory);torch.set_num_threads(4);out={}
for variant in ['Compact','Aggressive']:
 rows=[]
 for layer in m['layers']:
  values=[]
  for row in m['rows']:
   if row['split']!='train':continue
   d=torch.load(feature_path(root,row,layer),map_location='cpu',weights_only=True,mmap=True);loss=d['labels'][variant];energy=float(d['energy']);values.append(dict(noop_nmse=float(loss[...,0].sum())/energy,conditional_oracle_nmse=float(loss.min(-1).values.sum())/energy))
  rows.append(dict(layer=layer,noop_nmse=statistics.mean(r['noop_nmse']for r in values),conditional_oracle_nmse=statistics.mean(r['conditional_oracle_nmse']for r in values)))
 out[variant]=dict(noop_mean=statistics.mean(r['noop_nmse']for r in rows),conditional_oracle_mean=statistics.mean(r['conditional_oracle_nmse']for r in rows),layers=rows)
atomic_json(root/'conditional_oracle_bounds.json',dict(scope='Train only; one target query changes within each original group32; peers fixed to initial pretrained masks; not a lower bound on a jointly changed full routing policy',variants=out));print(out)
