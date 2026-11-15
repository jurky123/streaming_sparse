"""Adversarial cancellation: every first addition increases error, but budget must fill."""
import json,numpy as np
from pathlib import Path
from ..e11_compact.core import Problem
from .search import greedy
N=np.ones((1,5,1));Z=np.ones((1,5));raw_e=np.array([0.,1.,1.,-1.,-1.])[None,:,None];p=Problem(dict(N=N,Z=Z,Nh=N-raw_e,Zh=Z,own=np.array(0)),K=3,require_full=False);x,r=greedy(p)
assert x.sum()==3 and x[0] and r['negative_gain_steps']==1 and r['trace'][1]['all_additions_increase'] and r['final']==0
mask=np.zeros(5,bool);mask[0]=True
for t in r['trace'][1:]:
 scores=[]
 for page in np.where(~mask)[0]:
  trial=mask.copy();trial[page]=True;scores.append((p.value(trial),int(page)))
 winner=min(scores)[1];assert t['added']==winner;mask[winner]=True
out=dict(passed=True,scope='synthetic cancellation edge case, not a full-P result',trace=r['trace'],mask=np.where(x)[0].tolist());Path('/tmp/e11_set_search/greedy_edge_check.json').write_text(json.dumps(out,indent=2));print(out)
