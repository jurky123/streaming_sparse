"""Verify every saved decision against its original legal Medium panel."""
import json
from pathlib import Path
import numpy as np
from .run import DEST,E25

def main():
 audit=[]
 for seed in [17,29]:
  panels={}
  for path in (E25/f'C_s{seed}').glob('panel_*.npz'):
   d=np.load(path);panels[str(d['event'])]=dict(mask=d['states'][0],r=d['initial_remove'],a=d['initial_add'])
  for suffix in ['', '_uniform']:
   rows=json.loads((DEST/f's{seed}{suffix}'/'decisions.json').read_text())
   for row in rows:
    p=panels[row['event']];mask=p['mask'];own=len(mask)-72+row['tile'];r,a=row['remove'],row['add'];assert mask[own] and mask[r] and not mask[a] and r!=own;assert np.any((p['r']==r)&(p['a']==a));new=mask.copy()
    if row['execute']:new[r]=False;new[a]=True
    assert new.sum()==mask.sum()==max(1,int(len(mask)*.1)) and new[own]
   audit.append(dict(seed=seed,variant=suffix or 'primary',decisions=len(rows),count_own_panel_valid=True))
 (DEST/'all_action_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
