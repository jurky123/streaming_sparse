"""Separate process: SCIP may hold Python GIL during a long LP solve."""
import json,time,sys,os
from pathlib import Path
root=Path(sys.argv[1]);start=float(sys.argv[2]);deadlines=json.loads(sys.argv[3]);records=[]
for deadline in deadlines:
 while time.monotonic()-start<deadline:
  if (root/'search_finished').exists():sys.exit(0)
  time.sleep(min(.05,max(.001,deadline-(time.monotonic()-start))))
 state=json.loads((root/'latest_bound.json').read_text());records.append(dict(requested_seconds=deadline,wall_seconds=time.monotonic()-start,**state))
 (root/'checkpoints.json').write_text(json.dumps(records,indent=2))
