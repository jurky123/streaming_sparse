from pathlib import Path
import subprocess, sys, json, os
root=Path(__file__).resolve().parents[4]
out=Path(__file__).resolve().parent
commands=[
 [sys.executable,'-u','-m','streaming_sparse.v03.repair_20261016.experiment','--output',str(out),'--steps','1000','--device','cuda'],
 [sys.executable,'-u','-m','streaming_sparse.v03.repair_20261016.report','--round',str(out)],
]
for command in commands:
    print('COMMAND',json.dumps(command),flush=True)
    result=subprocess.run(command,cwd=root)
    if result.returncode:
        raise SystemExit(result.returncode)
print('REPAIR EXPERIMENT AND REPORT COMPLETE',flush=True)
