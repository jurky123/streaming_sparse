import subprocess
import sys
from pathlib import Path
out=Path(__file__).resolve().parent
workers=[]
def ready(handle):
    for line in handle.stdout:
        if line.strip()=='READY':
            return
    raise RuntimeError(f'Worker exited with {handle.wait()}')
try:
    for method in ('before-paired','fused-paired'):
        handle=subprocess.Popen([sys.executable,'-u',str(out/'quota_video.py'),method],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        workers.append(handle)
        ready(handle)
    for i in range(16):
        for j in ((0,1) if (i//4+i%4)%2==0 else (1,0)):
            h=workers[j]
            h.stdin.write('\n')
            h.stdin.flush()
            if i<15:
                ready(h)
            elif h.wait()!=0:
                raise RuntimeError('Worker failed')
finally:
    for h in workers:
        if h.poll() is None:
            h.terminate()
