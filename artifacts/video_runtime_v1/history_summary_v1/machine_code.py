"""Check instruction encodings after integrating the two prepare kernels."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
root=Path(__file__).resolve().parent
src=root.parents[2]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu'
target=Path('/tmp/streaming_sparse-0/fixed15_attention')/hashlib.sha256(src.read_bytes()).hexdigest()[:16]/'kernel.so'
def functions(path):
    text=subprocess.check_output(['cuobjdump','-sass',str(path)],text=True)
    result={}
    for code in text.split('Function : ')[1:]:
        name=code.splitlines()[0].strip()
        words=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',code)
        result[name]=dict(count=len(words)//2,sha=hashlib.sha256(''.join(words).encode()).hexdigest())
    return result
base=functions('/tmp/jzk_compact_half.so')
candidate=functions('/tmp/jzk_history_summary.so')
release=functions(target)
pairs={}
for name,ref in base.items():
    pairs[name]=dict(old=ref,release=release[name],identical=ref==release[name])
name=next(n for n in base if n.startswith('_Z7prepare'))
other=next(n for n in release if n.startswith('_Z15prepare_current'))
pairs[other]=dict(candidate=candidate[name],release=release[other],identical=candidate[name]==release[other])
(root/'machine_code.json').write_text(json.dumps(pairs,indent=2)+'\n')
print({k:v['identical'] for k,v in pairs.items()})
