"""Compare the disassembled SM80 main kernel, including instruction encodings."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

root=Path(__file__).parent
source=Path('local_experiments/video_runtime_v1/csrc/fixed15_attention.cu')
key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
paths={'candidate':Path('/tmp/jzk_tailmask_lane_fma.so'),
       'deployed':Path('/tmp/streaming_sparse-0/fixed15_attention')/key/'kernel.so'}
result={}
for name,path in paths.items():
    text=subprocess.check_output(['cuobjdump','-sass',str(path)],text=True)
    body=text.split('Function : _Z5exact',1)[1].split('Function :',1)[0]
    encoded=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body)
    assert len(encoded)>1000 and len(encoded)%2==0
    result[name]=dict(instructions=len(encoded)//2,sha256=hashlib.sha256(''.join(encoded).encode()).hexdigest())
assert result['candidate']==result['deployed']
(root/'machine_code.json').write_text(json.dumps(result,indent=2)+'\n')
