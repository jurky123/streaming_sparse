"""Check machine instruction encodings for both ends of the packed-summary contract."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root=Path(__file__).parent
source=Path('local_experiments/video_runtime_v1/csrc/fixed15_attention.cu')
key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
paths={'candidate':Path('/tmp/jzk_tailmax_direct_mass.so'),
       'release':Path('/tmp/streaming_sparse-0/fixed15_attention')/key/'kernel.so'}
result={}
for version,file in paths.items():
    disasm=subprocess.check_output(['cuobjdump','-sass',str(file)],text=True)
    result[version]={}
    for fn in ('exact','prepare'):
        body=disasm.split('Function : _Z'+str(len(fn))+fn,1)[1].split('Function :',1)[0]
        words=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body)
        assert len(words)>100 and len(words)%2==0
        result[version][fn]=dict(instructions=len(words)//2,sha256=hashlib.sha256(''.join(words).encode()).hexdigest())
assert result['release']==result['candidate']
(root/'machine_code.json').write_text(json.dumps(result,indent=2))
