"""Record exact-kernel code identity and resources separately from the other entry points."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent
source=root.parents[2]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu'
key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
binaries={'base':Path('/tmp/streaming_sparse-0/fixed15_attention')/key/'kernel.so'}
for name in ['rolled','auto216','auto432','const216','const288','const360','const432']:
    binaries[name]=Path(f'/tmp/jzk_route_{name}.so')
result={}
for name,path in binaries.items():
    sass=subprocess.check_output(['cuobjdump','-sass',str(path)],text=True)
    part=sass.split('Function : _Z5exact')[1].split('Function : ')[0]
    words=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',part)
    assert words and len(words)%2==0
    usage=subprocess.check_output(['cuobjdump','-res-usage',str(path)],text=True).split('Function _Z5exact')[1].splitlines()[1]
    result[name]=dict(instructions=len(words)//2,hash=hashlib.sha256(''.join(words).encode()).hexdigest(),resources=usage.strip())
(root/'codegen.json').write_text(json.dumps(result,indent=2))
for name,r in result.items():print(name,r['instructions'],r['hash'][:10],r['resources'][:26])
