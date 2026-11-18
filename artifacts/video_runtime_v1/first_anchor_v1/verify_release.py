"""Verify that normal deployment generated the same code as the measured prototype."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from local_experiments.video_runtime_v1.fixed15_attention import library

def instructions(path):
    sass=subprocess.check_output(['cuobjdump','-sass',path],text=True)
    pieces=re.split(r'Function : (\S+)',sass)
    counts={}
    for name,section in zip(pieces[1::2],pieces[2::2]):
        words=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',section)
        assert words and len(words)%2==0
        counts[name]=dict(instructions=len(words)//2,sha256=hashlib.sha256(''.join(words).encode()).hexdigest())
    assert len(counts)==6 and any('exact' in key for key in counts)
    return counts

prototype=instructions('/tmp/jzk_first_early.so')
assert instructions(library()._name)==prototype
Path(__file__).with_name('release_sass.json').write_text(json.dumps(prototype,indent=2))
