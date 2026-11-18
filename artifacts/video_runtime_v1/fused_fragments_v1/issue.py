"""Change independent accumulator issue order while preserving each scalar's reduction order."""
import argparse
import hashlib
import subprocess
import tempfile
import json
from collections import defaultdict
import re
from measure import bench, folder, page_tail
from triton.compiler.compiler import make_backend


def reorder(text, mode):
    lines=text.splitlines()
    split=next(i for i,l in enumerate(lines) if 'cp.async.wait_group 0x1' in l)
    hi,lo=[max(i for i,l in enumerate(lines[:split]) if f'gap_exact_round_kernel.py:153:{col}' in l) for col in (24,46)]
    end=next(i for i in range(lo+1,split) if '.loc' in lines[i])
    body=lines[hi:end]
    groups=defaultdict(list)
    prefix=[]
    for line in body:
        op=line.strip()
        if op.startswith('mma.sync'):
            dest=re.search(r'\{ (%f\d+)',line)[1]
            groups[dest].append(line)
        elif op.startswith(('mov.f32','ldmatrix.')):
            prefix.append(line)
        elif op and not op.startswith(('.loc','//')):
            raise ValueError(line)
    keys=list(groups)
    assert len(keys)==16 and all(len(v)==8 for v in groups.values())
    if mode.startswith('parallel_'):
        # Compute low AV into independent FP32 outputs, then add high and low.
        # Keep the four K tiles in order inside both terms.
        prefix.insert(0,'    .reg .f32 %fraglow<64>;')
        ops=[]
        width=int(mode.split('_')[-1])
        for j in range(0,16,width):
            batch=keys[j:j+width]
            for i,key in enumerate(batch):
                for n in range(4):
                    ops.append(f'    mov.f32 %fraglow{i*4+n}, 0f00000000;')
            for k in range(4):
                for i,key in enumerate(batch):
                    low=groups[key][k+4]
                    regs=re.search(r'\{ ([^}]+?) \}',low)[1].split(', ')
                    low=re.sub(r'%f\d+',lambda m:'%fraglow'+str(i*4+regs.index(m[0])),low)
                    ops.extend((groups[key][k],low))
            for i,key in enumerate(batch):
                regs=re.search(r'\{ ([^}]+?) \}',groups[key][0])[1].split(', ')
                ops.extend(f'    add.rn.f32 {r}, {r}, %fraglow{i*4+n};' for n,r in enumerate(regs))
        return '\n'.join(lines[:hi]+prefix+ops+lines[end:])+'\n'
    width=int(mode.split('_')[-1])
    if mode.startswith('column'):
        keys=[keys[r*4+c] for c in range(4) for r in range(4)]
    ops=[]
    for j in range(0,16,width):
        ops.extend(groups[key][k] for k in range(8) for key in keys[j:j+width])
    return '\n'.join(lines[:hi]+prefix+ops+lines[end:])+'\n'


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode')
    args=ap.parse_args()
    compile_original=bench.triton.compile
    def compile_variant(src,options):
        kernel=compile_original(src,options=options)
        backend=make_backend(kernel.metadata.target)
        ptx=reorder(kernel.asm['ptx'],args.mode)
        original=kernel.kernel
        assert kernel.module is None
        kernel.kernel=backend.make_cubin(ptx,kernel.metadata._asdict(),backend.parse_options(options),80)
        def text_digest(binary):
            with tempfile.NamedTemporaryFile() as f:
                f.write(binary)
                f.flush()
                sections=subprocess.check_output(['readelf','-SW',f.name],text=True)
            offset,size=re.search(r'\.text\._tail\s+PROGBITS\s+\w+\s+(\w+)\s+(\w+)',sections).groups()
            start=int(offset,16)
            return hashlib.sha256(binary[start:start+int(size,16)]).hexdigest()
        hashes=[text_digest(binary) for binary in (original,kernel.kernel)]
        (folder/(args.mode+'_binary.json')).write_text(json.dumps(hashes)+'\n')
        return kernel
    bench._tail=page_tail
    bench.folder=folder
    bench.transform=lambda text,mode:text
    bench.triton.compile=compile_variant
    bench.measure(args.mode,432)
