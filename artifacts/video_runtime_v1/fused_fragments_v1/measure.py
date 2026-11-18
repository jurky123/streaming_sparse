"""Isolate changes to the steady-state anchored page on top of deployed prefetch."""
import argparse
import importlib.util
from pathlib import Path
import re

from local_experiments.video_runtime_v1.page_prefetch import page_tail

folder = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("warp_measure", folder.parent / "warp_rows_v1/measure.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def schedule(text, mode):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if '= scf.for' in line)
    end = next(i for i in range(start, len(lines)) if 'scf.yield' in lines[i])
    body = lines[start + 1:end]
    assert any('%waitv =' in line for line in body)
    prob = [i for i, line in enumerate(body) if '!tt.memdesc<64x64xbf16, #shared,' in line]
    assert len(prob) == 4
    if mode.startswith('av_'):
        layout = '#avmma'
        axes = '[4, 1]' if mode == 'av_row' else '[2, 2]'
        old = 'tensor<64x128xf32, #mma1>'
        new = old.replace('#mma1',layout)
        head = lines[start]
        initial = re.search(r'iter_args\(%arg\d+ = (%\w+)', head)[1]
        final = head.strip().split(':')[0]+'#0'
        zero = next(line.split(' = ')[0].strip() for line in lines[:start] if 'arith.constant dense<0.000000e+00>' in line and old in line)
        prefix = [f'    %avstart = triton_gpu.convert_layout {initial} : {old} -> {new}',
                  f'    %avzero = arith.constant dense<0.000000e+00> : {new}']
        for i,line in enumerate(body):
            if ('local_load' in line and ('64x64xbf16, #shared,' in line or ('mutable' in line and '64x128xbf16, #shared,' in line))
                or (old in line and ('tt.dot' in line or 'arith.addf' in line))):
                body[i]=line.replace('#mma1',layout).replace(', '+zero+',',', %avzero,')
        decl = next(line for line in lines if line.startswith('#mma1 =')).replace('#mma1',layout).replace('[1, 4]',axes)
        after=[line.replace(final+',','%avfinal,') for line in lines[end+2:]]
        return '\n'.join([decl]+lines[:start]+prefix+[head.replace('= '+initial+',','= %avstart,',1).replace(old,new)]+body+
                          [lines[end].replace(old,new),lines[end+1],f'    %avfinal = triton_gpu.convert_layout {final} : {new} -> {old}']+after)+'\n'
    if mode in ('phase2','phase4','vec4','transpose'):
        fields = {'phase2':(8,2,4,'1, 0'), 'phase4':(8,4,2,'1, 0'),
                  'vec4':(4,1,8,'1, 0'), 'transpose':(8,1,8,'0, 1')}
        vec,per,maxp,order=fields[mode]
        for i in prob:
            body[i] = body[i].replace('#shared,','#sharedprob,')
        layout = f'#sharedprob = #triton_gpu.shared<{{vec = {vec}, perPhase = {per}, maxPhase = {maxp}, order = [{order}], hasLeadingOffset = false}}>'
        lines.insert(0,layout)
        start += 1
        end += 1
    elif mode == 'serial':
        # Release the high shared fragment before publishing the low fragment.
        hi,lo,readhi,readlo=prob
        body = body[:lo] + body[readhi:readlo] + body[lo:readhi] + body[readlo:]
    elif mode == 'readboth':
        hi,lo,readhi,readlo=prob
        body.insert(readhi,body.pop(readlo))
    elif mode!='control':
        raise ValueError(mode)
    return '\n'.join(lines[:start+1]+body+lines[end:])+'\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('pages',type=int,nargs='?',default=432)
    args = ap.parse_args()
    bench.folder=folder
    bench._tail=page_tail
    bench.transform=schedule
    bench.measure(args.mode,args.pages)
