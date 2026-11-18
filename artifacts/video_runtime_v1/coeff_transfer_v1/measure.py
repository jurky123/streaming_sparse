"""Measure the four representative coefficients in the deployed fused attention kernel."""
import argparse
import importlib.util
from pathlib import Path
import re
from local_experiments.video_runtime_v1.page_prefetch import page_tail

folder=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('warp_measure',folder.parent/'warp_rows_v1/measure.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def transform(text,mode):
    if mode=='control':
        return text
    if mode.startswith(('paired','aligned')):
        rows,cols=(2,16) if mode.startswith('paired') else (8,4)
        for name,size,threads,warps in (
                ('blocked6','1, 2, 4',f'{rows}, {cols}, 1','4, 1, 1'),
                ('blocked7','1, 8',f'{rows}, {cols}','4, 1')):
            text,n=re.subn(rf'(#{name} = .*?sizePerThread = )\[[^\]]+\], threadsPerWarp = \[[^\]]+\], warpsPerCTA = \[[^\]]+\]',
                           rf'\g<1>[{size}], threadsPerWarp = [{threads}], warpsPerCTA = [{warps}]',text)
            assert n==1
    if mode in ('transpose','unswizzled'):
        layout='#shared5' if mode=='transpose' else '#shared2'
        old='!tt.memdesc<64x16xbf16, #shared3,'
        assert text.count(old)==16
        text=text.replace(old,old.replace('#shared3',layout))
    if mode.endswith('direct'):
        lines=text.splitlines()
        buffers={}
        loads=0
        for i,line in enumerate(lines):
            if 'triton_gpu.local_alloc' in line and '!tt.memdesc<64x16xbf16, #shared3,' in line:
                buf,source,ty=re.search(r'(%\w+) = triton_gpu.local_alloc (%\w+) : \((.*?)\) ->',line).groups()
                buffers[buf]=(source,ty)
                lines[i]=''
            if 'triton_gpu.local_load' in line and '!tt.memdesc<64x16xbf16, #shared3,' in line:
                buf=re.search(r'local_load (%\w+)',line)[1]
                source,ty=buffers[buf]
                lines[i]=line.split('triton_gpu.local_load')[0]+f'triton_gpu.convert_layout {source} : {ty} -> '+line.split(' -> ')[1]
                loads+=1
        assert loads==len(buffers)==8
        text='\n'.join(lines)+'\n'
    return text


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('pages',type=int,nargs='?',default=432)
    args=parser.parse_args()
    bench.folder=folder
    bench._tail=page_tail
    bench.transform=transform
    bench.measure(args.mode,args.pages)
