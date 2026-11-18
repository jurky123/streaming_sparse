"""Ablations of the BF16 high/low AV accumulation in the deployed anchored loop."""
import importlib
from pathlib import Path
import sys
import tempfile
from local_experiments.video_runtime_v1.weights import ROOT

folder=tempfile.TemporaryDirectory()
sys.path.insert(0,folder.name)


def variant_for(src, mode):
    code=Path(__file__).with_name('anchor_reference.py').read_text()
    src=src.replace('h,H,D,EXACT_COUNT,QK_SCALE,COLUMN_MASS)','h,H,D,EXACT_COUNT,QK_SCALE)')
    accumulate_mass=mode.endswith('_mass64')
    mode=mode.removesuffix('_mass64')
    if mode=='packed':
        import ast
        prototype=ROOT/'artifacts/video_runtime_v1/packed_probability_v1/probability.py'
        split=next(ast.literal_eval(n.value) for n in ast.parse(prototype.read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='split')
        code=code.replace('@triton.jit\ndef _anchor_av',split+'\n@triton.jit\ndef _anchor_av')
        code=code.replace('high = prob.to(values.dtype)\n    low = (prob - high.to(tl.float32)).to(values.dtype)','high,low=split_pair(prob)')
    elif mode in ('twosum','twoearly','twoav'):
        start=code.index('    for i in tl.range(1, count):')
        stop=code.index('    return accum / total',start)
        loop=code[start:stop].split('\n',1)[1]
        part=loop[:loop.index('        total +=')]
        import re
        def page(index,idx):
            return re.sub(r'\b(tokens|prob|scores)\b',lambda m:m[0]+index,part.replace('ids + i','ids + '+idx))
        first=page('0','2*i+1')
        second=page('1','2*i+2')
        if mode=='twoearly':
            first+='        s0=tl.sum(prob0,1)\n'
            second+='        s1=tl.sum(prob1,1)\n'
        else:
            second+='        s0,s1=tl.split(tl.sum(tl.join(prob0,prob1),1))\n'
        if mode=='twoav':
            av=first[first.index('        accum +='):]
            first=first[:first.index('        accum +=')]
            second=second.replace('        accum +=',av+'        accum +=')
        code=code[:start]+'    for i in tl.range((count-1)//2):\n'+first+second+'        total=(total+s0)+s1\n    if (count-1)%2:\n'+loop.replace('ids + i','ids + count-1')+code[stop:]
    elif mode.startswith('mass'):
        groups=int(mode[4:])
        reduce=f'tl.sum(tl.reshape(prob,(64,{groups},{64//groups})),2)' if groups<64 else 'prob'
        code=code.replace('total = tl.sum(prob, 1)','partial = '+reduce)
        code=code.replace('total += tl.sum(prob, 1)','partial += '+reduce)
        code=code.replace('    return accum / total','    total=tl.sum(partial,1)\n    return accum / total')
    elif mode in ('nested','nestedlow'):
        x,y=('high','low') if mode=='nested' else ('low','high')
        code=code.replace('tl.dot(high, values) + tl.dot(low, values)',f'tl.dot({y},values,tl.dot({x},values))')
    else:
        loop='        accum += _anchor_av(prob, tl.load(values + tokens[:, None] * stride + dims[None, :]))'
        common='''        val=tl.load(values+tokens[:,None]*stride+dims[None,:])
        high=prob.to(val.dtype)
        low=(prob-high.to(tl.float32)).to(val.dtype)
'''
        updates={'chain':'accum=tl.dot(low,val,tl.dot(high,val,accum))','reverse':'accum=tl.dot(high,val,tl.dot(low,val,accum))','chainhi':'accum=tl.dot(high,val,accum)+tl.dot(low,val)','chainlo':'accum=tl.dot(low,val,accum)+tl.dot(high,val)','split':'accum=tl.dot(high,val,accum)\n        residual=tl.dot(low,val,residual)'}
        code=code.replace(loop,common+'        '+updates[mode])
        if mode=='split':
            code=code.replace('    for i in tl.range','    residual=tl.full((64,D),0.,tl.float32)\n    for i in tl.range').replace('    return accum / total','    return (accum+residual) / total')
    if accumulate_mass:
        code=code.replace('total = tl.sum(prob, 1)','partial = prob').replace('total += tl.sum(prob, 1)','partial += prob').replace('    return accum / total','    total=tl.sum(partial,1)\n    return accum / total')
        mode+='_mass64'
    name='anchor_accum_'+mode
    Path(folder.name,name+'.py').write_text(code)
    src=src.replace('from local_experiments.gap_exact_round_kernel import _anchored_exact','from '+name+' import _anchored_exact')
    tail='tail_accum_'+mode
    Path(folder.name,tail+'.py').write_text(src)
    return importlib.import_module(tail)._tail
