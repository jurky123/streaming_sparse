"""Specialize the exact-page loop for isolated softmax scaling experiments."""
import importlib
from pathlib import Path
import sys
import tempfile
import torch
from local_experiments.video_runtime_v1.weights import ROOT

folder=tempfile.TemporaryDirectory()
sys.path.insert(0,folder.name)


def variant_for(src,mode):
    kernel=(ROOT/'local_experiments/gap_exact_round_kernel.py').read_text()
    kernel=kernel.split('@triton.jit\ndef _anchor_logits')[0]
    if '  if ANCHOR_SCALE:' in src:
        start=src.index('  if ANCHOR_SCALE:')
        end=src.index(' mx=tl.full',start)
        baseline_call=src[start:end].split('  else:\n')[-1].strip()
        src=src[:start]+'  '+baseline_call+'\n'+src[end:]
        src=src.replace('from local_experiments.gap_exact_round_kernel import _anchored_exact\n','')
    start=kernel.index('        local_m =')
    end=kernel.index('        m_i = new_m',start)
    prefix=kernel[:start]
    body=kernel[start:end]
    suffix=kernel[end:]
    if mode in ('fixed','fixed16','first','guarded','guard16','repair'): 
        scale={'fixed':'0.', 'fixed16':'16.', 'first':'m_i','guarded':'0.','guard16':'16.','repair':'0.'}[mode]
        if mode=='first':
            scale='tl.max(qk,1) if block_idx==0 else m_i'
        else:
            scale='tl.full((BLOCK_M,),'+scale+',tl.float32)'
        body=body.replace('local_m = tl.max(qk, 1)\n        new_m = tl.maximum(m_i, local_m)','new_m = '+scale).replace('alpha = tl.math.exp2(m_i - new_m)','alpha = 1.')
        body=body.replace('tl.fma(o_s,alpha[:,None],page_av)','o_s+page_av').replace('tl.fma(l_i,alpha,l_ij)','l_i+l_ij')
    elif mode in ('local','localint'):
        body=body.replace('qk = qk - new_m[:, None]','qk = qk - local_m[:, None]')
        if mode=='localint':body=body.replace('local_m = tl.max(qk, 1)','local_m = tl.ceil(tl.max(qk, 1))')
        body=body.replace('tl.fma(o_s,alpha[:,None],page_av)','tl.fma(o_s,alpha[:,None],page_av*tl.exp2(local_m-new_m)[:,None])').replace('tl.fma(l_i,alpha,l_ij)','tl.fma(l_i,alpha,l_ij*tl.exp2(local_m-new_m))')
    elif mode=='integer':
        body=body.replace('local_m = tl.max(qk, 1)','local_m = tl.ceil(tl.max(qk, 1))')
    elif mode in ('same','sameav','sameall'):
        test='tl.max((local_m>m_i).to(tl.int32),0)!=0'
        body=body.replace('new_m = tl.maximum(m_i, local_m)','changed = '+test+'\n        new_m = tl.maximum(m_i, local_m)')
        a='            o_s = tl.fma(o_s,alpha[:,None],page_av)\n            l_i = tl.fma(l_i,alpha,l_ij)'
        b='            if changed:\n    '+a.replace('\n','\n    ')+'\n            else:\n                o_s=o_s+page_av\n                l_i=l_i+l_ij'
        body=body.replace(a,b)
        if mode=='sameall':
            body=body.replace('new_m = tl.maximum(m_i, local_m)','new_m = tl.maximum(m_i,local_m) if changed else m_i')
        if mode!='same':
            body=body.replace('alpha = tl.math.exp2(m_i - new_m)','').replace('                o_s = tl.fma','                alpha=tl.exp2(m_i-new_m)\n                o_s = tl.fma')
    elif mode=='peel':
        k=prefix.index('    for block_idx in tl.range(topk):')
        pre,loop=prefix[:k],prefix[k:]
        one=loop.split('\n',1)[1]+body+'        m_i = new_m\n'
        one=one.replace('block_idx','0').replace('tl.maximum(m_i, local_m)','local_m').replace('tl.fma(o_s,alpha[:,None],page_av)','page_av').replace('tl.fma(l_i,alpha,l_ij)','l_ij')
        prefix=pre+'    if topk>0:\n'+one+loop.replace('tl.range(topk)','tl.range(1,topk)')
    elif mode in ('pair','interleave','quad'):
        group=4 if mode=='quad' else 2
        width=64*group
        prefix=prefix.replace('offs_n = tl.arange(0, BLOCK_N)',f'offs_n = tl.arange(0, {width})')
        a=prefix.index('    for block_idx in tl.range(topk):')
        b=prefix.index('        k = tl.load(K_ptrs)',a)
        which=f'offs_n//64' if mode!='interleave' else 'offs_n%2'
        offset='offs_n%64' if mode!='interleave' else 'offs_n//2'
        prefix=prefix[:a]+f"""    for block_idx in tl.range(triton.cdiv(topk,{group})):
        pos=block_idx*{group}+{which}
        selected=tl.load(LUT_ptr+pos,pos<topk,0).to(tl.int32)
        token=selected*64+{offset}
        K_ptrs=K+kv_offset+token[None,:]*kv_stride+offs_d[:,None]
        V_ptrs=V+kv_offset+token[:,None]*kv_stride+offs_d[None,:]
"""+prefix[b:]
        prefix=prefix.replace('        v = tl.load(V_ptrs)','        qk=tl.where(pos[None,:]<topk,qk,-float("inf"))\n        v = tl.load(V_ptrs)')
    elif mode in ('adaptive','adaptive64','adaptive_peak'):
        limit=64 if mode=='adaptive64' else 32
        prefix=prefix.replace('tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)','tl.full([BLOCK_M], 0., dtype=tl.float32)')
        if mode=='adaptive_peak':
            begin=f"""        peak=tl.max(qk,1)
        adjust=(peak-m_i>{limit}.)|((l_i==0.)&(peak-m_i<-{limit}.))
        if tl.max(adjust.to(tl.int32),0)!=0:
            new_m=tl.where(adjust,peak,m_i)
            alpha=tl.math.exp2(m_i-new_m)
            alpha=tl.where(l_i==0.,0.,alpha)
            o_s=o_s*alpha[:,None]
            l_i=l_i*alpha
        else:
            new_m=m_i
        p=tl.math.exp2(qk-new_m[:,None])
        l_ij=tl.sum(p,1)
"""
        else:
            begin=f"""        p=tl.math.exp2(qk-m_i[:,None])
        l_ij=tl.sum(p,1)
        adjust=(l_ij>{2.**limit})|((l_i==0.)&(l_ij<{2.**(-limit)}))
        if tl.max(adjust.to(tl.int32),0)!=0:
            new_m=tl.where(adjust,tl.max(qk,1),m_i)
            alpha=tl.where(l_i==0.,0.,tl.math.exp2(m_i-new_m))
            o_s=o_s*alpha[:,None]
            l_i=l_i*alpha
            p=tl.math.exp2(qk-new_m[:,None])
            l_ij=tl.sum(p,1)
        else:
            new_m=m_i
"""
        body=begin+body[body.index('        if RETURN_VALUES:'):]
        body=body.replace('tl.fma(o_s,alpha[:,None],page_av)','o_s+page_av').replace('tl.fma(l_i,alpha,l_ij)','l_i+l_ij')
    elif mode in ('anchor','anchor_guard','anchor_guard_named','anchor_repair'):
        loc=prefix.index('    for block_idx in tl.range(topk):')
        head,loop=prefix[:loc],prefix[loc:]
        initial=loop.split('\n',1)[1]+body+'        m_i = new_m\n'
        initial=initial.replace('block_idx','0').replace('tl.maximum(m_i, local_m)','local_m').replace('tl.fma(o_s,alpha[:,None],page_av)','page_av').replace('tl.fma(l_i,alpha,l_ij)','l_ij')
        prefix=head+'    if topk>0:\n'+initial+loop.replace('tl.range(topk)','tl.range(1,topk)')
        body=body[body.index('        qk = qk - new_m'):].replace('new_m[:, None]','m_i[:, None]').replace('        alpha = tl.math.exp2(m_i - new_m)','        alpha=tl.full((BLOCK_M,),1.,tl.float32)')
        body=body.replace('tl.fma(o_s,alpha[:,None],page_av)','o_s+page_av').replace('tl.fma(l_i,alpha,l_ij)','l_i+l_ij')
        suffix=suffix.replace('        m_i = new_m','',1)
        if mode!='anchor':
            suffix=suffix.replace('return o_s, m_i + tl.math.log2(l_i)','return o_s, m_i + tl.math.log2(l_i), l_i')
    else:
        raise ValueError(mode)
    name='page_scale_'+mode
    source=prefix+body+suffix
    if mode=='anchor_guard_named':source=source.replace('def _attn_fwd(', 'def _anchored_exact(')
    Path(folder.name,name+'.py').write_text(source)
    tail='tail_scale_'+mode
    src=src.replace('from local_experiments.gap_exact_round_kernel import',f'from {name} import')
    if mode in ('anchor_guard','anchor_guard_named','anchor_repair'):
        src=src.replace('exact,le=_attn_fwd(','exact,le,denom=_attn_fwd(')
    if mode in ('guarded','guard16','anchor_guard','anchor_guard_named'):
        src=src.replace('import math','from local_experiments.gap_exact_round_kernel import _attn_fwd as stable\nimport math')
        call=next(line for line in src.splitlines() if '=_attn_fwd(' in line)
        check='  good=(le>-40.)&(le<40.)\n  if tl.min(good.to(tl.int32),0)==0:\n '
        if mode.startswith('anchor_guard'):check=check.replace('(le>-40.)&(le<40.)','(denom>=1.)&(denom<1.21e24)')
        src=src.replace(call,call+'\n'+check+call.replace('_attn_fwd','stable').replace('exact,le,denom=', 'exact,le='))
    if mode in ('repair','anchor_repair'):
        call=next(line for line in src.splitlines() if '=_attn_fwd(' in line)
        write_flag='  tl.store(MASK+h*(N//BQ)+t,tl.min(((le>-40.)&(le<40.)).to(tl.int32),0)==0)'
        if mode=='anchor_repair':write_flag=write_flag.replace('(le>-40.)&(le<40.)','(denom>=1.)&(denom<1.21e24)')
        src=src.replace(call,call+'\n'+write_flag)
        slow=src.replace('from '+name+' import','from local_experiments.gap_exact_round_kernel import').replace(write_flag,'').replace('exact,le,denom=', 'exact,le=')
        slow=slow.replace(' q=tl.load(Q+', ' if tl.load(MASK+h*(N//BQ)+t)==0:\n  return\n q=tl.load(Q+',1)
        Path(folder.name,'repair_slow.py').write_text(slow)
    if mode=='anchor_guard_named':src=src.replace('from '+name+' import _attn_fwd','from '+name+' import _anchored_exact as _attn_fwd')
    Path(folder.name,tail+'.py').write_text(src)
    fast=importlib.import_module(tail)._tail
    if mode not in ('repair','anchor_repair'):return fast
    slow=importlib.import_module('repair_slow')._tail
    class Repaired:
        def __getitem__(self,grid):
            def run(*args,**options):
                if not options.get('EXACT_COUNT'):
                    return native[grid](*args,**options)
                flags=torch.empty((grid[1],grid[0]),device=args[0].device,dtype=torch.uint8)
                args=(*args[:4],flags,*args[5:])
                comp=fast[grid](*args,**options)
                slow[grid](*args,**options)
                return comp
            return run
    from local_experiments.review_v1.tuned_attention import _tail as native
    return Repaired()
