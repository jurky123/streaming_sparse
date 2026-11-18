"""Measure alternative common scales for exact and omitted-page contributions."""
from pathlib import Path
import importlib
import sys
import tempfile
folder=tempfile.TemporaryDirectory()
sys.path.insert(0,folder.name)

def variant_for(src,mode):
    begin=src.index(' for first in range')
    end=src.index(' safe=tl.maximum(den',begin)
    original=src[begin:end]
    if mode.startswith('fixed'):
        loop=original.replace('nm=tl.maximum(mx,tl.max(z,1));fade=tl.exp2(mx-nm);w=tl.exp2(z-nm[:,None])','w=tl.exp2(z-mx[:,None])')
        loop=loop.replace('am=am*fade[:,None]+','am=am+').replace('ac=ac*fade[:,None]+','ac=ac+')
        loop=loop.replace('den=den*fade+tl.sum(w,1);mx=nm','den+=tl.sum(w,1)')
        if 'col' in mode:
            loop=' columns=tl.full((BQ,BR),0.,tl.float32)\n'+loop.replace('den+=tl.sum(w,1)','columns+=w')+' den=tl.sum(columns,1)\n'
        init=' mx=le-4.\n'
        if 'first' in mode:
            # Even a fully selected first block provides finite unmasked scores.
            before=loop[:loop.index('  z=tl.where')].replace(' for first in range(tr.cdiv(P*4,BR)):', ' if P>0:\n  first=0')
            init=before+'  mx=tl.max(z,1)\n'
        if 'postmask' in mode or 'fma' in mode:
            predicate='(rep[None,:]<P*4)&omitted[None,:]'
            loop=loop.replace("  z=tl.where("+predicate+",z,-float('inf'))\n","")
            if 'fma' in mode:
                loop=loop.replace('z=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D))','raw=tl.dot(q,key)')
                expr='tl.exp2(tl.fma(raw,1.4426950408889634/math.sqrt(D),-mx[:,None]))'
            else:expr='tl.exp2(z-mx[:,None])'
            loop=loop.replace('w=tl.exp2(z-mx[:,None])','w=tl.where('+predicate+','+expr+',0.)')
        if 'guard' in mode:
            reset='''  mx=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32)
  am=tl.zeros((BQ,D),tl.float32);ac=tl.zeros((BQ,16),tl.float32)
'''
            loop+=' if tl.min(((den>1.e-20)&(den<1.e20)).to(tl.int32),0)==0:\n'+reset+''.join(' '+line for line in original.splitlines(keepends=True))
        src=src[:begin]+init+loop+src[end:]
    elif mode.startswith('seed') or mode=='joint':
        src=src.replace(' mx=tl.full((BQ,),-1.e30,tl.float32);den=', ' mx=le-4.;den=')
    if mode.startswith('cols'):
        src=src.replace(' for first in range',' columns=tl.full((BQ,BR),0.,tl.float32)\n for first in range')
        src=src.replace('den=den*fade+tl.sum(w,1);mx=nm','columns=columns*fade[:,None]+w;mx=nm\n den=tl.sum(columns,1)')
    if 'alphasum' in mode:
        src=src.replace('tl.sum(w,1)','tl.sum(alpha,1)')
    if 'pagetotal' in mode:
        # Change only the fast loop; the recovery retains its original reduction.
        a=src.index(' for first in range')
        b=src.index(' safe=tl.maximum',a)
        loop=src[a:b]
        end=loop.find(' if tl.min')
        fast,guard=(loop[:end],loop[end:]) if end>=0 else (loop,'')
        fast=fast.replace('den=den*fade+tl.sum(w,1);mx=nm','part=part*fade[:,None]+alpha;mx=nm')
        fast=fast.replace('den+=tl.sum(w,1)','part+=alpha')
        src=src[:a]+' part=tl.full((BQ,BR//4),0.,tl.float32)\n'+fast+' den=tl.sum(part,1)\n'+guard+src[b:]
    if mode in ('joint','fixedjoint','fixedcoljoint'):
        a=src.index(' safe=tl.maximum(den')
        b=src.index('\n else:\n  tl.store(MU',a)
        src=src[:a]+''' if MERGE:
  exact_mass=tl.exp2(le-mx-4.)
  combined=exact_mass+den
  code=tl.where(rank[None,:]<8,ac/combined[:,None],0.)
  basis=tl.load(BASIS+(h*D+d[None,:])*8+rank[:,None],rank[:,None]<8,0).to(tl.float32)
  out=exact*(exact_mass/combined)[:,None]+am/combined[:,None]+tl.dot(code,basis,input_precision='tf32x3')
  tl.store(OUT+(qi[:,None]*H+h)*D+d[None,:],out)
'''+src[b:]
    if mode.startswith('adapt'):
        limit=32 if '32' in mode else 64
        src=src.replace(' mx=tl.full((BQ,),-1.e30,tl.float32);den=', ' mx=le-4.;den=')
        original='nm=tl.maximum(mx,tl.max(z,1));fade=tl.exp2(mx-nm);w=tl.exp2(z-nm[:,None]);'
        if 'peak' in mode:
            begin=f"""peak=tl.max(z,1)
  adjust=peak-mx>{limit}.
  if tl.max(adjust.to(tl.int32),0)!=0:
   nm=tl.where(adjust,peak,mx)
   fade=tl.exp2(mx-nm)
   den=den*fade
   ac=ac*fade[:,None]
   am=am*fade[:,None]
   mx=nm
  w=tl.exp2(z-mx[:,None])
  """
        else:
            begin=f"""w=tl.exp2(z-mx[:,None])
  page_mass=tl.sum(w,1)
  adjust=page_mass>{2.**limit}
  if tl.max(adjust.to(tl.int32),0)!=0:
   nm=tl.where(adjust,tl.max(z,1),mx)
   fade=tl.exp2(mx-nm)
   den=den*fade
   ac=ac*fade[:,None]
   am=am*fade[:,None]
   mx=nm
   w=tl.exp2(z-mx[:,None])
   page_mass=tl.sum(w,1)
  """
        src=src.replace('am=am*fade[:,None]+','am=am+').replace('ac=ac*fade[:,None]+','ac=ac+')
        src=src.replace('den=den*fade+tl.sum(w,1);mx=nm', 'den+=tl.sum(w,1)' if 'peak' in mode else 'den+=page_mass')
        src=src.replace(original,begin)
    if mode.startswith('recip'):
        helper="""
@tr.jit
def _row_inverse(x):
 return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0,$1;",constraints="=f,f",args=[x],dtype=tl.float32,is_pure=True,pack=1)
"""
        if 'rn' in mode:helper=helper.replace('tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0,$1;",constraints="=f,f",args=[x],dtype=tl.float32,is_pure=True,pack=1)','tl.div_rn(1.,x)')
        if 'exact' in mode or 'both' in mode:
            from local_experiments.video_runtime_v1.weights import ROOT
            anchor=(ROOT/'local_experiments/gap_exact_round_kernel.py').read_text()
            anchor=anchor[:anchor.index('@triton.jit')]+helper.replace('@tr.jit','@triton.jit')+anchor[anchor.index('@triton.jit\ndef _anchor_logits'):]
            anchor=anchor.replace('return accum / total[:, None]','return accum * _row_inverse(total)[:, None]')
            mod='anchor_'+mode
            Path(folder.name,mod+'.py').write_text(anchor)
            src=src.replace('from local_experiments.gap_exact_round_kernel import _anchored_exact','from '+mod+' import _anchored_exact')
        if 'tail' in mode or 'both' in mode:
            src=src.replace('@tr.jit\ndef _tail',helper+'\n@tr.jit\ndef _tail')
            src=src.replace(' if MERGE:\n  code=', ' if MERGE:\n  inv=_row_inverse(safe)\n  code=')
            src=src.replace('ac/safe[:,None],0.)','ac*inv[:,None],0.)').replace('tail=am/safe[:,None]','tail=am*inv[:,None]')
    wide=mode.startswith('wide')
    if wide:
        width=int(mode.split('_')[0][4:])
        src=src.replace('(BQ == 64 and BR == 64)','(BQ == 64)')
        if width==128:
            src=src.replace('==first//2,words','==first,words')
        elif width==256:
            src=src.replace('word=tl.sum(tl.where(tl.arange(0,16)==first//2,words,0),0)','w0=tl.sum(tl.where(tl.arange(0,16)==first*2,words,0),0)\n   w1=tl.sum(tl.where(tl.arange(0,16)==first*2+1,words,0),0)\n   word=tl.where(r<128,w0,w1)')
    name='compensation_'+mode
    Path(folder.name,name+'.py').write_text(src)
    kernel=importlib.import_module(name)._tail
    if not wide:return kernel
    class WideTile:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                kwargs['BR']=width
                if 'w8' in mode:kwargs['num_warps']=8
                if 's1' in mode:kwargs['num_stages']=1
                return kernel[grid](*args,**kwargs)
            return launch
    return WideTile()
