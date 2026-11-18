"""Generate fused attention variants for eight-channel detail projections."""
import importlib
from pathlib import Path
import sys
import tempfile

folder=tempfile.TemporaryDirectory()
sys.path.insert(0,folder.name)

def variant_for(src,mode):
    compact='narrow' in mode or 'simt' in mode
    if compact:
        if 'simt' not in mode:
            from triton.backends import backends
            compiler=backends['nvidia'].compiler
            compiler.get_codegen_implementation.__globals__['min_dot_size']=lambda target: lambda lhs,rhs: (16,8,16)
        src=src.replace('rank=tl.arange(0,16)','rank=tl.arange(0,8)').replace('ac=tl.zeros((BQ,16)','ac=tl.zeros((BQ,8)')
        src=src.replace('  code=tl.where(rank[None,:]<8,ac/safe[:,None],0.)', '''  rank=tl.arange(0,16)
  wide=tl.reshape(tl.permute(tl.join(ac,ac),(0,2,1)),(BQ,16))
  code=tl.where(rank[None,:]<8,wide/safe[:,None],0.)''')
    original='ac=ac*fade[:,None]+_av(c0,x)+_av(c1,y)+_av(c2,v)'
    if mode.startswith(('pair','triple')):
        combined="""coeff=tl.reshape(tl.join(c0,c1),(BQ,BR//2))
   g=tl.arange(0,BR//2)
   rp=first*(BR//4)+g//2
   rhs=tl.load(DETAIL+((h*hp+rp[:,None])*3+(g%2)[:,None])*8+rank[None,:],(rp[:,None]<P)&(rank[None,:]<8),0)
   ac=ac*fade[:,None]+_av(coeff,rhs)+_av(c2,v)"""
        if mode.startswith('triple'):
            combined="""coeff=tl.reshape(tl.join(tl.reshape(tl.join(c0,c2),(BQ,BR//2)),tl.reshape(tl.join(c1,c2*0.),(BQ,BR//2))),(BQ,BR))
   g=tl.arange(0,BR)
   rp=first*(BR//4)+g//4
   rhs=tl.load(DETAIL+((h*hp+rp[:,None])*3+(g%4)[:,None])*8+rank[None,:],(rp[:,None]<P)&((g%4)[:,None]<3)&(rank[None,:]<8),0)
   ac=ac*fade[:,None]+_av(coeff,rhs)"""
        src=src.replace(original,combined)
    if mode=='simt':
        helper="""
@tr.jit
def _small(a,b):
 hi=a.to(tl.bfloat16).to(tl.float32)
 lo=(a-hi).to(tl.bfloat16).to(tl.float32)
 return tl.sum(hi[:,:,None]*b[None,:,:],1)+tl.sum(lo[:,:,None]*b[None,:,:],1)

"""
        src=src.replace('@tr.jit\ndef _tail',helper+'@tr.jit\ndef _tail').replace(original,original.replace('_av','_small'))
    if mode.startswith('acc'):
        helper="""
@tr.jit
def _accum(a,b,c):
 hi=a.to(tl.bfloat16)
 lo=(a-hi.to(tl.float32)).to(tl.bfloat16)
 return tl.dot(lo,b.to(tl.bfloat16),tl.dot(hi,b.to(tl.bfloat16),c))

"""
        if mode.endswith('lowfirst'):helper=helper.replace('tl.dot(lo,b.to(tl.bfloat16),tl.dot(hi,','tl.dot(hi,b.to(tl.bfloat16),tl.dot(lo,')
        src=src.replace('@tr.jit\ndef _tail',helper+'@tr.jit\ndef _tail')
        if mode!='acc_mean':
            src=src.replace(original,'ac=_accum(c2,v,_accum(c1,y,_accum(c0,x,ac*fade[:,None])))')
        if mode!='acc_detail':
            src=src.replace('am=am*fade[:,None]+_av(alpha,mean)','am=_accum(alpha,mean,am*fade[:,None])')
    name='detail_'+mode
    Path(folder.name,name+'.py').write_text(src)
    return importlib.import_module(name)._tail
