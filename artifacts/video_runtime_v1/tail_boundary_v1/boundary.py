import importlib
from pathlib import Path
import sys
import tempfile

_folder=tempfile.TemporaryDirectory()
sys.path.insert(0,_folder.name)

def variant_for(src,mode):
    start=src.index(' for first in range(tr.cdiv(P*4,BR)):')
    end=src.index(' safe=tl.maximum',start)
    body=src[src.index('\n',start)+1:end]
    full=body.replace(',rep[None,:]<P*4,0)',')').replace(',page[:,None]<P,0)',')')
    full=full.replace('(page[:,None]<P)&(rank[None,:]<8)','rank[None,:]<8').replace('(rep[None,:]<P*4)&omitted','omitted')
    if mode=='aligned':
        block=' if P*4%BR==0:\n  for first in range(P*4//BR):\n'+indent(full,1)+' else:\n  for first in range(tr.cdiv(P*4,BR)):\n'+indent(body,1)
    else:
        start_index=0
        block=''
        if mode=='firstlast':
            initial=full.replace('tl.maximum(mx,tl.max(z,1))','tl.max(z,1)').replace("tl.max(z,1);fade=","tl.maximum(-1.e30,tl.max(z,1));fade=")
            # If the entire first tile is selected, every omitted mass is zero.
            initial=initial.replace('am*fade[:,None]+','').replace('ac*fade[:,None]+','').replace('den*fade+','')
            block=' first=0\n'+indent(initial,-1)
            start_index=1
        block+=f' for first in range({start_index},P*4//BR):\n'+full
        tail=body
        if mode=='half':
            a=tail.index('  key=tl.load(')
            b=tail.index('  z=tl.where',a)
            tail=tail[:a]+'''  small=first*BR+tl.arange(0,32)
  key=tl.load(KEY+(h*hp*4+small[None,:])*D+d[:,None])
  partial=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D))
  pad=tl.full((BQ,32),-float('inf'),tl.float32)
  z=tl.reshape(tl.permute(tl.join(partial,pad),(0,2,1)),(BQ,64))
'''+tail[b:]
        block+=' if P*4%BR:\n  first=P*4//BR\n'+tail
    if mode.startswith('mixed'):
        import ast
        start=src.index('@tr.jit\ndef _tail')
        header=src[start:src.index(' hp:tl.constexpr',start)]
        fwd=src[start:].replace('def _tail(', 'def _exact_first(',1)
        rev=fwd.replace('def _exact_first(', 'def _exact_last(',1)
        begin=rev.index('  exact,le=_attn_fwd')
        end=rev.index('\n',begin)
        call=rev[begin:end]
        rev=rev[:begin]+rev[end:]
        rev=rev.replace('  lt=tl.where',call+'\n  lt=tl.where')
        argnames=','.join(a.arg for a in ast.parse(fwd).body[0].args.args)
        group=int(mode[5:])
        src=src[:start]+fwd+'\n'+rev+'\n'+header+f' if (tl.program_id(0)//{group})%2==0:\n  _exact_first({argnames})\n else:\n  _exact_last({argnames})\n'
    elif mode.startswith('spill'):
        a=src.index(' mx=tl.full')
        if mode=='spillfull':
            stores='  addr=MU+(h*N+qi[:,None])*D+d[None,:]\n  tl.store(addr,exact)\n'
            loads='  exact=tl.load(addr,volatile=True)\n'
        else:
            stores='  left,right=tl.split(tl.reshape(exact,(BQ,D//2,2)))\n  addr=MU+(h*N+qi[:,None])*(D//2)+tl.arange(0,D//2)[None,:]\n  tl.store(addr,left)\n'
            loads='  left=tl.load(addr,volatile=True)\n  exact=tl.reshape(tl.join(left,right),(BQ,D))\n'
        src=src[:a]+stores+src[a:]
        src=src.replace('  lt=tl.where',loads+'  lt=tl.where')
    elif mode in ('readmask','wordpair'):
        a=src.index(' if EXACT_COUNT:\n  slots=')
        src=src[:a]+src[start:]
        src=src.replace('word=tl.sum(tl.where(tl.arange(0,16)==first//2,words,0),0)','word=tl.load(MASK+(h*(N//64)+t)*16+first//2)')
        if mode=='wordpair':
            i=src.index(' for first in range(tr.cdiv(P*4,BR)):')
            j=src.index(' safe=tl.maximum',i)
            body=src[src.index('  rep=',i):j]
            a=body.index('  if EXACT_COUNT:')
            b=body.index('  omitted=',a)
            body=body[:a]+body[b:]
            main=' for pair in range(tr.cdiv(P*4,BR)//2):\n  word=tl.load(MASK+(h*(N//64)+t)*16+pair)\n  for sub in tl.static_range(2):\n   first=pair*2+sub\n'+indent(body,1)
            main+=' if tr.cdiv(P*4,BR)%2:\n  first=tr.cdiv(P*4,BR)-1\n  word=tl.load(MASK+(h*(N//64)+t)*16+first//2)\n'+body
            src=src[:i]+main+src[j:]
    else:
        src=src[:start]+block+src[end:]
    name='boundary_'+mode
    Path(_folder.name,name+'.py').write_text(src)
    return importlib.import_module(name)._tail

def indent(src,n):
    return ''.join((' '*n+line if n>=0 else line[-n:]) for line in src.splitlines(True))
