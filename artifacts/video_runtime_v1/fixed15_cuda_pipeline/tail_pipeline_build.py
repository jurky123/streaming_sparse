from pathlib import Path
root=Path(__file__).parent
s=(root/'tail.py').read_text()
i=s.index(' for first in range')
j=s.index('  nm=tl.maximum',i)
body=s[s.index('  rep=first*BR',i):j]
iword=body.index('  if EXACT_COUNT:')
jword=body.index('  key=tl.load',iword)
body=body[:iword]+'''  word=tl.sum(tl.where(tl.arange(0,16)==first//2,words,0),0)
  omitted=((word>>(rep//4%32))&1)==0
'''+body[jword:]
helper='''@tr.jit
def next_logits(q,KEY,words,h,first,hp:tl.constexpr,P:tl.constexpr,BR:tl.constexpr,D:tl.constexpr):
  r=tl.arange(0,BR);d=tl.arange(0,D);pages=tl.arange(0,BR//4)
'''+body+'  return z\n\n'
tail=s[:i]+''' z=next_logits(q,KEY,words,h,0,hp,P,BR,D)
 for first in range(tr.cdiv(P*4,BR)):
  page=first*(BR//4)+pages
'''+s[j:]
i=tail.index('  alpha=tl.sum')
tail=tail[:i]+'''  if first+1<tr.cdiv(P*4,BR):
   z=next_logits(q,KEY,words,h,first+1,hp,P,BR,D)
'''+tail[i:]
(root/'tail_pipeline.py').write_text(tail.replace('@tr.jit\ndef complete',helper+'@tr.jit\ndef complete'))
