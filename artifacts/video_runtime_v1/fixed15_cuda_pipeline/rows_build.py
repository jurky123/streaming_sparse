from pathlib import Path
root=Path(__file__).parent
s=(root/'tail.py').read_text()
i=s.index(' if EXACT_COUNT:')
j=s.index(' mx=',i)
guard=s[s.index('  row=',i):j]
s=s[:i]+''' groups:tl.constexpr=BQ//64
 group=tl.arange(0,groups)
 lut=IDS+(h*(N//64)+t*groups+group[:,None])*EXACT_COUNT
'''+s[j:]
i=s.index('  ids=tl.load(');j=s.index(' for first',i)
s=s[:i]+'''  ids=tl.load(lut+slots[None,:],slots[None,:]<EXACT_COUNT,0)
  bit=tl.where(slots[None,:]<EXACT_COUNT,1<<(ids%32),0).to(tl.uint32)
  words=tl.sum(tl.where(tl.arange(0,16)[None,:,None]==ids[:,None,:]//32,bit[:,None,:],0),2)
'''+s[j:]
i=s.index('  if EXACT_COUNT:',s.index(' for first'));j=s.index('  key=tl.load',i)
s=s[:i]+'''  word=tl.sum(tl.where(tl.arange(0,16)[None,:]==first//2,words,0),1)
  omitted=((word[:,None]>>(rep[None,:]//4%32))&1)==0
  keep=tl.reshape(tl.broadcast_to(omitted[:,None,:],(groups,64,BR)),(BQ,BR))
'''+s[j:]
s=s.replace('omitted[None,:]','keep').replace('  if not EXACT_COUNT:\n   le=tl.load(ELSE+h*N+qi)\n   exact=tl.load(EXACT+(qi[:,None]*H+h)*D+d[None,:])','  le=tl.load(ELSE+qi*H+h)\n  exact=tl.load(EXACT+(qi[:,None]*H+h)*D+d[None,:])')
s=s.replace('def complete','def row_tail')
s+='''
@tr.jit
def guard(Q,K,V,IDS,O,L,M,P:tl.constexpr,COUNT:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);qi=t*64+tl.arange(0,64)
 row=qi*12+h
 mass=tl.load(M+row)
 if tl.min(((mass>=1.)&(mass<1.21e24)).to(tl.int32),0)==0:
  e,le=_attn_fwd(Q,K,V,128**-.5,COUNT,IDS,O,O,12,4608,P*64,72,128,64,64,HIGH_PRECISION_AV=True,RETURN_VALUES=True)
  tl.store(O+row[:,None]*128+tl.arange(0,128)[None,:],e);tl.store(L+row,le)
'''
(root/'rows.py').write_text(s)
