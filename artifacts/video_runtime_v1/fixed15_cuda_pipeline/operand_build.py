from pathlib import Path
p=Path(__file__).parent
baseline=(p/'paired_k.cu').read_text()
old=(p.parent/'fixed15_baseline_v1/warp_exact.cu').read_text()
for mode in ('rawkey','rawval','rawboth'):
    s=baseline
    # The tail always consumes the packed summary. Only exact K/V traffic changes.
    off=s.index('__global__ void exact')
    prefix,s=s[:off],s[off:]
    pos=s.index('            for(int m=threadIdx.x')
    end=s.index('\n        }\n        asm volatile',pos)
    copy=s[pos:end]
    for operand,idx,ptr in [('K',0,'fk'),('V',1,'fv')]:
        raw=(mode=='rawboth' or (mode=='rawkey')==(operand=='K'))
        if not raw:continue
        copy='\n'.join(l for l in copy.splitlines() if not ('"l"('+ptr+'+m)' in l or f'&slots[slot][{idx}][m]' in l))
        copy+=f'\n            stage(reinterpret_cast<bf*>(slots[slot][{idx}]),{operand},page,h);'
    s=s[:pos]+copy+s[end:]
    if mode!='rawval':
        i=s.index('                uint4 u=');j=s.index('                mma(score',i)
        a=old.index('                unsigned addr=');b=old.index('                mma(score',a)
        s=s[:i]+old[a:b].replace('kv[0]+tile','reinterpret_cast<bf*>(kv[0])+tile')+s[j:]
    if mode!='rawkey':
        i=s.index('                uint4 u=reinterpret_cast<const uint4*>(kv[1])');j=s.index('                mma(tmp',i)
        a=old.index('                unsigned addr=',old.index('Four tmp[2]'));b=old.index('                mma(tmp',a)
        s=s[:i]+old[a:b].replace('kv[1]+tile','reinterpret_cast<bf*>(kv[1])+tile')+s[j:]
    (p/f'{mode}.cu').write_text(prefix+s)
