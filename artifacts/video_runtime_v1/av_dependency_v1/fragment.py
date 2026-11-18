"""Probe SM80 word ownership, then substitute only a warp's own A fragment words."""
from pathlib import Path
import importlib.util
import json
import re
import sys
from triton.compiler.compiler import make_backend
from local_experiments.video_runtime_v1.page_prefetch import page_tail,coeff_tail
folder=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('warp_bench',folder.parent/'warp_rows_v1/measure.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
torch=bench.torch
mode,kind=sys.argv[1:3]

def modify(text):
    lines=text.splitlines()
    wait=next(i for i,l in enumerate(lines) if 'cp.async.wait_group 0x1' in l)
    def lastloc(col):
        return max(i for i,l in enumerate(lines[:wait]) if f'gap_exact_round_kernel.py:{col}' in l)
    hi_load=lastloc('151:19')
    hi_dot=lastloc('153:24')
    lo_load=lastloc('152:42')
    lo_dot=lastloc('153:46')
    start=max(i for i in range(hi_load) if 'gap_exact_round_kernel.py:151:19' in lines[i])
    stores=[i for i in range(start,hi_load) if 'st.shared.b32' in lines[i]]
    loads=[i for a,b in [(hi_load,hi_dot),(lo_load,lo_dot)] for i in range(a,b) if 'ldmatrix.sync' in lines[i]]
    assert len(stores)==len(loads)==32
    regs=[re.search(r', (%r\d+);',lines[i])[1] for i in stores]
    decl=['.reg .u32 %tidtag, %tag, %laneaddr, %warpno;', '.reg .pred %wp<4>;', 'mov.u32 %tidtag, %tid.x;', 'shr.u32 %warpno, %tidtag, 5;']
    decl += [f'setp.eq.u32 %wp{w}, %warpno, {w};' for w in range(4)]
    if kind=='probe':
        decl+=['.reg .u64 %dst, %byteoff;','ld.param.u64 %dst, [_tail_param_5];','mul.wide.u32 %byteoff, %tidtag, 512;', 'add.u64 %dst,%dst,%byteoff;']
        for j,i in enumerate(stores):
            lines[i]=f'mad.lo.u32 %tag,%tidtag,32,{j};\n'+lines[i].replace(regs[j]+';', '%tag;')
    else:
        mapping=json.loads((folder/f'{mode}_mapping.json').read_text())
    for j,i in enumerate(loads):
        line=lines[i]
        op=re.search(r'\{ ([^}]+?) \}, \[ (%r\d+) \+ 0 \]',line)
        dst,addr=op[1].split(', '),op[2]
        if kind=='probe':
            lines[i]+='\n'+'\n'.join(f'st.global.u32 [%dst+{16*j+4*n}],{r};' for n,r in enumerate(dst))
            continue
        own=mapping[j]
        # Nonowners consume all four 8x8 matrices. Each owner consumes two
        # remote matrices using x2, and takes two packed words from registers.
        prefix=[]
        if kind=='coalesced':
            # Two owners differ only in which half is local. Issue one x2
            # for either owner and one x4 for either nonowner.
            first=next(w for w,words in enumerate(own) if words and words[0] is not None)
            second=next(w for w,words in enumerate(own) if words and words[2] is not None)
            prefix += ['{', '.reg .pred %owns;', '.reg .b32 %remote<2>, %ptr;',
                       f'or.pred %owns, %wp{first}, %wp{second};',
                       f'@%wp{first} shfl.sync.bfly.b32 %laneaddr,{addr},16,31,0xffffffff;',
                       f'selp.b32 %ptr, %laneaddr, {addr}, %wp{first};',
                       '@!%owns '+line.strip(),
                       '@%owns ldmatrix.sync.aligned.m8n8.x2.shared.b16 { %remote0, %remote1 }, [ %ptr ];']
            for n in range(4):
                left=regs[own[first][n]] if n<2 else f'%remote{n-2}'
                right=regs[own[second][n]] if n>=2 else f'%remote{n}'
                prefix.append(f'@%owns selp.b32 {dst[n]}, {left}, {right}, %wp{first};')
            lines[i]='\n'.join(prefix+['}'])
            continue
        for w,words in enumerate(own):
            if not words:
                prefix.append(f'@%wp{w} '+line.strip())
                continue
            local=[n for n,k in enumerate(words) if k is not None]
            assert local in ([0,1],[2,3])
            remote=[n for n in range(4) if n not in local]
            ptr=addr
            if remote[0]==2:
                prefix.append(f'@%wp{w} shfl.sync.bfly.b32 %laneaddr,{addr},16,31,0xffffffff;')
                ptr='%laneaddr'
            prefix.append(f'@%wp{w} ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{ {dst[remote[0]]}, {dst[remote[1]]} }}, [ {ptr} ];')
            prefix.extend(f'@%wp{w} mov.b32 {dst[n]}, {regs[words[n]]};' for n in local)
        lines[i]='\n'.join(prefix)
    if kind=='probe':lines[lo_dot]='ret;\n'+lines[lo_dot]
    lines[start]='\n'.join(decl)+'\n'+lines[start]
    return '\n'.join(lines)+'\n'

compile_original=bench.triton.compile
def build(src,options):
    kernel=compile_original(src,options=options)
    backend=make_backend(kernel.metadata.target)
    kernel.kernel=backend.make_cubin(modify(kernel.asm['ptx']),kernel.metadata._asdict(),backend.parse_options(options),80)
    return kernel

profile = '--profile' in sys.argv
if kind=='probe' or profile:
    # Fixed deployed IR and the normal nine-pointer ABI.
    ir=folder.parent/('coeff_transfer_v1/aligned_432.ttgir' if mode=='coeff' else 'warp_rows_v1/prefetch_kvwait_432.ttgir')
    options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False)
    kernel=build(str(ir),options)
    data=torch.load(bench.ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
    q,k,v=[data[key].cuda().contiguous() for key in ('q','k','v')]
    keys,mean,det=[data['summary'][key].cuda().contiguous() for key in ('keys','mean','detail')]
    ids=data['ids'][...,:43].cuda().contiguous()
    basis=data['basis'].cuda().bfloat16().contiguous()
    if profile:
        out=torch.empty_like(q)
        args=(q,keys,mean,det,basis,out,k,v,ids)
        baseline=compile_original(str(ir),options=options)
        for _ in range(50):
            for ker in (baseline,kernel):
                ker[(72,12,1)](*args)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for ker in (baseline,kernel):
            ker[(72,12,1)](*args)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        sys.exit(0)
    out=torch.full((128,32,4),-1,device='cuda',dtype=torch.int32)
    kernel[(1,1,1)](q,keys,mean,det,basis,out,k,v,ids)
    tags=out.cpu().tolist()
    mapping=[]
    for j in range(32):
        warps=[]
        for w in range(4):
            words=[]
            for n in range(4):
                values=[tags[w*32+l][j][n] for l in range(32)]
                assert all(0<=v<4096 and v//32%32==l for l,v in enumerate(values))
                assert len({v%32 for v in values})==len({v//1024 for v in values})==1
                words.append(values[0]%32 if values[0]//1024==w else None)
            warps.append(words if any(x is not None for x in words) else [])
        mapping.append(warps)
    (folder/f'{mode}_mapping.json').write_text(json.dumps(mapping))
else:
    bench.triton.compile=build
    bench._tail=coeff_tail if mode=='coeff' else page_tail
    bench.folder=folder
    bench.transform=lambda text,_:text
    bench.measure(mode+'_'+kind,432)
