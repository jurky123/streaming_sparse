"""Compare lowering changes for the full SM80 fused attention kernel."""
import argparse
import json
from pathlib import Path
import re
from statistics import median

import torch
import triton
from triton.compiler import compiler
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.weights import ROOT

compiler.prototype_pattern['ttgir']=r'^\s*tt\.func\s+(?:public\s+)?(@\w+)\((.*?)\)\s*attributes\s*\{[^}]*\}\s*\{'
compiler.arg_type_pattern['ttgir']=r'%\w+: (!tt\.ptr<[^>]+>)'
folder=Path(__file__).parent

def transform(text,mode):
    if mode=='control':return text
    if mode.startswith('double'):
        lines=text.splitlines()
        start=next(i for i,l in enumerate(lines) if '= scf.for' in l)
        end=next(i for i in range(start+1,len(lines)) if 'scf.yield' in lines[i])
        loop=lines[start+1:end]
        ptr=next(i for i,l in enumerate(loop) if '= arith.addi ' in l)
        commit=[i for i,l in enumerate(loop) if 'async_commit_group' in l]
        wait=next(i for i,l in enumerate(loop) if 'async_wait' in l)
        stage=loop[ptr:commit[-1]+1]
        current=loop[1:ptr]
        if mode=='double_split':
            idx=next(i for i,l in enumerate(current) if 'triton_gpu.local_load' in l and 'opIdx = 1' in l and '64x128xbf16' in l)
            current.insert(idx,'      %waitv = triton_gpu.async_wait  {num = 2 : i32}')
        k_wait=3 if mode=='double_split' else 2
        body=[loop[0]]+stage+[loop[wait].replace('num = 0',f'num = {k_wait}')]+current+loop[commit[-1]+1:wait]+loop[wait+1:]
        # Both current read slots and next write slots advance modulo two.
        prefix=lines[:start];prefix.append('    %twoslots = arith.constant 2 : i32')
        chunk='\n'.join(prefix+[lines[start]]+body+[lines[end]])
        a=chunk.index('triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x64xbf16')
        before,slots=chunk[:a],chunk[a:]
        slots=slots.replace('!tt.memdesc<1x128x64xbf16','!tt.memdesc<2x128x64xbf16').replace('!tt.memdesc<1x64x128xbf16','!tt.memdesc<2x64x128xbf16')
        slots=re.sub(r'(arith.cmpi slt, %\w+, )%c1_i32',r'\g<1>%twoslots',slots)
        dealloc=0
        for j in range(end+1,len(lines)):
            if 'triton_gpu.local_dealloc' in lines[j]:
                lines[j]=lines[j].replace('!tt.memdesc<1x','!tt.memdesc<2x')
                dealloc+=1
                if dealloc==2:break
        return before+slots+'\n'+'\n'.join(lines[end+1:])+'\n'
    if mode.startswith('keepq'):
        if mode.endswith('both'):text=transform(text,'prefetch_both')
        lines=text.splitlines()
        start=next(i for i,l in enumerate(lines) if '= scf.for' in l)
        qload=next(i for i in range(start+1,len(lines)) if 'triton_gpu.local_load' in lines[i] and 'opIdx = 0' in lines[i])
        result=lines.pop(qload).strip()
        name=result.split(' = ')[0]
        assert '64x128xbf16' in result
        # Give the moved value a fresh SSA name in its enclosing region.
        end=next(i for i in range(start+1,len(lines)) if 'scf.yield' in lines[i])
        for j in range(start+1,end):
            lines[j]=re.sub(re.escape(name)+r'\b','%fixedq',lines[j])
        lines.insert(start,'    '+result.replace(name+' =','%fixedq =',1))
        return '\n'.join(lines)+'\n'
    if mode.startswith('prefetch'):
        lines=text.splitlines()
        start=next(i for i,l in enumerate(lines) if '= scf.for' in l)
        end=next(i for i in range(start+1,len(lines)) if 'scf.yield' in lines[i])
        loop=lines[start+1:end]
        def pos(pattern):return next(i for i,l in enumerate(loop) if pattern in l)
        qk=pos('= tt.dot ')
        vload=next(i for i in range(qk+1,len(loop)) if '= triton_gpu.local_load ' in loop[i])
        am=pos('= arith.addf ')
        ptr=next(i for i in range(am,len(loop)) if '= arith.addi ' in loop[i])
        commits=[i for i,l in enumerate(loop) if 'async_commit_group' in l]
        assert len(commits)==2
        kcopy=loop[ptr:commits[0]+1]
        vcopy=loop[commits[0]+1:commits[1]+1]
        # After current K/V fragments are in registers, reuse the same shared tiles.
        # The backend's shared-memory barrier pass runs after this IR stage.
        mid=loop[qk:vload]+loop[vload+1:ptr]
        if mode in ('prefetch_ping','prefetch_pinglate','prefetch_pingend','prefetch_pingindex'):
            wait=next(i for i,l in enumerate(loop) if 'async_wait' in l)
            loop[wait]=loop[wait].replace('num = 0','num = 1')
            waitv='      %waitv = triton_gpu.async_wait  {num = 1 : i32}'
            dot2=next(i for i in range(qk+1,len(loop)) if '= tt.dot ' in loop[i])
            if mode in ('prefetch_pinglate','prefetch_pingend'):
                prob=loop[qk:vload]+loop[vload+1:dot2]
                rest=loop[dot2:ptr]
                new=loop[:qk]+kcopy+prob+[waitv,loop[vload]]
                new+=vcopy+rest if mode.endswith('late') else rest+vcopy
            else:
                new=loop[:qk]+kcopy+loop[qk:vload]+[waitv,loop[vload]]+vcopy+loop[vload+1:ptr]
                if mode.endswith('index'):
                    load=next(i for i,l in enumerate(new) if 'tt.load ' in l and '!tt.ptr<i32>' in l)
                    lookahead=new[load-2:load+1]
                    del new[load-2:load+1]
                    new[1:1]=lookahead
        elif mode in ('prefetch_latewait','prefetch_kvwait'):
            wait=next(i for i,l in enumerate(loop) if 'async_wait' in l)
            loop[wait]=loop[wait].replace('num = 0','num = 1')
            waitv='      %waitv = triton_gpu.async_wait  {num = 0 : i32}'
            if mode=='prefetch_kvwait':
                new=loop[:vload]+[waitv,loop[vload]]+kcopy+loop[vload+1:ptr]+vcopy
            else:
                new=loop[:vload]+[waitv]+loop[vload:commits[1]+1]
        elif mode=='prefetch_both':
            new=loop[:qk]+[loop[vload]]+kcopy+vcopy+mid
        elif mode=='prefetch_k':
            new=loop[:qk]+kcopy+loop[qk:ptr]+vcopy
        else:
            new=loop[:qk]+kcopy+loop[qk:vload+1]+vcopy+loop[vload+1:ptr]
        return '\n'.join(lines[:start+1]+new+loop[commits[1]+1:]+lines[end:])+'\n'
    layout={'row':'4, 1','half':'2, 2','column':'1, 4'}[mode.split('_')[0]]
    text,n=re.subn(r'(#mma1 = .*?warpsPerCTA = )\[1, 4\]',rf'\g<1>[{layout}]',text)
    assert n==1
    if mode.endswith('direct'):
        # Scope each allocation/load pair to its own region; MLIR SSA names can repeat.
        lines=text.splitlines()
        count=0
        for i,line in enumerate(lines):
            alloc=re.search(r'(%\w+) = triton_gpu.local_alloc (%\w+) : \((tensor<64x64xbf16, #mma1>)\)',line)
            if not alloc:continue
            dst,src,ty=alloc.groups()
            for j in range(i+1,min(i+30,len(lines))):
                old=f'triton_gpu.local_load {dst} : '
                if old not in lines[j]:continue
                assert 'opIdx = 0, parent = #mma1' in lines[j]
                lines[j]=lines[j].split(old)[0]+f'triton_gpu.convert_layout {src} : {ty} -> '+lines[j].split(' -> ',1)[1]
                lines[i]=''
                count+=1
                break
        assert count>=4,count
        text='\n'.join(lines)+'\n'
    return text

def measure(mode,p):
    data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
    q,k,v=[data[key].cuda().contiguous() for key in ('q','k','v')]
    keys,mean,detail=[data['summary'][key].cuda().contiguous() for key in ('keys','mean','detail')]
    ids=(data['ids'][...,:p//10].cuda()%p).contiguous()
    basis=data['basis'].cuda().bfloat16().contiguous()
    ref=torch.empty_like(q)
    out=torch.empty_like(q)
    def native(launcher=_tail, dest=ref):
        return launcher[(72,12)](q,keys,mean,detail,None,None,None,None,p,4608,12,128,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=dest,HEAD_PAGES=432,K=k,V=v,IDS=ids,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=p//10,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
    baseline=native()
    path=folder/f'{mode}_{p}.ttgir'
    if mode == 'release':
        from local_experiments.video_runtime_v1.page_prefetch import page_tail
        def run():return native(page_tail,out)
        kernel=run()
        assert '%waitv =' in kernel.asm['ttgir']
    else:
        path.write_text(transform(baseline.asm['ttgir'],mode))
        kernel=triton.compile(str(path),options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False))
        def run():kernel[(72,12,1)](q,keys,mean,detail,basis,out,k,v,ids)
    graphs=[]
    for fn in (native,run):
        fn();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        graphs.append(graph)
    for _ in range(300):
        for graph in graphs:graph.replay()
    torch.cuda.synchronize()
    times=[[],[]]
    for rep in range(20):
        for i in (rep%2,1-rep%2):
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(8):graphs[i].replay()
            b.record();b.synchronize()
            times[i].append(a.elapsed_time(b)/8)
    delta=ref.float()-out.float()
    result=dict(bitwise=torch.equal(ref.view(torch.uint8),out.view(torch.uint8)),ms=list(map(median,times)),speedup=median(a/b for a,b in zip(*times)),samples=times,changed=int((ref!=out).sum()),nmse=float(delta.square().sum()/ref.float().square().sum()),max_abs=float(delta.abs().max()),resources=[dict(regs=x.n_regs,spills=x.n_spills,shared=x.metadata.shared) for x in (baseline,kernel)])
    path.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='samples'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('pages',type=int,default=432,nargs='?')
    args=ap.parse_args();measure(args.mode,args.pages)
