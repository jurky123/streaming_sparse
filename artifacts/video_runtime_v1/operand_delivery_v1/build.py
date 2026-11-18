"""Test packed operand delivery and Tensor Core denominator on the accepted CUDA base."""
import subprocess
import sys
from pathlib import Path

directory = Path(__file__).parent
name = sys.argv[1]
src = (directory/'control.cu').read_text()
if name.startswith('direct_'):
    # Cooperating warps consume identical fragments. Compare global loads via L1
    # against the baseline CTA-wide copy and subsequent shared load.
    for op, idx, ptr in [('k', 0, 'fk'), ('v', 1, 'fv')]:
        if op not in name.split('_')[1]:
            continue
        src = '\n'.join(line for line in src.split('\n') if f'"l"({ptr}+m)' not in line)
        src = src.replace(f'reinterpret_cast<const uint4*>(kv[{idx}])',
                          f'(reinterpret_cast<const uint4*>({op.upper()})+current_offset)')
    src = src.replace('Slot& kv=', 'int current_offset=__shfl_sync(~0u,step<32?first_offset:second_offset,step%32);\n        Slot& kv=')
    if name == 'direct_kv':
        a=src.index('        if(step>=0)')
        b=src.index('        // The first page', a)
        src=src[:a]+src[b:]
        src=src.replace('for(int step=-1;step<count;step++)','__syncthreads();\n    for(int step=0;step<count;step++)')
elif name in ('late_mask','peel_seed'):
    a=src.index('        // The first page barrier')
    b=src.index('        Slot& kv=',a)
    mask=src[src.index('        if(step==0',a):b].replace('step==0 && ','')
    src=src[:a]+src[b:]
    src=src.replace('    float* E=',mask+'    float* E=')
    if name == 'peel_seed':
        a=src.index('    for(int step=-1;step<count;step++){')
        b=src.index('        Slot& kv=',a)
        loop=src[a:b]
        start=loop.index('            int slot=')
        end=loop.index('        }\n        asm volatile("cp.async.commit_group')
        seed=loop[start:end].replace('(step+1)&1','0').replace('+offset','+__shfl_sync(~0u,first_offset,0)')
        loop=loop.replace('step=-1','step=0').replace('if(step>=0) {','{').replace('        if(step<0)continue;','')
        src=src[:a]+'    {\n'+seed+'        asm volatile("cp.async.commit_group;":::"memory");\n    }\n'+loop+src[b:]
elif name in ('fused_center','anchor_center','floor_center'):
    a=src.index('        #pragma unroll\n        for(int r=0;r<2;r++) {',src.index('Slot& kv='))
    b=src.index('        A high[4]',a)
    anchor=src[a:b].replace('score[n].x[j]=__fmul_rn','float scaled=__fmul_rn').replace('fmaxf(m,score[n].x[j])','fmaxf(m,scaled)')
    src=src[:a]+'        if(pg==0){\n'+anchor.replace('if(pg==0) {','{')+'        }\n'+src[b:]
    for j in ('r*2','r*2+1'):
        src=src.replace(f'exp2ptx(score[n].x[{j}]-anchor[r])',
                        f'exp2ptx(__fmaf_rn(score[n].x[{j}],float(1.4426950408889634 / 11.313708498984761),-anchor[r]))')
    if name == 'floor_center':
        src=src.replace('float scaled=__fmul_rn','float scaled=__fmul_rd')
    if name == 'anchor_center':
        # One maximum per row of page zero must contribute exactly exp2(0).
        for j in ('r*2','r*2+1'):
            fma=f'__fmaf_rn(score[n].x[{j}],float(1.4426950408889634 / 11.313708498984761),-anchor[r])'
            rounded=f'__fmul_rn(score[n].x[{j}],float(1.4426950408889634 / 11.313708498984761))-anchor[r]'
            src=src.replace('exp2ptx('+fma+')','exp2ptx(pg==0?('+rounded+'):'+fma+')')
elif name == 'reuse_round':
    src=src.replace('low[n/2].x[2*(n%2)+r]=pairbf(x-__bfloat162float(__float2bfloat16(x)),y-__bfloat162float(__float2bfloat16(y)));',
                    """unsigned rounded=high[n/2].x[2*(n%2)+r];
                low[n/2].x[2*(n%2)+r]=pairbf(x-__uint_as_float(rounded<<16),y-__uint_as_float(rounded&0xffff0000));""")
elif name == 'tensor_mass':
    src = src.replace('cols[8]={}', 'mass={}')
    src = src.replace('cols[n].x[r*2]+=x; cols[n].x[r*2+1]+=y;', '')
    src = src.replace('        // Accumulate both', '''        constexpr B ones={{0x3f803f80u,0x3f803f80u}};
        #pragma unroll
        for(int k=0;k<4;k++)mma(mass,high[k],ones);
        #pragma unroll
        for(int k=0;k<4;k++)mma(mass,low[k],ones);
        // Accumulate both''')
    a=src.index('        // Emulate the four')
    b=src.index('        int idx=',a)
    src=src[:a]+'        float s=mass.x[2*r];\n'+src[b:]
else:
    assert name == 'control'

source = Path('/tmp/jzk_delivery_'+name+'.cu')
source.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-arch=sm_80','-shared','-Xcompiler=-fPIC',
                '-Xptxas=-v',str(source),'-o',str(source.with_suffix('.so'))],check=True)
