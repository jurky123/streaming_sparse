from pathlib import Path
p=Path(__file__).parent
for mode in ('rawboth','paired_k','rawkey'):
 s=(p/f'{mode}.cu').read_text()
 pos=s.index('__global__ void exact')
 prefix,s=s[:pos],s[pos:]
 i=s.index('    A q[8];');j=s.index('    Four out[16]',i)
 s=s[:i]+'''    bf* query_tile=reinterpret_cast<bf*>(pool+4096);
    stage(query_tile,Q,t,h);
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
    __syncthreads();
'''+s[j:]
 i=s.index('        for(int k=0;k<8;k++)')
 j=s.index('        #pragma unroll\n        for(int r=0;r<2;r++)',i)
 body=s[i:j].replace('for(int k=0;k<8;k++)','''for(int k=0;k<8;k++){
            unsigned qa=__cvta_generic_to_shared(query_tile+tile(w*16+l%16,16*k+l/16*8));
            A frag;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                :"=r"(frag.x[0]),"=r"(frag.x[1]),"=r"(frag.x[2]),"=r"(frag.x[3]):"r"(qa));''').replace('q[k]','frag')
 s=s[:i]+body+'        }\n'+s[j:]
 s=s.replace('constexpr int bytes=2656*16+(8192+64)*4','constexpr int bytes=81920')
 (p/f'qshare_{mode}.cu').write_text(prefix+s)
 t=(p/f'{mode}.py').read_text().replace('fixed15_'+mode+'.so','fixed15_qshare_'+mode+'.so').replace("f'"+mode+"_{p}.json'","f'qshare_"+mode+"_{p}.json'")
 (p/f'qshare_{mode}.py').write_text(t)
