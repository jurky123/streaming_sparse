from pathlib import Path
p=Path(__file__).parent
s=(p/'shared_overlap.cu').read_text()
src=(p.parent/'fixed15_baseline_v1/warp_exact.cu').read_text()
s=s.replace('float* O,float* L,float* M,int count)', 'float* O,float* L,float* M,int count,const bf* rawk,const bf* rawv,int hist)')
i=s.index('            for(int m=threadIdx.x');j=s.index('\n        }\n        asm volatile',i)
s=s[:i]+'''            if(page>=hist){
                stage(reinterpret_cast<bf*>(slots[slot][0]),rawk,page,h);
                stage(reinterpret_cast<bf*>(slots[slot][1]),rawv,page,h);
            }else{
'''+s[i:j]+'\n            }'+s[j:]
i=s.index('                uint4 u=');j=s.index('                mma(score',i)
bstart=src.index('                unsigned addr=');bend=src.index('                mma(score',bstart)
raw=src[bstart:bend].replace('kv[0]+tile','reinterpret_cast<bf*>(kv[0])+tile').replace('                B b[2];\n','')
s=s[:i]+'''                B b[2];
                if(list[pg]>=hist){
'''+raw+'''                }else{
                    uint4 u=kv[0][(k*4+n/2)*32+l];
                    b[0]=B{{u.x,u.y}};b[1]=B{{u.z,u.w}};
                }
'''+s[j:]
i=s.index('                uint4 u=reinterpret_cast');j=s.index('                mma(tmp',i)
bstart=src.index('                unsigned addr=',bend);bend=src.index('                mma(tmp',bstart)
raw=src[bstart:bend].replace('kv[1]+tile','reinterpret_cast<bf*>(kv[1])+tile')
s=s[:i]+'''                if(list[pg]>=hist){
'''+raw+'''                }else{
                    uint4 u=kv[1][(k*8+n/2)*32+l];
                    b[k][0]=B{{u.x,u.y}};b[k][1]=B{{u.z,u.w}};
                }
'''+s[j:]
s=s.replace('int count,cudaStream_t s)','int count,cudaStream_t s,const bf* rawk,const bf* rawv,int hist)').replace('out,lse,mass,count);','out,lse,mass,count,rawk,rawv,hist);')
(p/'hybrid.cu').write_text(s)
s=(p/'full.py').read_text().replace("os.getenv('EXACT_VARIANT','shared_overlap')","'hybrid'").replace('[C.c_int,C.c_void_p]','[C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int]').replace('count,torch.cuda.current_stream().cuda_stream)','count,torch.cuda.current_stream().cuda_stream,k.data_ptr(),v.data_ptr(),p-72)')
(p/'hybrid.py').write_text(s)
