from pathlib import Path
p=Path(__file__).parent
s=(p/'paired_k.cu').read_text()
s=s.replace('int page=pg+n*2+z/2;', 'int page=pg+8*(n/4)+2*z+(n%4)/2;')
i=s.index('                float a[4],b[4];');j=s.index('                float sums[4],diffs[4];',i)
s=s[:i]+'''                float a[4],b[4];
                #pragma unroll
                for(int j=0;j<4;j++){
                    a[j]=score[4*chunk+j].v[2*r];
                    b[j]=score[4*chunk+j].v[2*r+1];
                }
'''+s[j:]
(p/'local_coeff.cu').write_text(s)
# Restore the original lane-local sum and its warp tree independently.
i=s.index('            partial+=__shfl_xor_sync')
s=s.replace('partial+=a;partial+=b;','')
i=s.index('            partial+=__shfl_xor_sync')
s=s[:i]+'''            #pragma unroll
            for(int chunk=0;chunk<2;chunk++){
                float a[4],b[4];
                #pragma unroll
                for(int j=0;j<4;j++){a[j]=score[4*chunk+j].v[2*r];b[j]=score[4*chunk+j].v[2*r+1];}
                #pragma unroll
                for(int bit=1;bit<=2;bit*=2){
                    #pragma unroll
                    for(int j=0;j<4;j++)if(!(j&bit)){
                        float x=__shfl_xor_sync(~0u,(z&bit)?a[j]:a[j+bit],bit);
                        float y=__shfl_xor_sync(~0u,(z&bit)?b[j]:b[j+bit],bit);
                        if(z&bit){a[j]=x;b[j]=y;}else{a[j+bit]=x;b[j+bit]=y;}
                    }
                }
                #pragma unroll
                for(int j=0;j<4;j++){partial+=a[j];partial+=b[j];}
            }
'''+s[i:]
(p/'local_exactden.cu').write_text(s)
s=(p/'summary_fragments.py').read_text()
s=s.replace('rep=t*64+token','rep=t*64+(token//32*8+2*((token%8)//2)+token%32//16)*4+2*((token//8)%2)+token%2')
(p/'local_summary.py').write_text(s)
for v in ['local_coeff','local_exactden']:
 s=(p/'paired_k.py').read_text().replace('from summary_fragments','from local_summary').replace("assert torch.equal(summaries,summaries_ref)","assert torch.equal(summaries[:,:,8192:],summaries_ref[:,:,8192:])").replace('paired_k',v)
 (p/f'{v}.py').write_text(s)
