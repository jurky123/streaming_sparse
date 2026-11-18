import ctypes,json,statistics,os
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.profile_route import capture
variant=os.environ.get('CTX_VARIANT','')
lib=ctypes.CDLL('/tmp/jzk_compact_context'+variant+'.so')
lib.compact.argtypes=[ctypes.c_void_p]*5
lib.compact.restype=None
def prepare(k,v):
    if 'pack' in variant:
        kt=k[0,:64].transpose(0,1).reshape(12,8,8,8,2,4,2).permute(0,3,1,2,5,4,6).contiguous()
        vt=v[0,:64].permute(1,2,0).reshape(12,16,8,4,2,4,2).permute(0,1,3,2,5,4,6).reshape(12,-1)
        return kt,torch.cat((vt,v[0,63]),dim=1)
    return k[0,:64].transpose(0,1).contiguous(),v[0,:64].permute(1,2,0).contiguous()
def apply(q,k,v):
    out=torch.empty_like(q)
    lib.compact(*(x.data_ptr() for x in (q,k,v,out)),torch.cuda.current_stream().cuda_stream)
    return out
def main():
    rows=[]
    for q,k,v in torch.load('/tmp/jzk_cross_three.pt',weights_only=True):
        q,k,v=[x.cuda() for x in (q,k,v)]
        pk,pv=prepare(k,v)
        cases=[capture(lambda:torch.nn.functional.scaled_dot_product_attention(*(x.transpose(1,2) for x in (q,k,v))).transpose(1,2)),
               capture(lambda:apply(q,pk,pv))]
        times=[[],[]]
        for rep in range(24):
            for i in [rep%2,1-rep%2]:
                a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                for _ in range(16):cases[i][0].replay()
                b.record();b.synchronize()
                if rep>=4:times[i].append(a.elapsed_time(b)/16)
        ref,new=[out.float() for _,out in cases];d=ref-new
        row=dict(ms=list(map(statistics.median,times)),nmse=float(d.square().sum()/ref.square().sum()),max_abs=float(d.abs().max()),different=int(d.count_nonzero()),samples=times)
        print({k:v for k,v in row.items() if k!='samples'},flush=True);rows.append(row)
    Path(__file__).with_name('compact'+variant+'_module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
