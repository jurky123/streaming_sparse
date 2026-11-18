"""Root power iteration with resident-chain tiles versus the accepted batched GEMVs."""
import ctypes,json,statistics
import torch
from .profile_route import capture
from .weights import ROOT
from ..gap_group_norm import normalize

def main():
    lib=ctypes.CDLL('/tmp/jzk_root_direction.so')
    lib.root_direction.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.root_direction.restype=None
    rows=[]
    torch.manual_seed(701);torch.backends.cuda.matmul.allow_tf32=False
    for n in (1536,4608):
        g=12*4608//n
        x=torch.randn(g,n,32,device='cuda');x-=x.mean(1,keepdim=True)
        axis=x.square().mean(1).argmax(-1,keepdim=True)
        def old():
            projection=x.gather(2,axis[:,None].expand(-1,n,-1))
            for _ in range(4):
                v=normalize(torch.bmm(x.transpose(1,2),projection).squeeze(-1))
                projection=torch.bmm(x,v.unsqueeze(-1))
            return projection.squeeze(-1)
        def new():
            scratch=x.new_empty(g,32,32);v=x.new_empty(g,32);z=x.new_empty(g,n)
            lib.root_direction(x.data_ptr(),axis.data_ptr(),scratch.data_ptr(),v.data_ptr(),z.data_ptr(),g,n,torch.cuda.current_stream().cuda_stream)
            return z
        variants=[capture(f) for f in (old,new)]
        times=[[],[]]
        for rep in range(26):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
                for _ in range(20):variants[i][0].replay()
                b.record();b.synchronize()
                if rep>=2:times[i].append(a.elapsed_time(b)/20)
        a,b=[pair[1] for pair in variants]
        ms=list(map(statistics.median,times));row=dict(n=n,ms=ms,speedup=ms[0]/ms[1],mismatch=int((a!=b).sum()),max_abs=float((a-b).abs().max()))
        print(row,flush=True);rows.append(dict(**row,samples_ms=times))
    (ROOT/'artifacts/video_runtime_v1/group_gemv_order_v1/root_module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
