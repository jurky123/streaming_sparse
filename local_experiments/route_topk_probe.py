"""Isolated optional sorting candidate; not deployed without tie-policy validation."""
import torch,triton as tr,triton.language as tl,json,statistics
from .learned_cache.io import atomic
from pathlib import Path
@tr.jit
def topk(MASS,IDS,MASK,P:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,B);m=tl.load(MASS+row*P+p,p<P,-float('inf'));m=tl.where(p==row%72+P-72,float('inf'),m)
    key=(m.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)+(B-p).to(tl.uint64)
    key=tl.where(p<P,key,0)
    ordered=tl.sort(key,descending=True);idx=(B-(ordered & 0xffffffff).to(tl.int32))
    tl.store(IDS+row*K+p,idx,p<K)
    threshold=tl.min(tl.where(p<K,ordered,0xffffffffffffffff),0)
    mask=key>=threshold
    tl.store(MASK+row*P+p,mask,p<P)
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(31);results=[]
    for p in [216,288,360,410]:
        m=torch.randn(12,72,p,device='cuda').softmax(-1);k=int(p*.1+1e-6);ids=torch.empty(12,72,k,device='cuda',dtype=torch.int64);mask=torch.empty_like(m,dtype=torch.bool)
        own=(torch.arange(72,device='cuda')+p-72)[None,:,None].expand(12,-1,-1)
        def native():
            z=m.clone();z.scatter_(-1,own,float('inf'));ii=z.topk(k,-1).indices
            return ii,torch.zeros_like(m,dtype=torch.bool).scatter_(-1,ii,True)
        def fast():topk[(864,)](m,ids,mask,p,k,tr.next_power_of_2(p),num_warps=4);return ids,mask
        ref=native()[1];fast();difference=int((mask!=ref).any(-1).sum())
        row=dict(p=p,different_tiles=difference)
        for name,fn in [('native',native),('sort',fast)]:
            for _ in range(4):fn()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):o=fn()
            times=[]
            for _ in range(5):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(100):g.replay()
                b.record();b.synchronize();times.append(a.elapsed_time(b)/100)
            row[name+'_ms']=statistics.median(times)
        results.append(row)
    atomic(Path('artifacts/route_kernel_20261031/topk_probe.json'),results);print(json.dumps(results))
if __name__=='__main__':main()
