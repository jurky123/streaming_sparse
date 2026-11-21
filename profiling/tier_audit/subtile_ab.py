import torch,triton
from subtile_tail import _tail
torch.manual_seed(701)
def rand(*shape):
    return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
q=rand(1,4608,12,128)
for p,count in ((216,33),(432,43),(432,64),(432,105)):
    k,v=[rand(1,p*64,12,128) for _ in range(2)]
    keys,mean,detail,basis=rand(12,p,4,128),rand(12,p,128),rand(12,p,3,8),rand(12,128,8)
    ids=torch.rand(12,72,p,device='cuda').argsort(-1)[...,:count].int().contiguous()
    out=torch.empty_like(q)
    for bq,warps in ((64,4),(32,4),(32,2)):
        stages=2
        def run():
            return _tail[(4608//bq,12)](q,keys,mean,detail,None,None,None,None,p,4608,12,128,HEAD_PAGES=p,BQ=bq,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=k,V=v,IDS=ids,EXACT_COUNT=count,QK_SCALE=128**-.5,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
        compiled=run()
        if bq==64: ref=out.clone()
        print(p,count,bq,warps,compiled.n_regs,compiled.n_spills,torch.equal(ref,out),triton.testing.do_bench_cudagraph(run,rep=60),flush=True)
