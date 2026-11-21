import torch,triton
from candidate import _tail
torch.manual_seed(701)
def rand(*shape):
    return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
q=rand(1,4608,12,128)
for p,count in ((216,33),(432,43),(432,64),(432,105)):
    k,v=[rand(1,p*64,12,128) for _ in range(2)]
    keys,mean,detail,basis=rand(12,p,4,128),rand(12,p,128),rand(12,p,3,8),rand(12,128,8)
    ids=torch.rand(12,72,p,device='cuda').argsort(-1)[...,:count].int().contiguous()
    out=torch.empty_like(q)
    for tile in (0,16,32,64,128):
        warps,stages=4,2
        pk=k if not tile else k.reshape(p,64,12,128//tile,tile).permute(0,2,3,1,4).contiguous()
        pv=v if not tile else v.reshape(p,64,12,128).transpose(1,2).contiguous()
        def run():
            return _tail[(72,12)](q,keys,mean,detail,None,None,None,None,p,4608,12,128,HEAD_PAGES=p,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,K=pk,V=pv,KTILE=tile,IDS=ids,EXACT_COUNT=count,QK_SCALE=128**-.5,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
        compiled=run()
        if not tile: ref=out.clone()
        print(p,count,tile,compiled.n_regs,compiled.n_spills,torch.equal(ref,out),triton.testing.do_bench_cudagraph(run,rep=60),flush=True)
