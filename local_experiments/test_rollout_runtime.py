import json,torch
from .broad_common import Engine,noise
from .rollout_runtime import Runtime,ROOT
from .learned_cache.io import atomic
from .rollout_kernels import aggregate

@torch.no_grad()
def chunk(engine,emb,seed,index):
    x=noise(seed,index,0)
    for step in range(4):
        clean=engine.forward(x,emb,index,step)
        if step<3:x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),noise(seed,index,step+1).flatten(0,1),
            torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
    engine.forward(clean,emb,index,4)
    return clean

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Engine();emb=torch.load('artifacts/broad_coverage_20261030/embeddings/horse_field.pt',weights_only=True).cuda()
    base=Runtime(engine,'dense');first=chunk(engine,emb,101,0).clone()
    saved=torch.load('artifacts/broad_coverage_20261030/trajectories/val_horse_field_s101/chunk00.pt',weights_only=True)['clean'].cuda()
    assert torch.equal(first,saved),'native baseline drift'
    base.close();del base
    rt=Runtime(engine,'mass_var',audit=True)
    outputs=[]
    for i in range(7):
        outputs.append(chunk(engine,emb,101,i).clone());print('continuous chunk',i,flush=True)
    assert torch.equal(outputs[0],first)
    assert rt.commits[1:]==[7]*29 and rt.sparse_calls==5*4*29
    checks=rt.finish()
    for l in [1,14,29]:
        k=engine.cache[l]['k'][0,:338*64].transpose(0,1).contiguous().float()
        torch.testing.assert_close(rt.pk[l],aggregate(k,rt.weights[l]),atol=3e-5,rtol=3e-4)
    rt.reset()
    for i in range(6):engine.forward(outputs[i],emb,i,4)
    resumed=chunk(engine,emb,101,6)
    assert torch.equal(resumed,outputs[6]),(resumed-outputs[6]).abs().max()
    checks.update(passed=True,native_first_chunk_exact=True,resume_clean_prefix_exact=True,
        ring_summary_matches_physical_kv=True,once_per_clean_commit=True)
    atomic(ROOT/'runtime_checks.json',checks);rt.close()
    print('All ring, physical budget, kernel, fused score, commit and resume checks passed',flush=True)
if __name__=='__main__':run()
