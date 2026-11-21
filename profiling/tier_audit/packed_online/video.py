import ctypes,json,sys,time
from pathlib import Path
import torch
repo=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from queue_policy import attach
from local_experiments.broad_common import Engine
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.video_runtime_v1 import legacy_components
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import pack_range,retry

@torch.no_grad()
def main():
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('intersection',local_assets=repo)
    configure(cfg)
    engine=Engine()
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    gen=attach(rt,engine,10)
    emb=torch.load(repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt',weights_only=True).cuda()
    lib=ctypes.CDLL('/tmp/tier_variable.so')
    kernel=lib.run
    kernel.argtypes=library().run.argtypes
    old=legacy_components.attention
    caches={}
    seen={}
    def candidate(q,k,v,s,basis,ids,workspace,**kwargs):
        p=k.shape[1]//64
        layer=engine.current['layer']
        if layer not in caches:
            caches[layer]=[q.new_empty(432,12,8192) for _ in range(2)]
        pk,pv=caches[layer]
        chunk=engine.current['chunk']
        if seen.get(layer)!=chunk:
            pack_range[(p-72,12,16)](k,v,pk,pv,0)
            seen[layer]=chunk
        packed=workspace.tensor('packed_var',(12,(p+15)//16,10624),q.device,q.dtype)
        hp=s.mean.stride(0)//128
        stream=torch.cuda.current_stream().cuda_stream
        assert not library().prepare_fixed15(*(x.data_ptr() for x in (k,v,pk,pv,s.keys,s.mean,s.detail,packed)),p,hp,stream)
        result=workspace.tensor('var_out',q.shape,q.device,q.dtype)
        flags=workspace.tensor('var_flags',(12,72),q.device,torch.int32)
        basis=basis.bfloat16().contiguous()
        assert not kernel(*(x.data_ptr() for x in (q,pk,pv,ids,result,packed,basis,flags)),p,ids.shape[-1],stream)
        retry[(72,12)](flags,q,k,v,ids,s.keys,s.mean,s.detail,basis,result,p,hp,ids.shape[-1],num_warps=4,num_stages=2,enable_fp_fusion=False)
        return result
    rows=[]
    for i,fn in enumerate([old,candidate,old,candidate,candidate,old]):
        legacy_components.attention=fn
        seen.clear()
        rt.reset()
        torch.cuda.synchronize()
        start=time.perf_counter()
        z=gen(emb,701,7)
        torch.cuda.synchronize()
        sec=time.perf_counter()-start
        if i==0: ref=z.clone()
        diff=z.float()-ref.float()
        row=[i,sec,diff.abs().max().item(),(diff!=0).float().mean().item(),(diff.square().sum()/ref.float().square().sum()).item()]
        print(row,flush=True)
        rows.append(row)
        (out/'video.json').write_text(json.dumps(rows))
        if i<2:torch.save(z.cpu(),'/tmp/variable_latent_'+str(i)+'.pt')
main()
