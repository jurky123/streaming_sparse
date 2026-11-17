"""Independently rebuild summaries/coefficients and verify all surviving ring slots."""
import argparse,json,math
from pathlib import Path
import torch
from .weights import load,ROOT
from .runtime import VideoRuntime
from ..broad_common import Engine
from ..review_v1.ops import Summary,SummaryOps
from ..learned_cache.io import atomic

@torch.no_grad()
def run(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=torch.load(args.source,map_location='cpu',weights_only=True);chunks=source.shape[1]//3
    model,reader,phi,B,identity=load();engine=Engine();engine.wrapper.requires_grad_(False)
    rt=VideoRuntime(engine,model,reader,phi,B,full_pset=True,tile_batch=6)
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda();snapshots=[];names=('keys','mean','detail','stats');layers=(1,15,29)
    try:
        for chunk in range(chunks):
            engine.forward(source[:,chunk*3:chunk*3+3].cuda(),emb,chunk,4)
            if chunk<chunks-math.ceil(338/72):continue
            slots=(torch.arange(72,device='cuda')+chunk*72)%338
            for l in layers:
                store=rt.stores[l];p=store.n
                k=store.cache['k'][0,:p*64].transpose(0,1).reshape(12,p,64,128).index_select(1,slots)
                v=store.cache['v'][0,:p*64].transpose(0,1).reshape(12,p,64,128).index_select(1,slots)
                expected=SummaryOps.build(model[l],k,v);encoding=rt.encode(l,expected)
                for name in names:assert torch.equal(getattr(expected,name),getattr(store.summary,name).index_select(1,slots)),(chunk,l,name)
                for name,value in encoding.items():assert torch.equal(value,rt.encodings[l][name].index_select(1,slots)),(chunk,l,name)
                snapshots.append((chunk,l,slots.cpu(),{name:getattr(expected,name).cpu() for name in names},{name:value.cpu() for name,value in encoding.items()}))
        covered={l:torch.zeros(338,dtype=torch.bool) for l in layers}
        for chunk,l,slots,summary,en in snapshots:
            store=rt.stores[l];origin=store.token_ids[0,:,0].cpu()//4608;alive=origin[slots]==chunk;kept=slots[alive];covered[l][kept]=True
            for name in names:assert torch.equal(getattr(store.summary,name).cpu().index_select(1,kept),summary[name][:,alive]),(l,chunk,name)
            for name in en:assert torch.equal(rt.encodings[l][name].cpu().index_select(1,kept),en[name][:,alive]),(l,chunk,name)
        assert all(x.all() for x in covered.values())
        report=dict(passed=True,chunks=chunks,layers=list(layers),head_pages=338*12*len(layers),all_338_surviving_slots_verified=True,raw_KV_summary_rebuild_exact=True,reader_phi_projection_rebuild_exact=True,ring_write=rt.stores[1].write)
        atomic(Path(args.output)/'cache_check.json',report);print(json.dumps(report))
    finally:rt.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='artifacts/video_runtime_v1/long_pilot/videos/city_walk_s701/chunks41/c_pset/latents.pt');p.add_argument('--output',default='artifacts/video_runtime_v1/cache_check');run(p.parse_args())
