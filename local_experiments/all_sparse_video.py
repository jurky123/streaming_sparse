"""Early RGB acceptance for the initialized and trained full sparse architecture."""
import gc,json,os,signal,torch
from pathlib import Path
from .broad_common import Engine
from .all_sparse_runtime import AllSparseRuntime
from .rollout_suite import generate_chunk,video
from .high_sparse_experiment import prefix_decode,prefix_quality,OLD
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/sparse_expansion_20261031/all_sparse_video');TRAIN=ROOT.parent/'all_sparse_train'
@torch.no_grad()
def main():
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity=dict(checkpoints={n:digest(TRAIN/(n+'.pt')) for n in ['initial','epoch0','epoch1']},code={n:digest(Path('local_experiments')/n) for n in ['all_sparse_video.py','all_sparse_runtime.py','all_sparse_model.py','all_sparse_attention.py']},window=[21,101],seed=1001)
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        for case in ['basketball_court','butterfly_garden']:
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda();saved=torch.load(OLD/(case+'_s1001')/'dense_host.latents.pt',weights_only=True).cuda();ref=prefix_decode(vae,list(saved[:,:27].split(3,1)));del saved
            for label in ['initial','epoch0','epoch1']:
                dest=ROOT/case/(label+'.json')
                if dest.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted'));return
                disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache();rt=AllSparseRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt');rt.model.load_state_dict(torch.load(TRAIN/(label+'.pt'),map_location='cpu',weights_only=False)['model'])
                try:
                    atomic(ROOT/'status.json',dict(status='running',case=case,checkpoint=label,pid=os.getpid()))
                    latents=[generate_chunk(engine,emb,1001,c) for c in range(9)];frames=prefix_decode(vae,latents);q=prefix_quality(frames,ref,metric)
                    atomic(ROOT/case/(label+'.latents.pt'),torch.cat(latents,1).cpu(),True);video(ROOT/case/(label+'.mp4'),frames)
                    atomic(dest,dict(case=case,checkpoint=label,quality=q,runtime=rt.finish()));print(json.dumps(dict(case=case,checkpoint=label,quality={k:v for k,v in q.items() if k!='per_frame'})),flush=True)
                    del latents,frames
                finally:rt.close();del rt
        atomic(ROOT/'status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
