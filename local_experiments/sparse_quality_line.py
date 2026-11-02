"""Prospective early-RGB held-out coverage and sparse-family leave-one-out audit."""
import argparse,gc,json,os,signal,torch
from pathlib import Path
from .broad_common import Engine,cases,ROOT as DATA
from .rollout_runtime import Runtime
from .all_sparse_runtime import AllSparseRuntime
from .sparse_expansion_fast_runtime import ExpansionRuntime
from .rollout_suite import generate_chunk,video
from .high_sparse_experiment import prefix_decode,prefix_quality,OLD
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/sparse_quality_line_20261031');PREV=Path('artifacts/sparse_expansion_20261031')
BASE=Path('artifacts/full_route_grad_20261031/epoch1.pt')
PATHS={'initial':PREV/'all_sparse_train/initial.pt','horizon1':PREV/'horizon1_train/epoch0.pt','horizon2':PREV/'horizon2_train/epoch0.pt'}
FAMILIES={'group':['bank','group_v','group_gate'],'pool':['pool_k','pool_v','pool_feature'],'route':['route'],'value':['basis','center'],'mass':['comp_logscale']}
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['broad','ablation'],required=True);ap.add_argument('--seed',type=int,default=202);args=ap.parse_args()
    root=ROOT/(f'broad_s{args.seed}' if args.mode=='broad' else 'ablation');lock=RunLock(root);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        configs=['initial','horizon1','horizon2','old16'] if args.mode=='broad' else ['initial','horizon2']+['restore_'+k for k in FAMILIES]+['half_update']
        samples=[c for c in cases('test') if c['seed']==args.seed] if args.mode=='broad' else [dict(id=n+'_s1001',name=n,seed=1001) for n in ['basketball_court','butterfly_garden']]
        assert samples
        identity=dict(mode=args.mode,samples=samples,configs=configs,families=FAMILIES,checkpoints={k:digest(p) for k,p in PATHS.items()},base=digest(BASE),code={n:digest(Path('local_experiments')/n) for n in ['sparse_quality_line.py','all_sparse_model.py','all_sparse_runtime.py','all_sparse_attention.py','sparse_expansion_fast_runtime.py']},window=[21,101],selection='No fitting or threshold tuning on these RGB test results; report paired LPIPS and SSIM, all per-case regressions; no full-speed quality pass unless both LPIPS and SSIM gates hold')
        identity=json.loads(json.dumps(identity))
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        states={k:torch.load(p,map_location='cpu',weights_only=False)['model'] for k,p in PATHS.items()}
        def state(label):
            if label in states:return states[label]
            chosen={k:v.clone() for k,v in states['horizon2'].items()}
            if label=='half_update':return {k:(states['initial'][k]+v)*.5 for k,v in chosen.items()}
            names=FAMILIES[label.removeprefix('restore_')]
            for k in chosen:
                if k.rsplit('.',1)[-1] in names:chosen[k]=states['initial'][k]
            return chosen
        for case in samples:
            folder=root/case['id'];folder.mkdir(parents=True,exist_ok=True)
            ep=DATA/'embeddings'/f'{case["name"]}.pt' if args.mode=='broad' else Path('artifacts/mass_variance_20261030/embeddings')/f'{case["name"]}.pt'
            emb=torch.load(ep,weights_only=True).cuda();densepath=folder/'dense.latents.pt'
            if args.mode=='ablation':
                refz=torch.load(OLD/(case['name']+'_s1001')/'dense_host.latents.pt',weights_only=True)[:,:27].cuda();ref=prefix_decode(vae,list(refz.split(3,1)));del refz
            elif densepath.exists():
                refz=torch.load(densepath,weights_only=True).cuda();ref=prefix_decode(vae,list(refz.split(3,1)));del refz
            else:
                if stop[0]:atomic(root/'status.json',dict(status='interrupted'));return
                disk_guard(root,50,1);atomic(root/'status.json',dict(status='running',case=case['id'],config='dense',pid=os.getpid()))
                rt=Runtime(engine,'dense')
                try:
                    zs=[generate_chunk(engine,emb,case['seed'],c) for c in range(9)];ref=prefix_decode(vae,zs);video(folder/'dense.mp4',ref);atomic(densepath,torch.cat(zs,1).cpu(),True);atomic(folder/'dense.json',dict(runtime=rt.finish(),reference='chronological full dense attention, same backbone, seed and original KV limit'));del zs
                finally:rt.close();del rt
            for label in configs:
                dest=folder/(label+'.json')
                if dest.exists():continue
                if stop[0]:atomic(root/'status.json',dict(status='interrupted'));return
                disk_guard(root,50,1);gc.collect();torch.cuda.empty_cache()
                rt=ExpansionRuntime(engine,BASE,16,1) if label=='old16' else AllSparseRuntime(engine,BASE)
                if label!='old16':rt.model.load_state_dict(state(label))
                try:
                    atomic(root/'status.json',dict(status='running',case=case['id'],config=label,pid=os.getpid()))
                    zs=[generate_chunk(engine,emb,case['seed'],c) for c in range(9)];frames=prefix_decode(vae,zs);q=prefix_quality(frames,ref,metric);value=torch.cat(zs,1).cpu();assert torch.isfinite(value).all()
                    video(folder/(label+'.mp4'),frames);atomic(folder/(label+'.latents.pt'),value,True);atomic(dest,dict(case=case,config=label,quality=q,runtime=rt.finish(),note='Leave-one-family-out changes interact; no claim of independent causal effects. Metrics use uncompressed early RGB.'))
                    print(json.dumps(dict(case=case['id'],config=label,lpips=q['lpips'],ssim=q['ssim'])),flush=True);del zs,frames,value
                finally:rt.close();del rt
        atomic(root/'status.json',dict(status='complete',units=len(samples)*len(configs)))
    except Exception as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
