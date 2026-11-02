"""Validation-only checkpoint selection, then fresh-seed RGB quality acceptance."""
import argparse,gc,json,os,time,signal,torch
from pathlib import Path
from .broad_common import Engine,cases,ROOT as DATA
from .gap_runtime import GapRuntime
from .sparse_expansion_fast_runtime import ExpansionRuntime
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk,video
from .high_sparse_experiment import prefix_decode,prefix_quality
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/four_gaps_20261031');BASE=Path('artifacts/full_route_grad_20261031/epoch1.pt')
def wait_model(path):
    while not path.exists():
        status=path.parent/'status.json'
        if status.exists() and json.loads(status.read_text())['status'] in ['failed','interrupted']:raise RuntimeError(status.read_text())
        time.sleep(10)
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['val','test'],required=True);ap.add_argument('--seed',type=int,default=404);args=ap.parse_args()
    root=ROOT/('validation' if args.phase=='val' else f'test_s{args.seed}');lock=RunLock(root);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if args.phase=='val':
            paths={'initial':ROOT/'shared/initial.pt','old16':None,'shared_e0':ROOT/'shared/epoch0.pt','matched_e0':ROOT/'matched/epoch0.pt','shared_e1':ROOT/'shared/epoch1.pt','matched_e1':ROOT/'matched/epoch1.pt'};samples=cases('val')
        else:
            wait_model(ROOT/'selection.json');selection=json.loads((ROOT/'selection.json').read_text())
            paths={'initial':ROOT/'shared/initial.pt','old16':None,**{k:Path(v['path']) for k,v in selection['selected'].items()}}
            samples=[c|dict(seed=args.seed,id=f'test_{c["name"]}_s{args.seed}') for c in cases('test') if c['seed']==202]
        ident=dict(phase=args.phase,samples=samples,paths={k:str(v) for k,v in paths.items()},window=[21,101],probes=16,code={n:digest(Path('local_experiments')/n) for n in ['gap_quality.py','gap_runtime.py','gap_model.py','gap_route.py','all_sparse_attention.py']})
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==ident
        else:atomic(root/'identity.json',ident)
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        for label,path in paths.items():
            if path is not None:wait_model(path)
            model_hash=digest(path if path is not None else BASE)
            for case in samples:
                folder=root/case['id'];folder.mkdir(parents=True,exist_ok=True);dest=folder/(label+'.json')
                if dest.exists():assert json.loads(dest.read_text())['model_hash']==model_hash;continue
                if stop[0]:atomic(root/'status.json',dict(status='interrupted'));return
                disk_guard(root,50,1);gc.collect();torch.cuda.empty_cache();emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda();dp=folder/'dense.latents.pt'
                if dp.exists():z=torch.load(dp,weights_only=True).cuda();ref=prefix_decode(vae,list(z.split(3,1)));del z
                else:
                    dense=Runtime(engine,'dense')
                    try:
                        zs=[generate_chunk(engine,emb,case['seed'],c) for c in range(9)];ref=prefix_decode(vae,zs);video(folder/'dense.mp4',ref);atomic(dp,torch.cat(zs,1).cpu(),True);del zs
                    finally:dense.close();del dense
                rt=ExpansionRuntime(engine,BASE,16,1) if label=='old16' else GapRuntime(engine,BASE,probes=16)
                if path is not None:rt.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
                try:
                    atomic(root/'status.json',dict(status='running',case=case['id'],config=label,pid=os.getpid()))
                    zs=[generate_chunk(engine,emb,case['seed'],c) for c in range(9)];frames=prefix_decode(vae,zs);q=prefix_quality(frames,ref,metric)
                    atomic(folder/(label+'.latents.pt'),torch.cat(zs,1).cpu(),True);video(folder/(label+'.mp4'),frames);atomic(dest,dict(case=case,config=label,model_hash=model_hash,quality=q,runtime=rt.finish()))
                    print(json.dumps(dict(case=case['id'],config=label,lpips=q['lpips'],ssim=q['ssim'])),flush=True);del zs,frames
                finally:rt.close();del rt
        if args.phase=='val':
            means={}
            for label in paths:
                qs=[json.loads((root/c['id']/(label+'.json')).read_text())['quality'] for c in samples]
                means[label]={m:sum(q[m] for q in qs)/len(qs) for m in ['lpips','ssim','psnr']};means[label]['score']=means[label]['lpips']+1-means[label]['ssim']
            selected={}
            for arm in ['shared','matched']:
                candidates=['initial',arm+'_e0',arm+'_e1'];best=min(candidates,key=lambda k:means[k]['score']);selected['selected_'+arm]=dict(label=best,path=str(paths[best]),score=means[best]['score'])
            atomic(ROOT/'selection.json',dict(selected=selected,means=means,criterion='validation mean LPIPS + (1-SSIM), initial eligible; selection is not quality acceptance'))
        atomic(root/'status.json',dict(status='complete'))
    except Exception as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
