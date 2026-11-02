"""Paired full generation+VAE wall timing and uncompressed RGB quality."""
import argparse,gc,json,math,os,signal,time
from pathlib import Path
import numpy as np
import torch
from .broad_common import Engine,noise
from .mass_common import cases as source_cases
from .rollout_runtime import Runtime,ROOT,METHODS
from .learned_cache.io import atomic,digest,disk_guard,RunLock

DEFAULT_CASES=['basketball_court','butterfly_garden','harbor_cranes','lantern_festival']

def guard():return disk_guard(ROOT,min_free_gib=50,max_run_gib=4)

@torch.no_grad()
def generate_chunk(engine,emb,seed,chunk):
    x=noise(seed,chunk,0)
    for step in range(4):
        clean=engine.forward(x,emb,chunk,step)
        if step<3:x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),noise(seed,chunk,step+1).flatten(0,1),
            torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
    engine.forward(clean,emb,chunk,4)
    return clean

@torch.no_grad()
def decode(vae,latents):
    vae.model.clear_cache();frames=[];count=0
    for z in latents:
        rgb=vae.decode_to_pixel(z,use_cache=True)[0]
        assert torch.isfinite(rgb).all()
        pixels=((rgb[:480-count].float().permute(0,2,3,1)+1)*127.5).clamp(0,255).byte().cpu().numpy()
        frames.append(pixels);count+=len(pixels)
        if count==480:break
    assert count==480
    return frames

@torch.no_grad()
def timed_run(engine,rt,vae,emb,seed):
    rt.reset();vae.model.clear_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();latents=[]
    for chunk in range(41):latents.append(generate_chunk(engine,emb,seed,chunk))
    torch.cuda.synchronize();generation=time.perf_counter()-start
    frames=decode(vae,latents);torch.cuda.synchronize();total=time.perf_counter()-start
    result=dict(generation_seconds=generation,total_generation_vae_seconds=total,decode_seconds=total-generation,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**rt.finish())
    return latents,frames,result

@torch.no_grad()
def quality(frames,reference,lpips):
    lpips=lpips.cuda();mses=[];perceptual=[];ssims=[];brightness=[];chroma=[];motion=[];previous=None
    for pixels,ref in zip(frames,reference):
        for i in range(0,len(pixels),4):
            a=torch.from_numpy(pixels[i:i+4].copy()).cuda().permute(0,3,1,2).float()/255
            b=torch.from_numpy(ref[i:i+4].copy()).cuda().permute(0,3,1,2).float()/255
            mses.extend((a-b).square().mean((1,2,3)).cpu().tolist())
            a=torch.nn.functional.interpolate(a,size=(256,384),mode='bilinear',align_corners=False)
            b=torch.nn.functional.interpolate(b,size=(256,384),mode='bilinear',align_corners=False)
            perceptual.extend(lpips(a*2-1,b*2-1).flatten().cpu().tolist())
            pool=lambda x:torch.nn.functional.avg_pool2d(x,11,1,5,count_include_pad=False)
            ma,mb=pool(a),pool(b);va=(pool(a*a)-ma*ma).clamp_min(0);vb=(pool(b*b)-mb*mb).clamp_min(0);cov=pool(a*b)-ma*mb
            ssim=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
            ssims.extend(ssim.mean((1,2,3)).cpu().tolist())
        for frame in pixels:
            small=frame[::4,::4].astype(np.float32)/255
            color=small.mean((0,1));brightness.append(float(color@np.array([.2126,.7152,.0722])))
            chroma.append((color/(color.sum()+1e-8)).tolist())
            motion.append(0. if previous is None else float(np.abs(small-previous).mean()));previous=small
    lpips.cpu();torch.cuda.synchronize()
    assert len(mses)==480
    y=np.array(brightness);padded=np.pad(y,4,mode='edge');med=np.array([np.median(padded[i:i+9]) for i in range(480)])
    mean=float(np.mean(mses[9:]));tail=float(np.mean(mses[400:]))
    return dict(psnr_db=100. if mean==0 else -10*math.log10(mean),lpips=float(np.mean(perceptual[9:])),
        ssim=float(np.mean(ssims[9:])),tail_psnr_db=100. if tail==0 else -10*math.log10(tail),tail_lpips=float(np.mean(perceptual[400:])),
        brightness_step_pp=float(np.abs(np.diff(y[9:])).mean()*100),dark_dip_max_pp=float(np.maximum(med-y,0)[9:].max()*100),
        motion_l1=float(np.mean(motion[9:])),chroma_drift=float(np.linalg.norm(np.array(chroma)-np.mean(chroma[:9],0),axis=1)[9:].mean()),
        per_frame=dict(mse=mses,lpips=perceptual,ssim=ssims,brightness=brightness,chroma=chroma,motion=motion),
        definitions='PSNR full512x768 uncompressed8bit RGB; LPIPS AlexNet and uniform11-window SSIM at256x384; main excludes9initial frames; tail frames400:480. Temporal proxies are not standalone quality criteria.')

def video(path,frames):
    import imageio.v2 as imageio
    temp=path.with_name(path.stem+'.tmp.mp4')
    with imageio.get_writer(temp,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,
         ffmpeg_log_level='error',output_params=['-crf','18','-threads','4']) as writer:
        for pixels in frames:
            for frame in pixels:writer.append_data(frame)
    os.replace(temp,path)

@torch.no_grad()
def run(args):
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['TORCH_HOME']=str((ROOT/'torch_cache').resolve())
    import lpips
    from utils.wan_wrapper import WanVAEWrapper
    stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    items=[c for c in source_cases('test') if c['name'] in args.cases.split(',') and c['seed']==606]
    engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False);units=0
    sources=['rollout_runtime.py','rollout_kernels.py','rollout_suite.py','broad_common.py','response_hard.py','cache_grouping.py']
    identity=dict(code={s:digest(Path('local_experiments')/s) for s in sources},
        weights=digest('artifacts/mass_variance_20261030/train_00_30/best.pt'),repeats=args.repeats,methods=METHODS,
        output='480frames/16fps/512x768',protocol='physical338+summary<=dense360;layer0andclean dense;no rebase',fused=True)
    for case in items:
        root=ROOT/case['name'];lock=RunLock(root);reference=None
        try:
            ident=identity|dict(case=case)
            if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==ident
            else:atomic(root/'identity.json',ident)
            if all((root/f'{m}.rep{r}.json').exists() for m in METHODS for r in range(args.repeats)):
                atomic(root/'status.json',dict(status='completed'));continue
            emb=torch.load(Path('artifacts/mass_variance_20261030/embeddings')/f'{case["name"]}.pt',weights_only=True).cuda()
            for rep in range(args.repeats):
                order=METHODS if rep==0 else METHODS[rep*2:]+METHODS[:rep*2]
                for method in order:
                    path=root/f'{method}.rep{rep}.json'
                    if path.exists():continue
                    if stop[0]:break
                    guard();gc.collect();torch.cuda.empty_cache()
                    rt=Runtime(engine,method)
                    try:
                        # All growing/steady kernel shapes and both VAE decode shapes warmed.
                        for ch in range(6):warm=generate_chunk(engine,emb,case['seed'],ch)
                        vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True)
                        del warm
                        latents,frames,result=timed_run(engine,rt,vae,emb,case['seed'])
                        assert rt.commits[1:]==([41]*29 if rt.grouped else [0]*29)
                        if rep==0:
                            atomic(root/f'{method}.latents.pt',torch.cat(latents,1).cpu(),True)
                            if method=='dense':reference=frames
                            if reference is None:
                                saved=torch.load(root/'dense.latents.pt',weights_only=True).cuda()
                                reference=decode(vae,list(saved.split(3,1)));del saved
                            assert np.array_equal(frames[0],reference[0]),'first9 RGB frames differ'
                            qm=quality(frames,reference,metric);atomic(root/f'{method}.quality.json',qm)
                            video(root/f'{method}.mp4',frames)
                            result['quality']= {k:v for k,v in qm.items() if k!='per_frame'}
                        else:
                            first=torch.load(root/f'{method}.latents.pt',weights_only=True)
                            assert torch.equal(torch.cat(latents,1).cpu(),first),'timing repeat changed generated latents'
                        result.update(case=case['name'],rep=rep,completed=True)
                        atomic(path,result);units+=1
                        print(json.dumps(dict(case=case['name'],method=method,rep=rep,generation=result['generation_seconds'],total=result['total_generation_vae_seconds'])),flush=True)
                        atomic(root/'status.json',dict(status='running',method=method,rep=rep,disk=guard()))
                        del latents,frames
                    finally:rt.close();del rt
                    if args.max_units and units>=args.max_units:stop[0]=True
                if stop[0]:break
            done=all((root/f'{m}.rep{r}.json').exists() for m in METHODS for r in range(args.repeats))
            atomic(root/'status.json',dict(status='completed' if done else 'interrupted',disk=guard()))
        except Exception as exc:
            atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
        finally:lock.close()
        if stop[0]:break

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',default=','.join(DEFAULT_CASES));p.add_argument('--repeats',type=int,default=3);p.add_argument('--max-units',type=int,default=0)
    run(p.parse_args())
