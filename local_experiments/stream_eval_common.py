"""Isolated official-method benchmark utilities, common native VAE and metrics."""
import ast,importlib.util,json,os,time,gc,threading
from pathlib import Path
import numpy as np
import torch
from .learned_cache.io import atomic,digest,disk_guard
from .gap_gpu_processes import inventory
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'artifacts/stream_methods_20261102'

def native_vae():
    path=REPO/'wan/modules/vae.py';spec=importlib.util.spec_from_file_location('stream_eval_native_vae',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    tree=ast.parse((REPO/'utils/wan_wrapper.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WanVAEWrapper')
    ns=dict(torch=torch,nn=torch.nn,_video_vae=m._video_vae)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(REPO/'utils/wan_wrapper.py'),'exec'),ns)
    cwd=os.getcwd()
    try:os.chdir(REPO);return ns['WanVAEWrapper']().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    finally:os.chdir(cwd)

@torch.no_grad()
def decode(vae,z,limit=480):
    vae.model.clear_cache();frames=[];count=0
    for chunk in z.split(3,1):
        rgb=vae.decode_to_pixel(chunk,use_cache=True)[0]
        pixels=((rgb[:limit-count].float().permute(0,2,3,1)+1)*127.5).clamp(0,255).byte().cpu().numpy()
        frames.append(pixels);count+=len(pixels)
        if count==limit:break
    assert count==limit
    return frames

class Watch:
    def __init__(self):self.rows=[];self.stop=threading.Event()
    def sample(self):
        try:
            rows=[p for ps in inventory(os.environ['CUDA_VISIBLE_DEVICES']).values() for p in ps]
            # The identified parent holds idle VRAM only; no kernels run in it during subprocess execution.
            allowed={os.getpid(),int(os.environ['STREAM_RESERVATION_PID'])}
            self.rows.append(dict(time=time.time(),contexts=rows,peers=[r for r in rows if r['pid'] not in allowed]))
        except Exception as e:self.rows.append(dict(time=time.time(),error=repr(e)))
    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop.wait(2):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();self.sample()
    def validate(self):assert self.rows and all(not r.get('peers') and not r.get('error') for r in self.rows), self.rows

def save_video(path,frames):
    import imageio.v2 as imageio
    tmp=path.with_suffix('.tmp.mp4')
    with imageio.get_writer(tmp,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18','-threads','4']) as w:
        for chunk in frames:
            for f in chunk:w.append_data(f)
    tmp.replace(path)

@torch.no_grad()
def quality(out,method,frames,vae):
    if method=='dense':return dict(reference=True)
    import lpips
    from .high_sparse_experiment import prefix_quality
    os.environ['TORCH_HOME']=str(REPO/'artifacts/rollout_acceptance_20261030/torch_cache')
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    reference=torch.load(out.parent/'dense/latents.pt',map_location='cpu',weights_only=True)[:,:27].cuda()
    ref=decode(vae,reference,105);q=prefix_quality(frames,ref,metric)
    a=np.concatenate(frames)[:101].astype(np.float32)/255
    brightness=a.mean((1,2))@np.array([.2126,.7152,.0722]);motion=np.abs(a[22:101:4,::4,::4]-a[21:100:4,::4,::4]).mean()
    q.update(brightness_step_pp=float(np.abs(np.diff(brightness[21:])).mean()*100),motion_l1=float(motion),interpretation='LPIPS/SSIM/PSNR measure closeness to same-input dense, not absolute perceptual quality; temporal proxies alone cannot establish better quality.')
    return q

@torch.no_grad()
def benchmark(method,out,generate,metadata,after_warm=None,storage=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);vae=native_vae()
    disk_guard(ROOT,50,1)
    atomic(out/'status.json',dict(status='warming',method=method))
    z=generate();assert torch.isfinite(z).all();decode(vae,z);del z
    if after_warm:after_warm()
    gc.collect();torch.cuda.empty_cache()
    existing=out/'latents.pt';ref=None;reps=[]
    for rep in range(2):
        dest=out/f'rep{rep}.json'
        if dest.exists():reps.append(json.loads(dest.read_text()));continue
        atomic(out/'status.json',dict(status='measuring',method=method,rep=rep))
        with Watch() as watch:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            z=generate();torch.cuda.synchronize();gen=time.perf_counter()-start
            frames=decode(vae,z);torch.cuda.synchronize();total=time.perf_counter()-start
        atomic(out/f'rep{rep}.peers.json',dict(reservation_pid=int(os.environ['STREAM_RESERVATION_PID']),scope='all compute and graphics contexts; parent VRAM reservoir is idle',samples=watch.rows));watch.validate()
        cpu=z.cpu();assert cpu.shape==(1,123,16,64,96) and torch.isfinite(cpu).all()
        if existing.exists():assert torch.equal(cpu,torch.load(existing,weights_only=True)), 'Repeated video differs; inspect state reset'
        else:atomic(existing,cpu,True)
        row=dict(rep=rep,generation_seconds=gen,decode_seconds=total-gen,total_generation_vae_seconds=total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),storage=storage() if storage else metadata.get('storage'),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],metadata=metadata)
        atomic(dest,row);reps.append(row);print(json.dumps(dict(method=method,rep=rep,total=total,generation=gen)),flush=True)
        if not (out/'video.mp4').exists():save_video(out/'video.mp4',frames)
        if not (out/'quality.json').exists():atomic(out/'quality.json',quality(out,method,frames,vae))
        del z,frames,cpu;gc.collect();torch.cuda.empty_cache();disk_guard(ROOT,50,1)
    if not (out/'quality.json').exists():
        z=torch.load(existing,weights_only=True).cuda();frames=decode(vae,z)
        atomic(out/'quality.json',quality(out,method,frames,vae));del z,frames
    atomic(out/'result.json',dict(status='complete',method=method,repetitions=reps,quality=json.loads((out/'quality.json').read_text())))
    atomic(out/'status.json',dict(status='complete'))
