"""Bounded early closed-loop budget sweep, then explicit full-video candidates."""
import argparse,gc,json,math,os,signal,time
from pathlib import Path
import numpy as np
import torch
from .broad_common import Engine
from .high_sparse_probe_runtime import ProbeHighSparseRuntime as HighSparseRuntime
from .rollout_suite import generate_chunk,video,timed_run,quality,decode
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/high_sparse_probe4_20261030');OLD=Path('artifacts/compensation_probe_20261030/acceptance')
CASES=['basketball_court','butterfly_garden']
CONFIGS=[('global',s,c) for s,c in [(.9,False),(.9,True),(.8,True),(.925,True)]]
def name(cfg):return f'{cfg[0]}_{round(cfg[1]*1000):03d}_'+('comp' if cfg[2] else 'drop')
@torch.no_grad()
def prefix_decode(vae,latents):
    vae.model.clear_cache();out=[]
    for z in latents:
        rgb=vae.decode_to_pixel(z,use_cache=True)[0];assert torch.isfinite(rgb).all()
        out.append(((rgb.float().permute(0,2,3,1)+1)*127.5).clamp(0,255).byte().cpu().numpy())
    assert sum(map(len,out))==105
    return out
@torch.no_grad()
def prefix_quality(frames,reference,metric):
    a=np.concatenate(frames)[21:101];b=np.concatenate(reference)[21:101];metric.cuda();values={k:[] for k in ['mse','lpips','ssim']}
    for i in range(0,80,4):
        x=torch.from_numpy(a[i:i+4].copy()).cuda().permute(0,3,1,2).float()/255;y=torch.from_numpy(b[i:i+4].copy()).cuda().permute(0,3,1,2).float()/255
        values['mse'].extend((x-y).square().mean((1,2,3)).cpu().tolist())
        x=torch.nn.functional.interpolate(x,size=(256,384),mode='bilinear',align_corners=False);y=torch.nn.functional.interpolate(y,size=(256,384),mode='bilinear',align_corners=False)
        values['lpips'].extend(metric(x*2-1,y*2-1).flatten().cpu().tolist())
        pool=lambda t:torch.nn.functional.avg_pool2d(t,11,1,5,count_include_pad=False)
        mx,my=pool(x),pool(y);vx=(pool(x*x)-mx*mx).clamp_min(0);vy=(pool(y*y)-my*my).clamp_min(0);cov=pool(x*y)-mx*my
        ss=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2));values['ssim'].extend(ss.mean((1,2,3)).cpu().tolist())
    metric.cpu();m=float(np.mean(values['mse']))
    return dict(lpips=float(np.mean(values['lpips'])),ssim=float(np.mean(values['ssim'])),psnr=100. if m==0 else -10*math.log10(m),per_frame=values,window=[21,101])
def report():
    rows=[]
    for case in CASES:
        for cfg in CONFIGS:
            p=ROOT/'early'/case/(name(cfg)+'.json')
            if p.exists():rows.append(json.loads(p.read_text()))
    atomic(ROOT/'early_summary.json',dict(units=len(rows),expected=8,complete=len(rows)==8,rows=rows))
    lines=['# 4-query探针路由前段闭环实验','','同模型/seed1001/KV上限；冻结已有权重。质量窗口RGB[21,101)。9chunk=105帧仅作前段筛选，耗时不是30秒端到端结果。80–92.5%指稀疏调用的精确token对稀疏率，不包括补偿成本及dense路径。','','| 场景 | 配置 | 稳态精确稀疏率 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        q=r['quality'];rate=r['runtime']['budget_trace']['338']['exact_sparsity']
        lines.append(f"| {r['case']} | {r['name']} | {rate:.2%} | {q['lpips']:.5f} | {q['ssim']:.5f} | {q['psnr']:.3f} |")
    lines+=['','global统一路由历史与当前页，强制保留query所在当前页；当前摘要为临时4个均值K＋rank8均值V投影，沿用历史训练基底，存在分布迁移。history10保留当前72页、只选10历史页。drop不执行补偿，但保留同常驻分配额度。训练query探针32不变，query tile仍64。历史层0/clean/早期dense不变。']
    (ROOT/'early_report.zh.md').write_text('\n'.join(lines)+'\n')
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['early','full'],default='early');ap.add_argument('--cases',default=','.join(CASES));ap.add_argument('--methods',default='');args=ap.parse_args()
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT/'kernel_checks.json').read_text())['passed']
        identity=dict(code={s:digest(Path('local_experiments')/s) for s in ['high_sparse_runtime.py','high_sparse_checks.py','high_sparse_probe_experiment.py','high_sparse_experiment.py','high_sparse_probe_runtime.py','compensation_stable_kernels.py','compensation_ordered_kernels.py','compensation_runtime.py','rollout_runtime.py','rollout_kernels.py']},model=digest('artifacts/compensation_probe_20261030/model.pt'),configs=CONFIGS,seed=1001,window=[21,101])
        identity=json.loads(json.dumps(identity))
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        configs=[x for x in CONFIGS if not args.methods or name(x) in args.methods.split(',')]
        for case in args.cases.split(','):
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            saved=torch.load(OLD/(case+'_s1001')/'dense_host.latents.pt',weights_only=True).cuda()
            ref=prefix_decode(vae,list(saved[:,:27].split(3,1))) if args.phase=='early' else decode(vae,list(saved.split(3,1)));del saved
            folder=ROOT/args.phase/case;folder.mkdir(parents=True,exist_ok=True)
            for cfg in configs:
                nm=name(cfg);path=folder/(nm+'.json')
                if path.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted',phase=args.phase));return
                disk_guard(ROOT,50,1.5);gc.collect();torch.cuda.empty_cache()
                rt=HighSparseRuntime(engine,*cfg)
                try:
                    if args.phase=='early':
                        rt.reset();latents=[generate_chunk(engine,emb,1001,c) for c in range(9)];frames=prefix_decode(vae,latents);performance=None
                        q=prefix_quality(frames,ref,metric)
                    else:
                        for c in range(6):warm=generate_chunk(engine,emb,1001,c)
                        vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                        telemetry=Telemetry();telemetry.start()
                        try:latents,frames,performance=timed_run(engine,rt,vae,emb,1001)
                        finally:performance_load=telemetry.finish()
                        performance['telemetry']=performance_load
                        q=quality(frames,ref,metric)
                        prefix=torch.load(ROOT/'early'/case/(nm+'.latents.pt'),weights_only=True)
                        assert torch.equal(torch.cat(latents,1)[:,:27].cpu(),prefix),'Full/prefix trajectory differs'
                    value=torch.cat(latents,1).cpu();assert torch.isfinite(value).all() and np.array_equal(frames[0],ref[0])
                    if cfg[0]=='legacy':
                        raw=torch.load(OLD/(case+'_s1001')/'rank8_raw.latents.pt',weights_only=True)
                        assert torch.equal(value,raw[:,:value.shape[1]]),'Legacy generalized path must be bit-exact'
                    result=dict(case=case,name=nm,config=cfg,quality=q,runtime=rt.finish(),performance=performance)
                    atomic(folder/(nm+'.latents.pt'),value,True);video(folder/(nm+'.mp4'),frames);atomic(path,result);report()
                    print(json.dumps(dict(case=case,name=nm,phase=args.phase,quality={k:v for k,v in q.items() if k not in ['per_frame']},seconds=None if performance is None else performance['total_generation_vae_seconds'])),flush=True)
                    del frames,latents,value
                finally:rt.close();del rt
        atomic(ROOT/'status.json',dict(status='completed_requested_units',phase=args.phase));report()
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',phase=args.phase,error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
