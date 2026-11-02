"""Four prompts, fresh seed1001, independent actual 30s runs incl VAE, 3 repeats."""
import gc,json,os,signal
from pathlib import Path
import numpy as np
import torch
from .broad_common import Engine
from .rollout_dense_control import HostDense
from .improve_runtime import FastRuntime
from .compensation_runtime import CompensationRuntime
from .rollout_suite import generate_chunk,timed_run,decode,quality,video
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .compensation_probe import ROOT
METHODS=['dense_host','mass_var','rank8_raw','rank8_guarded']
CASES=['basketball_court','butterfly_garden','harbor_cranes','lantern_festival']

def report():
    rows=[]
    for folder in sorted((ROOT/'acceptance').glob('*_s1001')):
        for method in METHODS:
            paths=sorted(folder.glob(f'{method}.rep*.json'))
            if not paths:continue
            runs=[json.loads(p.read_text()) for p in paths];q=json.loads((folder/f'{method}.quality.json').read_text())['per_frame']
            rows.append(dict(condition=folder.name,method=method,n=len(runs),early_lpips=float(np.mean(q['lpips'][21:101])),early_ssim=float(np.mean(q['ssim'][21:101])),early_psnr=100. if np.mean(q['mse'][21:101])==0 else float(-10*np.log10(np.mean(q['mse'][21:101]))),seconds=float(np.median([r['total_generation_vae_seconds'] for r in runs])),min_s=min(r['total_generation_vae_seconds'] for r in runs),max_s=max(r['total_generation_vae_seconds'] for r in runs),clean=all(r['telemetry']['no_observed_foreign_processes'] for r in runs),cache_bytes=runs[0]['persistent_self_kv_and_aux_bytes']))
    units=sum(r['n'] for r in rows);atomic(ROOT/'acceptance_summary.json',dict(units=units,expected_units=48,complete=units==48,rows=rows))
    lines=['# 补偿方案端到端验收','','固定Self-Forcing权重、原RoPE、338历史页＋辅助不超过native360预算；每query精确72历史页＋72当前页。补偿额外计算单独计入真实耗时，不声称同总FLOPs。4个已有prompt，新的seed1001；不是新prompt盲测。', '',f'完成 {units}/48 单元。质量主窗口为[21,101)帧（1.3125–6.3125秒）。每单位独立完整30秒生成含VAE，加载、预热、指标与压制不计时。共享GPU采样污染时不据此认定稳定速度比。','','| 条件 | 方法 | 次数 | 前段LPIPS↓ | SSIM↑ | PSNR↑ | 完整耗时中位秒 | 范围秒 | 无外部进程采样 |','|---|---|---:|---:|---:|---:|---:|---|---|']
    for r in rows:lines.append(f"| {r['condition']} | {r['method']} | {r['n']} | {r['early_lpips']:.5f} | {r['early_ssim']:.5f} | {r['early_psnr']:.3f} | {r['seconds']:.3f} | {r['min_s']:.3f}–{r['max_s']:.3f} | {r['clean']} |")
    lines+=['','rank8_guarded 的层/head补偿强度仅用val拟合；rank8_raw所有头强度1。全片／尾段逐帧数据仍保留在各quality.json供稳定性诊断，不使用旧尾段保真门槛。尚未完成视觉审核，不宣布质量速度联合通过。']
    (ROOT/'acceptance.zh.md.tmp').write_text('\n'.join(lines)+'\n');os.replace(ROOT/'acceptance.zh.md.tmp',ROOT/'acceptance.zh.md')

@torch.no_grad()
def main():
    assert json.loads((ROOT/'deployment/gate.json').read_text())['passed']
    select=json.loads((ROOT/'selection.json').read_text());stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import lpips
    from utils.wan_wrapper import WanVAEWrapper
    engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    identity=dict(model=digest(ROOT/'model.pt'),selection=digest(ROOT/'selection.json'),gate=digest(ROOT/'deployment/gate.json'),code={s:digest(Path('local_experiments')/s) for s in ['compensation_accept.py','compensation_runtime.py','compensation_kernels.py','rollout_suite.py','rollout_runtime.py','improve_runtime.py']},methods=METHODS,seed=1001,repeats=3)
    for case in CASES:
        folder=ROOT/'acceptance'/f'{case}_s1001';lock=RunLock(folder);ref=None
        try:
            ident=identity|dict(case=case)
            if (folder/'identity.json').exists():assert json.loads((folder/'identity.json').read_text())==ident
            else:atomic(folder/'identity.json',ident)
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            for rep in range(3):
                order=METHODS[rep:]+METHODS[:rep]
                for method in order:
                    path=folder/f'{method}.rep{rep}.json'
                    if path.exists():continue
                    if stop[0]:report();return
                    disk_guard(ROOT,50,2);gc.collect();torch.cuda.empty_cache()
                    rt=HostDense(engine) if method=='dense_host' else FastRuntime(engine) if method=='mass_var' else CompensationRuntime(engine,ROOT/'model.pt',select,'raw' if method=='rank8_raw' else 'guarded')
                    try:
                        for c in range(6):warm=generate_chunk(engine,emb,1001,c)
                        vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                        telemetry=Telemetry();telemetry.start()
                        try:latents,frames,result=timed_run(engine,rt,vae,emb,1001)
                        finally:load=telemetry.finish()
                        value=torch.cat(latents,1).cpu();lp=folder/f'{method}.latents.pt'
                        result.update(method=method,case=case,seed=1001,rep=rep,telemetry=load)
                        if rep==0:
                            if method=='dense_host':ref=frames
                            if ref is None:
                                saved=torch.load(folder/'dense_host.latents.pt',weights_only=True).cuda();ref=decode(vae,list(saved.split(3,1)));del saved
                            assert np.array_equal(frames[0],ref[0]),'initial9 frames changed'
                            qm=quality(frames,ref,metric);atomic(folder/f'{method}.quality.json',qm);atomic(lp,value,True);video(folder/f'{method}.mp4',frames)
                        else:assert torch.equal(value,torch.load(lp,weights_only=True)),'repeat full latent differs'
                        atomic(path,result);report();print(json.dumps(dict(case=case,method=method,rep=rep,seconds=result['total_generation_vae_seconds'],clean=load['no_observed_foreign_processes'])),flush=True)
                        del value,latents,frames
                    finally:rt.close();del rt
        except Exception as exc:atomic(folder/'status.json',dict(stage='failed',error=repr(exc)));raise
        finally:lock.close()
    report()
if __name__=='__main__':main()
