"""CPU-only, restartable acceptance reporting. Never treats incomplete runs as passes."""
import argparse,csv,hashlib,json,math,os,statistics,time
from contextlib import ExitStack
from pathlib import Path

METHODS=['dense','dense_trim','dense_grouped','mean','mass_mean','mass_var']
CASES=['basketball_court','butterfly_garden','harbor_cranes','lantern_festival']
ROOT=Path('artifacts/rollout_acceptance_20261030')

def write(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);os.replace(tmp,path)
def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def visuals(root,case):
    import av
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    folder=root/case
    if not all((folder/f'{m}.rep0.json').exists() for m in METHODS):return
    output=folder/'comparison.mp4';sheet=folder/'contact_sheet.png'
    if output.exists() and sheet.exists():return
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    canvas=Image.new('RGB',(7*192,6*156+30),'white');draw=ImageDraw.Draw(canvas)
    ids=[0,80,160,240,320,400,479]
    for col,i in enumerate(ids):draw.text((col*192+4,3),f'{i/16:.2f}s',font=font,fill='black')
    metadata={}
    temporary=output.with_name('comparison.tmp.mp4')
    with ExitStack() as stack:
        containers=[stack.enter_context(av.open(str(folder/f'{m}.mp4'))) for m in METHODS]
        readers=[c.decode(video=0) for c in containers]
        for m,c in zip(METHODS,containers):
            s=c.streams.video[0]
            assert (s.width,s.height)==(768,512) and float(s.average_rate)==16
            metadata[m]=dict(width=s.width,height=s.height,fps=float(s.average_rate))
        writer=stack.enter_context(imageio.get_writer(temporary,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18','-threads','2']))
        for i in range(480):
            frame=Image.new('RGB',(1152,568),'white');labels=ImageDraw.Draw(frame)
            for j,(m,reader) in enumerate(zip(METHODS,readers)):
                image=Image.fromarray(next(reader).to_ndarray(format='rgb24'))
                x,y=(j%3)*384,(j//3)*284
                labels.text((x+4,y+3),f'{m} | {i/16:.2f}s',font=font,fill='black')
                frame.paste(image.resize((384,256),Image.Resampling.LANCZOS),(x,y+28))
                if i in ids:
                    draw.text((4,j*156+32),m,font=font,fill='black')
                    canvas.paste(image.resize((192,128),Image.Resampling.LANCZOS),(ids.index(i)*192,j*156+58))
            writer.append_data(np.asarray(frame))
        assert all(next(r,None) is None for r in readers)
    os.replace(temporary,output);canvas.save(sheet)
    write(folder/'visual_provenance.json',json.dumps(dict(videos={m:sha(folder/f'{m}.mp4') for m in METHODS},metadata=metadata,frames=480,contact_frame_indices=ids,processing='MP4 decode; identical Lanczos downsample; no color adjustment. Metrics use pre-encoding RGB, not these images.',visual_review='pending'),indent=2))

def report(root):
    rows=[];raw=[];complete=True
    for case in CASES:
        for m in METHODS:
            results=[read(root/case/f'{m}.rep{r}.json') for r in range(3) if (root/case/f'{m}.rep{r}.json').exists()]
            complete &= len(results)==3
            for r in results:raw.append(dict(case=case,method=m,repeat=r['rep'],generation_s=r['generation_seconds'],total_s=r['total_generation_vae_seconds'],peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes']))
            if not results:continue
            times=[r['total_generation_vae_seconds'] for r in results]
            qpath=root/case/f'{m}.quality.json';q=read(qpath) if qpath.exists() else {}
            rows.append(dict(case=case,method=m,n=len(results),total_s=statistics.median(times),min_s=min(times),max_s=max(times),generation_s=statistics.median(r['generation_seconds'] for r in results),peak_gib=max(r['peak_allocated_bytes'] for r in results)/2**30,persistent_bytes=max(r['persistent_self_kv_and_aux_bytes'] for r in results),budget_ok=all(r['persistent_self_kv_and_aux_bytes']<=r['limit_bytes'] for r in results),lpips=q.get('lpips'),tail_lpips=q.get('tail_lpips'),ssim=q.get('ssim'),dark_dip_pp=q.get('dark_dip_max_pp')))
    lookup={(r['case'],r['method']):r for r in rows}
    for r in rows:
        base=lookup.get((r['case'],'dense'));r['speedup']=base['total_s']/r['total_s'] if base else None
    lines=['# 真实端到端验收进度与结果','',f'已提交 {len(raw)}/72 个完整视频计时单元。状态：'+('全部测量完成；仍需视觉与负载审核。' if complete else '测试进行中；不能宣称联合通过。'),'',
        '固定同模型、同 seed、原生 RoPE，无 rebase。30 秒、480 帧、512×768、16fps。计时包含生成、clean 编码、缓存预处理/更新、真实稀疏注意力、VAE 和 RGB 回传；不包含加载/编译预热/视频压制/指标计算。', '',
        '三次计时重复是同一视频的复现，不是三个独立质量样本。表中时间为已完成重复的中位数及最小–最大范围；n<3 为临时结果。共享 H200 的其他任务会干扰计时，未经负载审核不能作为独占 GPU 加速结论。','',
        '| 场景 | 方法 | n | 含VAE秒 [min,max] | 生成秒 | 对dense速度比 | LPIPS | 尾5秒LPIPS | 峰值allocated GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        val=lambda x:'待测' if x is None else f'{x:.3f}'
        lines.append(f"| {r['case']} | {r['method']} | {r['n']} | {r['total_s']:.2f} [{r['min_s']:.2f},{r['max_s']:.2f}] | {r['generation_s']:.2f} | {val(r['speedup'])} | {val(r['lpips'])} | {val(r['tail_lpips'])} | {r['peak_gib']:.2f} |")
    lines+=['','预声明工程门槛：四场景均更快，含 VAE 几何平均速度比 ≥1.05；每场景 LPIPS≤0.10、尾5秒≤0.15；另需视觉稳定性检查。这是参考保真标准，不是通用主观画质标准。','']
    gates={}
    for m in METHODS[3:]:
        rr=[lookup[(c,m)] for c in CASES if (c,m) in lookup]
        ready=len(rr)==4 and all(r['n']==3 and lookup[(r['case'],'dense')]['n']==3 for r in rr)
        failed=[r['case'] for r in rr if r['lpips'] is not None and (r['lpips']>.10 or r['tail_lpips']>.15)]
        gm=math.exp(statistics.mean(math.log(r['speedup']) for r in rr)) if len(rr)==4 else None
        speed=bool(ready and gm>=1.05 and all(r['speedup']>1 for r in rr))
        gates[m]=dict(measurements_complete=ready,geometric_mean_speedup=gm,numerical_speed_gate=speed,reference_fidelity_failed_cases=failed,visual_review='pending',shared_gpu_timing_review='pending',joint_pass=False)
        lines.append(f"- {m}："+('参考保真门槛未通过：'+', '.join(failed)+'。' if failed else ('已测场景尚无参考保真超限。' if rr else '该方法尚无完整视频结果。'))+('速度数值门槛通过。' if speed else '速度门槛未通过或测量未齐。')+'视觉与共享负载审核未完成，不判定联合通过。')
    lines+=['','dense_trim 隔离为辅助信息腾出空间造成的历史缩短；dense_grouped 进一步隔离硬分组、页顺序和部分淘汰政策。它们保留的 token 集合不同。后三种 sparse 方法选择相同的72历史页+72当前页。常驻KV与辅助信息共同受5,096,079,360字节上限约束；峰值显存另列，不能声称峰值相等。','',
        '质量指标来自压制前 RGB：全分辨率PSNR，256×384 LPIPS AlexNet及11×11均匀窗SSIM；主指标排除最初9帧。每帧指标和亮度/色度/运动代理保存在各方法 quality.json，代理指标不单独判定画质。','',
        '已测四个场景曾用于离线确认，不是全新盲测；本轮冻结权重，不能根据这些视频调参再称独立验收。','',
        '视频与接触图（对所有方法相同缩放，无颜色调整；接触图不能替代完整时序观看）：','']
    for c in CASES:
        if (root/c/'comparison.mp4').exists():lines.append(f'- {c}：[并排视频]({c}/comparison.mp4) · [接触图]({c}/contact_sheet.png)')
    lines+=['','已确认共享GPU干扰：basketball_court native dense 首轮含VAE 96.38秒，第二轮57.35秒。当前速度不能视为可靠加速验收；不删掉慢样本，也不选择性报告最快值。GPU过程记录见 gpu_telemetry.jsonl。','', '[视觉检查记录](visual_review.zh.md) · [时序与计时图](figures/timing.png)','']
    controls=[]
    for method in ['dense','dense_host','mass_var']:
        values=[read(p) for p in sorted((root/'dense_control').glob(f'{method}.rep*.json'))]
        if values:
            ts=[v['total_generation_vae_seconds'] for v in values]
            controls.append(dict(method=method,n=len(values),median_s=statistics.median(ts),min_s=min(ts),max_s=max(ts),exact_native=all(v.get('bit_exact_native_dense',False) for v in values) if method=='dense_host' else None))
    if controls:
        lines+=['独立强dense对照（GPU3内部配对，不能和GPU5数字混用；也受共享负载干扰）：','', '| 方法 | n | 含VAE中位秒 [min,max] | 与native逐元素一致 |','|---|---:|---:|---|']
        for r in controls:lines.append(f"| {r['method']} | {r['n']} | {r['median_s']:.2f} [{r['min_s']:.2f},{r['max_s']:.2f}] | {r['exact_native']} |")
    quiet=root/'quiet_control/summary.json'
    if quiet.exists():
        q=read(quiet)
        lines+=['','空闲卡补充对照（单场景，不能替代四场景验收）：', '',f"mass_var对native dense {q['speedup_vs_native']:.3f}×，对保持输出逐元素一致的host-index dense {q['speedup_vs_host_dense']:.3f}×。5秒进程采样未见外部任务：{q['no_observed_contention']}；mass_var与主实验latent一致：{q['mass_var_matches_primary_latents']}。",'', '[补充对照完整报告](quiet_control/report.zh.md)']
    profiles=[read(p) for p in sorted((root/'profile').glob('*.json')) if p.name!='status.json']
    if profiles:
        lines+=['','独立组件profile：真实闭环生成，但插入CUDA events；无VAE、仅层1..29，不作为主计时。事件间隔可包含CPU提交等待，不等于纯kernel活跃时间。','', '| 方法 | profile生成wall秒 | 分组重排秒 | 原型聚合秒 | 环写秒 | 路由秒 | sparse attention秒 | clean attention秒 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for p in profiles:
            t=p['profile_gpu_ms'];v=[t.get(k,0)/1000 for k in ['group_reorder','aggregate','ring_commit','route','sparse_attention','clean_attention']]
            lines.append(f"| {p['method']} | {p['profiled_generation_wall_s']:.3f} | "+' | '.join(f'{x:.3f}' for x in v)+' |')
    secondary=[read(p) for p in sorted((root/'secondary_quality').glob('*.json')) if p.name!='status.json']
    if secondary:
        lines+=['','辅助参考比较（事后诊断，不改变验收门槛；仍是闭环轨迹差异，不是单独attention误差）：','', '| 场景 | sparse方法 | 辅助dense参考 | LPIPS | 尾5秒LPIPS |','|---|---|---|---:|---:|']
        for q in secondary:lines.append(f"| {q['case']} | {q['method']} | {q['reference']} | {q['lpips']:.3f} | {q['tail_lpips']:.3f} |")
    lines+=['','独立机制实验（V/集合相关选择、page内重分配、V指导分组）不在本冻结部署验收中，尚不能报告为已验证。','', '[协议](protocol.zh.md) · [恢复说明](RESUME.zh.md) · [完整数据](summary.json)']
    write(root/'report.zh.md','\n'.join(lines)+'\n')
    write(root/'summary.json',json.dumps(dict(completed_units=len(raw),complete=complete,rows=rows,gates=gates,dense_controls=controls,timing_acceptance='inconclusive: observed shared-GPU interference'),indent=2))
    if raw:
        tmp=root/'timings.csv.tmp'
        with tmp.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
        os.replace(tmp,root/'timings.csv')
    return complete

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    os.environ['OMP_NUM_THREADS']='2'
    previous=None;last_change=time.monotonic()
    while True:
        for case in CASES:visuals(ROOT,case)
        done=report(ROOT)
        count=(read(ROOT/'summary.json')['completed_units'],len(list((ROOT/'dense_control').glob('*.rep*.json'))), (ROOT/'quiet_control/summary.json').exists())
        if count!=previous:
            from .rollout_figures import main as figures
            figures();previous=count;last_change=time.monotonic()
        if time.monotonic()-last_change>3600:
            print('No new committed unit for one hour; exiting watcher. Restart after resuming suite.',flush=True);break
        if done or not a.watch:break
        time.sleep(30)

if __name__=='__main__':main()
