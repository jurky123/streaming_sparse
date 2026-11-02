"""CPU-only Markdown status, truthful gates and synchronized video comparisons."""
import argparse,json,math,os,statistics,time
from pathlib import Path
ROOT=Path('artifacts/rollout_improve_20261030')
def read(p):return json.loads(p.read_text())
def write(p,text):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(text);os.replace(tmp,p)
def visuals(folder,proxy):
    import av
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from contextlib import ExitStack
    names=['dense','mass_var','pilot_anchor','pilot_candidate']+(['pilot_swap'] if proxy else [])
    if not all((folder/f'{m}.rep0.json').exists() for m in names):return
    if (folder/'comparison.mp4').exists() and (folder/'contact_sheet.png').exists():return
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);ids=[0,80,160,240,320,400,479]
    sheet=Image.new('RGB',(1344,len(names)*156+30),'white');d=ImageDraw.Draw(sheet)
    for j,i in enumerate(ids):d.text((j*192+4,3),f'{i/16:.2f}s',font=font,fill='black')
    columns=2 if len(names)<=4 else 3;rows=math.ceil(len(names)/columns)
    tmp=folder/'comparison.tmp.mp4'
    with ExitStack() as stack:
        readers=[stack.enter_context(av.open(str(folder/f'{m}.mp4'))).decode(video=0) for m in names]
        writer=stack.enter_context(imageio.get_writer(tmp,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18','-threads','2']))
        for i in range(480):
            frame=Image.new('RGB',(columns*384,rows*284),'white');labels=ImageDraw.Draw(frame)
            for j,(m,r) in enumerate(zip(names,readers)):
                im=next(r).to_image();x,y=j%columns*384,j//columns*284
                labels.text((x+4,y+3),f'{m} | {i/16:.2f}s',font=font,fill='black');frame.paste(im.resize((384,256),Image.Resampling.LANCZOS),(x,y+28))
                if i in ids:
                    d.text((4,30+j*156),m,font=font,fill='black');sheet.paste(im.resize((192,128),Image.Resampling.LANCZOS),(ids.index(i)*192,58+j*156))
            writer.append_data(np.asarray(frame))
        assert all(next(r,None) is None for r in readers)
    os.replace(tmp,folder/'comparison.mp4');sheet.save(folder/'contact_sheet.png')
    write(folder/'visual_metadata.json',json.dumps(dict(frames=480,fps=16,methods=names,frame_indices=ids,processing='Identical Lanczos downsampling, no color adjustment; source videos compressed. Quality metrics use uncompressed RGB.',visual_review='pending'),indent=2))
def report():
    ROOT.mkdir(exist_ok=True);lines=['# 实施与端到端验收进度','', '本轮先做6层pilot（4/9/14/19/24/29），其余23个稀疏层保留冻结mass_var，第0层dense。硬64-token物理page，4代表/page，不引入按距离递减代表数。原生RoPE，不rebase。','']
    if (ROOT/'driver_status.json').exists():lines+=['当前阶段：`'+read(ROOT/'driver_status.json').get('stage','unknown')+'`。','']
    if (ROOT/'pilot/status.json').exists():
        t=read(ROOT/'pilot/status.json');lines+=['训练状态：`'+str(t.get('status'))+'`；cursor：`'+str(t.get('cursor'))+'`。','']
    checks=[read(p) for p in (ROOT/'correctness').glob('*.*.json')]
    if checks:
        lines+=['已完成长轨迹数值检查：','', '| 场景 | 变体 | 与冻结视频latent逐元素一致 | 最坏所选KV kernel相对MSE |','|---|---|---|---:|']
        for r in checks:lines.append(f"| {r['case']} | {r['variant']} | {r['exact_primary_latents']} | {max(x['max_nmse'] for x in r['kernel_checks']):.3g} |")
    lines+=['','原commit_fast融合改变原型归约舍入，虽满足局部容差却改变完整轨迹，已拒绝作为等价工程加速。另测只融合K/V搬移、保留原聚合kernel的scatter_fast；只有全轨迹精确通过才进入确认组。','']
    if (ROOT/'selection.json').exists():
        s=read(ROOT/'selection.json');lines+=['验证集选出的机制：`'+s['selected_method']+'`；离线改善门槛通过：'+str(s['offline_gate_passed'])+'；一次proxy修正纳入确认：'+str(s['proxy_promoted'])+'。','', '| 方法 | 验证NMSE | teacher一次swap | 无teacher的4D V proxy一次swap |','|---|---:|---:|---:|']
        for m,v in s['scores'].items():lines.append(f"| {m} | {v['nmse']:.6f} | {v['oracle']:.6f} | {v['proxy']:.6f} |")
        lines+=['','teacher swap只表示诊断潜力，不能部署，也不计作方法收益。V-aware训练目标仅训练时读取teacher mass/V，在线不访问dense teacher。确认使用现有四prompt的新seed808/909，不是全新prompt。','']
    rows=[];methods=[];complete=False
    folders=sorted((ROOT/'acceptance').glob('*_s*'))
    for folder in folders:
        if not (folder/'identity.json').exists():continue
        ident=read(folder/'identity.json');methods=ident['methods']
        for m in methods:
            results=[read(folder/f'{m}.rep{r}.json') for r in range(3) if (folder/f'{m}.rep{r}.json').exists()]
            if not results:continue
            ts=[r['total_generation_vae_seconds'] for r in results];q=read(folder/f'{m}.quality.json')
            rows.append(dict(condition=folder.name,method=m,n=len(results),total_s=statistics.median(ts),min_s=min(ts),max_s=max(ts),
                generation_s=statistics.median(r['generation_seconds'] for r in results),peak_gib=max(r['peak_allocated_bytes'] for r in results)/2**30,
                cache_bytes=max(r['persistent_self_kv_and_aux_bytes'] for r in results),lpips=q['lpips'],tail_lpips=q['tail_lpips'],dark_dip_pp=q['dark_dip_max_pp'],
                all_clean=all(r['telemetry']['no_observed_foreign_processes'] for r in results)))
        visuals(folder,'pilot_swap' in methods)
    units=sum(r['n'] for r in rows);expected=8*3*len(methods) if methods else None;complete=bool(expected and units==expected)
    lines += [f'真实30秒含VAE计时已提交 {units}/'+(str(expected) if expected else '待冻结方法数')+' 个单元。','']
    if rows:
        lines+=['| 场景/seed | 方法 | n | 含VAE中位秒 [min,max] | LPIPS | 尾5秒LPIPS | 全部采样未见外部GPU进程 |','|---|---|---:|---:|---:|---:|---|']
        for r in rows:lines.append(f"| {r['condition']} | {r['method']} | {r['n']} | {r['total_s']:.2f} [{r['min_s']:.2f},{r['max_s']:.2f}] | {r['lpips']:.3f} | {r['tail_lpips']:.3f} | {r['all_clean']} |")
    lookup={(r['condition'],r['method']):r for r in rows};gates={}
    for method in methods:
        if method in ['dense','dense_host']:continue
        rr=[r for r in rows if r['method']==method];ratios=[];valid=len(rr)==8
        fidelity_fail=[r['condition'] for r in rr if r['lpips']>.10 or r['tail_lpips']>.15]
        for r in rr:
            base=lookup.get((r['condition'],'dense_host'))
            if base:
                ratios.append(base['total_s']/r['total_s']);valid &= r['n']==3 and base['n']==3 and r['all_clean'] and base['all_clean'] and r['max_s']/r['min_s']<=1.15 and base['max_s']/base['min_s']<=1.15
            else:valid=False
        gm=math.exp(statistics.mean(math.log(x) for x in ratios)) if len(ratios)==8 else None
        gates[method]=dict(complete=len(rr)==8 and all(r['n']==3 for r in rr),timing_valid=valid,geomean_speedup_vs_host_dense=gm,
            speed_numerical_gate=bool(gm and gm>=1.05 and all(x>1 for x in ratios)),reference_fidelity_failed=fidelity_fail,visual_review='pending',joint_pass=False)
    lines+=['','工程门槛：八个场景/seed条件均更快、含VAE几何平均≥1.05；三次最大/最小≤1.15且采样未见外部GPU进程，否则速度不可确认。每条件LPIPS≤0.10、尾5秒≤0.15，并另审查可见质量。这些是预先设定的工程阈值，不是用户指定或通用画质标准。','']
    for m,g in gates.items():lines.append(f"- {m}：参考保真失败 {len(g['reference_fidelity_failed'])} 个条件；速度可确认={g['timing_valid']}；几何平均速度比={g['geomean_speedup_vs_host_dense']}；视觉审核待完成，未宣布联合通过。")
    lines+=['','同常驻self-KV+辅助上限5,096,079,360字节。相同338历史页、选择72历史页+72当前页；native/host dense保留360历史页。动态workspace和整模型峰值另计，不声称峰值相等。计时包含全部生成、clean、缓存维护、路由、attention、VAE及RGB回传；指标/压制/磁盘I/O在计时外。','',
        '质量使用压制前RGB。工程方法和dense_host只有在完整latent逐元素一致后才复用对应视频文件；它们的计时仍为独立完整运行。不得把复制视频当作另一次生成。','', '[协议与恢复](PROTOCOL.zh.md) · [机器可读结果](summary.json)','']
    for f in folders:
        if (f/'comparison.mp4').exists():lines.append(f'- {f.name}：[并排视频](acceptance/{f.name}/comparison.mp4) · [接触图](acceptance/{f.name}/contact_sheet.png)')
    write(ROOT/'report.zh.md','\n'.join(lines)+'\n');write(ROOT/'summary.json',json.dumps(dict(units=units,expected_units=expected,complete=complete,rows=rows,gates=gates),indent=2))
    return complete
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        done=report()
        if done or not a.watch:break
        if (ROOT/'driver_status.json').exists() and read(ROOT/'driver_status.json').get('stage') in ['failed','finished']:break
        time.sleep(30)
