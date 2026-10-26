"""Guarded: preserve raw measurements, source checks, videos, and a concise report."""
import collections,csv,hashlib,json,statistics,subprocess
from pathlib import Path
import numpy as np
from .optimization_report import attribution

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csv_write(p,rows):
    with p.open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    base=Path(__file__).resolve().parents[2];root=base/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024';out=root/'optimization6'
    seal=read(out/'guard_manifest.json');digests={**seal['code'],**seal['prior_unchanged']}
    for file,d in digests.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==d,file
    assert read(out/'guard_status.json')['status']=='completed'
    speed={};samples=[];hashes={}
    for folder in sorted((out/'guard_benchmark').iterdir()):
        records=read(folder/'timings.json')['records'];rows=records[1:]
        assert len(rows)==3 and len({r['latent_sha256']for r in records})==1
        hashes[folder.name]=rows[0]['latent_sha256']
        speed[folder.name]=dict(variant=folder.name,n=3,generation_s=statistics.mean(r['generation_seconds']for r in rows),
            generation_std_s=statistics.stdev(r['generation_seconds']for r in rows),decode_s=statistics.mean(r['decode_seconds']for r in rows),
            total_s=statistics.mean(r['total_seconds']for r in rows),peak_allocated_GiB=max(r['peak_allocated_GiB']for r in rows))
        samples.extend(dict(variant=folder.name,**r)for r in rows)
    assert len(samples)==24 and len({r['first_chunk_sha256']for r in samples})==1
    for name in ['Opt5-K32','Guarded-K32','Guarded-V32']:assert hashes[name]==hashes[name+'-replica']
    import torch
    pid=read(root/'e2e_manifest.json')['prompt_ids'][0]
    for kind in ['K32','V32']:
        timed=torch.load(out/f'guard_benchmark/Guarded-{kind}/output.pt',map_location='cpu',weights_only=True)['latent']
        validated=torch.load(out/f'guard_quality/{pid}.seed0/Guarded-{kind}.latent.pt',map_location='cpu',weights_only=True)
        assert torch.equal(timed,validated),('timing/quality mismatch',kind)
        before=torch.load(out/f'guard_benchmark/Opt5-{kind}/output.pt',map_location='cpu',weights_only=True)['latent']
        saved=torch.load(root/f'optimization5/quality/{pid}.seed0/Opt5-{kind}.latent.pt',map_location='cpu',weights_only=True)
        assert torch.equal(before,saved),('accepted baseline mismatch',kind)
    boundary=read(out/'guard_boundary_checks.json');assert boundary['status']=='passed'and len(boundary['cases'])==7
    quality=[];audit=[];cases=sorted((out/'guard_quality').glob('*/result.json'));assert len(cases)==8
    for file in cases:
        doc=read(file)
        for kind in ['K32','V32']:
            old=read(root/'optimization5/quality'/file.parent.name/f'Opt5-{kind}.json')
            new=doc['variants']['Guarded-'+kind];assert new['first_chunk_identical']
            for variant,row in [('Opt5',old),('Guarded',new)]:
                met=row['quality_vs_dense']['sparse_frames_only'];prior=old['quality_vs_dense']['sparse_frames_only']
                quality.append(dict(prompt=file.parent.name,variant=variant,mode=kind,lpips=met['lpips'],psnr_db=met['psnr_db'],
                    weighted_keep=row['weighted_keep'],delta_lpips_vs_opt5=met['lpips']-prior['lpips'],
                    lpips_to_opt5=new['quality_vs_opt5']['sparse_frames_only']['lpips']if variant=='Guarded'else 0.))
            audit.extend(dict(prompt=file.parent.name,mode=kind,**r)for r in new['attention_audit'])
    assert all(np.isfinite(r[k])for r in quality for k in ['lpips','psnr_db','weighted_keep'])
    rng=np.random.default_rng(20261024);qs={}
    for variant in ['Opt5','Guarded']:
        for kind in ['K32','V32']:
            rr=[r for r in quality if(r['variant'],r['mode'])==(variant,kind)];assert len(rr)==8
            delta=np.array([r['delta_lpips_vs_opt5']for r in rr]);ci=np.quantile(delta[rng.integers(0,8,(10000,8))].mean(1),[.025,.975])
            worst=max(rr,key=lambda r:r['delta_lpips_vs_opt5'])
            qs[variant+'-'+kind]=dict(lpips=statistics.mean(r['lpips']for r in rr),psnr_db=statistics.mean(r['psnr_db']for r in rr),
                keep=statistics.mean(r['weighted_keep']for r in rr),delta_lpips=float(delta.mean()),delta_ci95=list(ci),
                lpips_to_opt5=statistics.mean(r['lpips_to_opt5']for r in rr),worst_prompt=worst['prompt'],worst_delta=worst['delta_lpips_vs_opt5'])
    assert len(audit)==144
    aa=dict(blocks=len(audit),queries=sum(r['queries']for r in audit),changed_queries=sum(r['changed_queries']for r in audit),
        delta_nmse=sum(r['delta_sse']for r in audit)/sum(r['reference_energy']for r in audit),
        nmse_to_dense=sum(r['sse_to_dense']for r in audit)/sum(r['dense_energy']for r in audit),
        reference_nmse_to_dense=sum(r['reference_sse_to_dense']for r in audit)/sum(r['dense_energy']for r in audit))
    assert read(out/'guard_profile/modules.json')['observer_latents_bit_identical']
    profiles={'Opt5':read(root/'optimization5/profile/kernel_attribution.json'),'Guarded':attribution(out/'guard_profile')}
    modules={k:{r['module']:r['kernel_ms']for r in p['modules']}for k,p in profiles.items()}
    kernels=collections.defaultdict(float)
    for r in profiles['Guarded']['kernels']:
        if r['module']=='route_graph_copy_and_replay':kernels[r['kernel']]+=r['milliseconds']
    checks=read(out/'checks.json');assert checks['finite_nonempty']and len(checks['rows'])==18
    correct=dict(queries=sum(r['queries']for r in checks['rows']),changed_queries=sum(r['changed_queries']for r in checks['rows']),
        delta_nmse=sum(r['attention_delta_sse']for r in checks['rows'])/sum(r['reference_energy']for r in checks['rows']))
    probes=[]
    for f in (out/'probes').glob('*.json'):
        if f.name.endswith('.status.json'):continue
        r=read(f);probes.append(dict(name=r['name'],reference_route_ms=r['timings_ms']['reference_route'],route_ms=r['timings_ms']['route'],reference_head_ms=r['timings_ms']['reference_head'],head_ms=r['timings_ms']['head'],
            changed_queries=r['changed_queries'],attention_nmse=r['attention_delta_nmse']))
    csv_write(out/'final_microbenchmarks.csv',probes);csv_write(out/'final_timing_samples.csv',samples);csv_write(out/'final_speed_summary.csv',list(speed.values()))
    csv_write(out/'final_quality_cases.csv',quality);csv_write(out/'final_attention_audit.csv',audit)
    summary=dict(selected_entry='migration.unified_router.opt6_guarded.load_guarded_head',options=seal['options'],speed=speed,quality=qs,
        actual_attention_audit=aa,correctness=correct,profiles={k:dict(total_kernel_ms=p['total_kernel_ms'],modules=p['modules'])for k,p in profiles.items()},
        baseline='Opt5 accepted FP16 route',retrained=False,test_used=False,source_hashes_verified=len(digests),boundary_checks=7,timing_quality_latents_identical=True)
    summary['latent_identical_cases']=sum(read(f)['latent_bit_identical']for f in(out/'guard_quality').glob('*.seed0/Guarded-*.json'))
    summary['rgb_identical_cases']=sum(read(f)['quality_vs_opt5']['full']['identical']for f in(out/'guard_quality').glob('*.seed0/Guarded-*.json'))
    summary['head_prediction_max_nmse']=max(r['head_prediction_nmse']for r in checks['rows'])
    summary['guard_fallback_fraction']=sum(r['guard_fallback_queries']for r in audit)/sum(r['guard_checked_queries']for r in audit)
    summary['head_prediction_max_abs']=max(r['head_prediction_max_abs']for r in checks['rows'])
    write(out/'final_summary.json',summary);write(out/'final_checks.json',dict(status='passed',source_hashes=len(digests),first_chunk_identical=True,
        timed_repeats_deterministic=True,cross_gpu_replicas_identical=True,profile_latents_identical=True,boundary_checks=7,timing_quality_latents_identical=True,accepted_baseline_reproduced=True,quality_cases=16,actual_attention_blocks=144))
    raw=read(out/'summary.json')
    lines=['# SlimAV Opt6：动作head分解与临界评分回退','',
        '**最终推荐入口：`migration.unified_router.opt6_guarded.load_guarded_head`。以已接受的Opt5为基线，保留24维特征与81动作；通常走快速分解head，临界评分用原Opt5 head复算。模型权重、预算不变，未重训。**','',
        '## 最终端到端速度','',
        '单张A100-SXM4-40GB，45帧480×832、16fps，4 chunks×4去噪步。同一Val prompt/seed0，预热一次后测三次；新K32/V32均有跨GPU replica。表列主GPU均值，逐次数据另存CSV。生成含上下文重建；总计含VAE与RGB回传，加载/编译/LPIPS/MP4编码不计。最终回退版独立重新测速，质量评估与profile在测速结束后执行。','',
        '|方案|生成 / s|VAE及回传 / s|总计 / s|峰值allocated / GiB|','|---|---:|---:|---:|---:|']
    for name in ['Dense','Opt5-K32','Guarded-K32','Opt5-V32','Guarded-V32']:
        r=speed[name];label=name.replace('Guarded','Opt6最终');lines.append(f"|{label}|{r['generation_s']:.3f}|{r['decode_s']:.3f}|{r['total_s']:.3f}|{r['peak_allocated_GiB']:.2f}|")
    for kind in ['K32','V32']:
        old=speed['Opt5-'+kind];new=speed['Guarded-'+kind]
        lines+=['',f"{kind}生成耗时减少{1-new['generation_s']/old['generation_s']:.2%}，速度{old['generation_s']/new['generation_s']:.3f}×；总时长减少{1-new['total_s']/old['total_s']:.2%}。"]
    lines+=['','仍慢于Dense；head只占全模型的一部分，打分、选候选、均值、attention、主干和VAE仍需要执行。','',
        '## 怎样减少重复计算','',
        '原head对每query的81个动作（GPU补齐128行）反复计算候选几何量，构造24维特征，再做24→16→1 MLP。24维特征可分成11维公共项、6维删除项、6维添加项、1维交互点积：','',
        '`y(i,j) = shared(query) + remove(i) + add(j) + cross(i,j) × W1[:,5]`','',
        '第一阶段各计算一次16个候选的范数/点积等几何量，用小矩阵计算删除和添加候选的16维贡献；公共项只计算一次。第二阶段组合原81动作，执行原erf GELU、末层、合法性和最小编号破同分规则，更新逐query mask。','',
        '保持原归一化和FP16取整位置；特征/第一层权重乘法沿用FP16输入、FP32累加；交互点积用局部TF32x3。主干仍BF16，全局PyTorch TF32关闭。变化主要来自求和顺序和交互点积归约，不改变动作空间。',
        '理想第一层标量乘法约从81×24×16=31,104次变为11×16+2×9×6×16+81×16=3,200次。但GPU补齐、GELU、交互计算、启动与访存仍存在，不能据此声称约10倍实际加速。',
        '每query增加384个FP32临时槽，512-query×12heads约9.4MB，整层累计约86.3MB。旧head特征本就在寄存器中，本轮收益是减少重复工作及改善执行布局，不是消除了某个旧HBM特征张量。','',
        '## 为什么最终加临界回退','',
        '无回退版head微基准约0.154→0.090ms；整机K32/V32约6.438/6.516秒，较同期Opt5减少约3%。然而完整视频发现V32海豹场景LPIPS对Dense增加0.02168。虽然同输入实际attention增量NMSE仅约4.43e-10，仍不能据此推断视频无变化。这个未保护版本保留为消融，详见[原始报告](UNGUARDED_REPORT.md)。','',
        '最终先比较合法动作（含得分固定0的不动作）的第一名和第二名；若 `gap ≤ 0.001 × (1 + |best_score|)`，由独立GPU kernel调用原Opt5的完整head公式复算此query，覆盖快速输出。没有CPU读回或同步分支；无需回退的CTA直接返回。','',
        f"代表512-query批次约3.9%回退，head约0.109ms；完整视频抽查的回退比例为{summary['guard_fallback_fraction']:.3%}。保护阈值是保守的经验规则，不是对所有可能输入的形式化等价证明；最终采用依据是下面的整视频和真实attention验证。",'',
        '完全融合的单kernel已试过：9×9动作补齐16×16，GELU和寄存器压力更高，慢于两阶段。FP16交互点积也试过，与TF32x3差距很小，最终保留TF32x3。原16组head调参和临界回退微基准均存档。','',
        '## 最终视频与误差','',
        f"8个注册Val场景×K32/V32共16组，最终有 **{summary['latent_identical_cases']}/16** 组完整latent与Opt5逐bit一致，RGB逐bit一致{summary['rgb_identical_cases']}/16组。首chunk均与Dense一致。所有计时重复、跨GPU replica一致；计时latent与质量评估latent一致，且本轮Opt5复现了上一轮已接受输出。",'',
        'LPIPS/PSNR用未压缩VAE输出的后36帧；95%区间按8个prompt bootstrap10000次。LPIPS是相对Dense的感知差异，并非绝对画质评分。','',
        '|方案|LPIPS对Dense↓|相对Opt5变化|95% prompt bootstrap区间|保留率|','|---|---:|---:|---|---:|']
    for kind in ['K32','V32']:
        for version in ['Opt5','Guarded']:
            r=qs[version+'-'+kind];ci=r['delta_ci95'];label=version.replace('Guarded','Opt6最终')
            lines.append(f"|{label}-{kind}|{r['lpips']:.6f}|{r['delta_lpips']:+.6f}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|{r['keep']:.3%}|")
    lines+=['',f"同一Q/K/V抽查layer1/24/29、三个稀疏chunk、step3，共{aa['blocks']}个block、{aa['queries']}个head-query：最终mask变化{aa['changed_queries']}个；实际attention增量NMSE **{aa['delta_nmse']:.3e}**。新输出NMSE对Dense为{aa['nmse_to_dense']:.9f}，对应Opt5为{aa['reference_nmse_to_dense']:.9f}。",'',
        f"基础分解版18组跨层/历史/尾块检查覆盖{correct['queries']}个head-query，mask全部一致，原始81动作预测的最大case NMSE {summary['head_prediction_max_nmse']:.3e}。最终回退版另通过7组动作边界检查：无候选、只删、只加、单页、全选、零head和同分正收益动作。",'',
        '## 模块profile','',
        'layer24、第4chunk、step3，profile观察器已验证不改变latent。按CUDA API correlation唯一归属kernel，表为GPU kernel累计时间。Opt5列取上轮同位置profile；端到端表是本轮重新测量。','',
        '|模块 / ms|Opt5|Opt6最终|','|---|---:|---:|']
    for key,label in [('route_graph_copy_and_replay','路由及辅助数据'),('sparse_attention_including_union','稀疏attention及并集'),('ffn','FFN'),('cross_attention','交叉attention'),('current_prototypes','当前原型'),('prototype_projection','原型投影')]:
        lines.append('|'+label+'|'+'|'.join(f'{modules[n][key]:.3f}'for n in ['Opt5','Guarded'])+'|')
    lines+=['|整层合计|'+'|'.join(f"{profiles[n]['total_kernel_ms']:.3f}"for n in ['Opt5','Guarded'])+'|','','|路由kernel|累计 ms|','|---|---:|']
    for k,v in sorted(kernels.items(),key=lambda r:-r[1]):
        if len(k)<80:lines.append(f'|{k}|{v:.3f}|')
    oldhead=sum(r['milliseconds']for r in profiles['Opt5']['kernels']if r['kernel']=='_candidate_actions')
    newhead=sum(kernels[k]for k in ['_factor_prepare','_guard_finish','_head_fallback'])
    summary['head_profile']=dict(reference_ms=oldhead,new_ms=newhead,reduction=1-newhead/oldhead)
    lines+=['',f"含回退的完整层head：{oldhead:.3f}→{newhead:.3f}ms，减少{1-newhead/oldhead:.2%}。",'',
        '下一步主要热点是页打分/energy、均值聚合和候选选择。可以继续做数据复用与归约融合；若减少候选或改用更便宜的内容统计，应重新生成对应特征/动作标签并训练小head，loss及best仍按真实硬动作AV NMSE。本轮没有重训。','',
        '## 结果与入口','',
        '- [K32全部8场景](ALL8_Dense_Opt5_Guarded_K32.mp4)、[V32全部8场景](ALL8_Dense_Opt5_Guarded_V32.mp4)：左Dense、中Opt5、右Opt6最终（Guarded）；360帧，22.5秒。',
        '- [最终速度/质量/profile图](final_speed_quality_profile.png)、[抽帧对比](final_worst_case_frames.jpg)、[最终汇总](final_summary.json)。',
        '- [最终逐次计时](final_timing_samples.csv)、[逐场景质量](final_quality_cases.csv)、[实际attention审计](final_attention_audit.csv)。',
        '- [最终模块profile](guard_profile/modules.csv)、[kernel明细](guard_profile/kernels.csv)、[trace](guard_profile/trace.json.gz)。',
        '- [冻结记录](guard_manifest.json)、[最终核验](final_verification.json)、[无回退消融](UNGUARDED_REPORT.md)。','',
        '```python','from migration.unified_router.opt6_guarded import load_guarded_head','net, config = load_guarded_head(experiment_directory)','net.set_mode("V32-Set")  # 或 K32-Set','```','',
        '状态：`./v10_optimization6_progress.sh --once`。最终kernel与入口为`migration/unified_router/opt6_guarded.py`，复用`opt6/factor.py`第一阶段。原Opt5与未保护Opt6均保留。适用当前batch1、12heads、128维、4680-token chunk、64-token页、顺序CUDA Graph推理；首chunk/layer0 Dense，后续首chunk为普通历史，当前P16、历史P8/P4/P2，约15%预算。Test未使用。',
        '实现参考[Triton dot精度说明](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)；所有新kernel为本地编写。']
    if not(out/'UNGUARDED_REPORT.md').exists():(out/'UNGUARDED_REPORT.md').write_bytes((out/'REPORT.md').read_bytes())
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'final_summary.json',summary)
    # Export plots as standalone artifacts.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    names=['Dense','Opt5-K32','Guarded-K32','Opt5-V32','Guarded-V32'];x=np.arange(len(names))
    axes[0].bar(x,[speed[n]['generation_s']for n in names],label='Generation')
    axes[0].bar(x,[speed[n]['decode_s']for n in names],bottom=[speed[n]['generation_s']for n in names],label='VAE + transfer')
    axes[0].set_xticks(x,names,rotation=35,ha='right');axes[0].set_ylabel('Seconds');axes[0].set_title('Matched end-to-end timing');axes[0].legend(fontsize=8)
    for i,kind in enumerate(['K32','V32']):
        rr=[r for r in quality if r['variant']=='Guarded'and r['mode']==kind]
        axes[1].scatter([i]*8,[r['delta_lpips_vs_opt5']for r in rr],alpha=.7)
        ci=qs['Guarded-'+kind]['delta_ci95'];mean=qs['Guarded-'+kind]['delta_lpips']
        axes[1].errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],color='black',fmt='s',capsize=5)
    if summary['rgb_identical_cases']==16:axes[1].text(.5,.95,'16/16 RGB bit-identical',ha='center',va='top',transform=axes[1].transAxes)
    axes[1].axhline(0,color='gray',lw=1);axes[1].set_xticks([0,1],['K32','V32']);axes[1].set_ylabel('Delta LPIPS to Dense vs Opt5');axes[1].set_title('8 paired Val prompts; lower is better')
    modules['Opt5']['head']=oldhead;modules['Guarded']['head']=newhead
    keys=['route_graph_copy_and_replay','head','sparse_attention_including_union'];labels=['Route total','Action head','Sparse attention'];x=np.arange(3)
    for i,n in enumerate(['Opt5','Guarded']):axes[2].bar(x+(i-.5)*.34,[modules[n][k]for k in keys],width=.34,label=n)
    axes[2].set_xticks(x,labels,rotation=20);axes[2].set_ylabel('Kernel milliseconds');axes[2].set_title('Layer24 / chunk4 / step3');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'final_speed_quality_profile.png',dpi=180);fig.savefig(out/'final_speed_quality_profile.pdf');plt.close(fig)
    # Concatenate every prompt in registry order, retain native three-column size.
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();pids=read(root/'e2e_manifest.json')['prompt_ids']
    for kind in ['K32','V32']:
        playlist=out/f'concat_{kind}.txt';playlist.write_text(''.join(f"file '{out/'guard_quality'/f'{pid}.seed0'/f'Dense_Opt5_Guarded_{kind}.mp4'}'\n"for pid in pids))
        subprocess.run([ffmpeg,'-y','-v','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy',str(out/f'ALL8_Dense_Opt5_Guarded_{kind}.mp4')],check=True)
    video_metadata={}
    import imageio.v2 as imageio
    for kind in ['K32','V32']:
        path=out/f'ALL8_Dense_Opt5_Guarded_{kind}.mp4';reader=imageio.get_reader(path);meta=reader.get_meta_data();frames=reader.count_frames();reader.close()
        assert frames==360 and meta['fps']==16 and tuple(meta['size'])==(2496,480),(kind,frames,meta)
        video_metadata[kind]=dict(frames=frames,fps=meta['fps'],size=meta['size'],duration=meta['duration'],bytes=path.stat().st_size)
    write(out/'final_video_metadata.json',video_metadata)
    # Inspect the highest measured per-prompt degradation for each head.
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    panels=[]
    for kind in ['K32','V32']:
        prompt=qs['Guarded-'+kind]['worst_prompt'];reader=imageio.get_reader(out/'guard_quality'/prompt/f'Dense_Opt5_Guarded_{kind}.mp4')
        im=Image.fromarray(reader.get_data(40));reader.close();im.thumbnail((1248,240))
        panel=Image.new('RGB',(1248,268),'white');panel.paste(im,(0,28));ImageDraw.Draw(panel).text((8,8),kind+' / '+prompt+' / frame40',fill='black');panels.append(panel)
    canvas=Image.new('RGB',(1248,536),'white')
    for i,im in enumerate(panels):canvas.paste(im,(0,i*268))
    canvas.save(out/'final_worst_case_frames.jpg',quality=92)
    write(out/'final_verification.json',dict(read(out/'final_checks.json'),latent_identical_cases=summary['latent_identical_cases'],rgb_identical_cases=summary['rgb_identical_cases'],guard_fallback_fraction=summary['guard_fallback_fraction'],video_metadata_verified=True))
    write(out/'final_analysis_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__).resolve(),Path(__file__).with_name('opt6_final_status.py').resolve(),Path(__file__).with_name('opt6_guard_boundary.py').resolve(),base/'v10_optimization6_progress.sh']})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
