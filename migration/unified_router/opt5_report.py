"""Opt5: preserve raw measurements, source checks, videos, and a concise report."""
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
    base=Path(__file__).resolve().parents[2];root=base/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024';out=root/'optimization5'
    seal=read(out/'manifest.json');digests={**seal['code'],**seal['prior_unchanged']}
    for file,d in digests.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==d,file
    assert read(out/'status.json')['status']=='completed'
    speed={};samples=[];hashes={}
    for folder in sorted((out/'benchmark').iterdir()):
        records=read(folder/'timings.json')['records'];rows=records[1:]
        assert len(rows)==3 and len({r['latent_sha256']for r in records})==1
        hashes[folder.name]=rows[0]['latent_sha256']
        speed[folder.name]=dict(variant=folder.name,n=3,generation_s=statistics.mean(r['generation_seconds']for r in rows),
            generation_std_s=statistics.stdev(r['generation_seconds']for r in rows),decode_s=statistics.mean(r['decode_seconds']for r in rows),
            total_s=statistics.mean(r['total_seconds']for r in rows),peak_allocated_GiB=max(r['peak_allocated_GiB']for r in rows))
        samples.extend(dict(variant=folder.name,**r)for r in rows)
    assert len(samples)==24 and len({r['first_chunk_sha256']for r in samples})==1
    for name in ['Opt4-K32','Opt5-K32','Opt5-V32']:assert hashes[name]==hashes[name+'-replica']
    import torch
    pid=read(root/'e2e_manifest.json')['prompt_ids'][0]
    for kind in ['K32','V32']:
        timed=torch.load(out/f'benchmark/Opt5-{kind}/output.pt',map_location='cpu',weights_only=True)['latent']
        validated=torch.load(out/f'quality/{pid}.seed0/Opt5-{kind}.latent.pt',map_location='cpu',weights_only=True)
        assert torch.equal(timed,validated),('timing/quality mismatch',kind)
        before=torch.load(out/f'benchmark/Opt4-{kind}/output.pt',map_location='cpu',weights_only=True)['latent']
        saved=torch.load(root/f'optimization4/quality/{pid}.seed0/FP16-{kind}.latent.pt',map_location='cpu',weights_only=True)
        assert torch.equal(before,saved),('accepted baseline mismatch',kind)
    boundary=read(out/'boundary_checks.json');assert boundary['status']=='passed'and len(boundary['cases'])==6
    quality=[];audit=[];cases=sorted((out/'quality').glob('*/result.json'));assert len(cases)==8
    for file in cases:
        doc=read(file)
        for kind in ['K32','V32']:
            old=read(root/'optimization4/quality'/file.parent.name/f'FP16-{kind}.json')
            new=doc['variants']['Opt5-'+kind];assert new['first_chunk_identical']
            for variant,row in [('Opt4',old),('Opt5',new)]:
                met=row['quality_vs_dense']['sparse_frames_only'];prior=old['quality_vs_dense']['sparse_frames_only']
                quality.append(dict(prompt=file.parent.name,variant=variant,mode=kind,lpips=met['lpips'],psnr_db=met['psnr_db'],
                    weighted_keep=row['weighted_keep'],delta_lpips_vs_opt4=met['lpips']-prior['lpips'],
                    lpips_to_opt4=new['quality_vs_opt4_fp16']['sparse_frames_only']['lpips']if variant=='Opt5'else 0.))
            audit.extend(dict(prompt=file.parent.name,mode=kind,**r)for r in new['attention_audit'])
    assert all(np.isfinite(r[k])for r in quality for k in ['lpips','psnr_db','weighted_keep'])
    rng=np.random.default_rng(20261024);qs={}
    for variant in ['Opt4','Opt5']:
        for kind in ['K32','V32']:
            rr=[r for r in quality if(r['variant'],r['mode'])==(variant,kind)];assert len(rr)==8
            delta=np.array([r['delta_lpips_vs_opt4']for r in rr]);ci=np.quantile(delta[rng.integers(0,8,(10000,8))].mean(1),[.025,.975])
            worst=max(rr,key=lambda r:r['delta_lpips_vs_opt4'])
            qs[variant+'-'+kind]=dict(lpips=statistics.mean(r['lpips']for r in rr),psnr_db=statistics.mean(r['psnr_db']for r in rr),
                keep=statistics.mean(r['weighted_keep']for r in rr),delta_lpips=float(delta.mean()),delta_ci95=list(ci),
                lpips_to_opt4=statistics.mean(r['lpips_to_opt4']for r in rr),worst_prompt=worst['prompt'],worst_delta=worst['delta_lpips_vs_opt4'])
    assert len(audit)==144
    aa=dict(blocks=len(audit),queries=sum(r['queries']for r in audit),changed_queries=sum(r['changed_queries']for r in audit),
        delta_nmse=sum(r['delta_sse']for r in audit)/sum(r['reference_energy']for r in audit),
        nmse_to_dense=sum(r['sse_to_dense']for r in audit)/sum(r['dense_energy']for r in audit),
        reference_nmse_to_dense=sum(r['reference_sse_to_dense']for r in audit)/sum(r['dense_energy']for r in audit))
    assert read(out/'profile/modules.json')['observer_latents_bit_identical']
    profiles={'Opt4':read(root/'optimization4/profile_fused_fp16/kernel_attribution.json'),'Opt5':attribution(out/'profile')}
    modules={k:{r['module']:r['kernel_ms']for r in p['modules']}for k,p in profiles.items()}
    kernels=collections.defaultdict(float)
    for r in profiles['Opt5']['kernels']:
        if r['module']=='route_graph_copy_and_replay':kernels[r['kernel']]+=r['milliseconds']
    checks=read(out/'checks.json');assert checks['finite_nonempty']and len(checks['rows'])==18
    correct=dict(queries=sum(r['queries']for r in checks['rows']),changed_queries=sum(r['changed_queries']for r in checks['rows']),
        delta_nmse=sum(r['attention_delta_sse']for r in checks['rows'])/sum(r['reference_energy']for r in checks['rows']))
    probes=[]
    for f in (out/'probes').glob('*.json'):
        if f.name.endswith('.status.json'):continue
        r=read(f);probes.append(dict(name=r['name'],reference_route_ms=r['reference_route_ms'],route_ms=r['route_ms'],prepare_ms=r['prepare_ms'],
            changed_queries=r['changed_queries'],attention_nmse=r['attention_nmse_vs_reference']))
    csv_write(out/'microbenchmarks.csv',probes);csv_write(out/'timing_samples.csv',samples);csv_write(out/'speed_summary.csv',list(speed.values()))
    csv_write(out/'quality_cases.csv',quality);csv_write(out/'attention_audit.csv',audit)
    summary=dict(selected_entry='migration.unified_router.opt5.network.load_flash',options=seal['options'],speed=speed,quality=qs,
        actual_attention_audit=aa,correctness=correct,profiles={k:dict(total_kernel_ms=p['total_kernel_ms'],modules=p['modules'])for k,p in profiles.items()},
        baseline='Opt4 final fused FP16 explicitly accepted by user',retrained=False,test_used=False,source_hashes_verified=len(digests),boundary_checks=6,timing_quality_latents_identical=True)
    write(out/'summary.json',summary);write(out/'verification.json',dict(status='passed',source_hashes=len(digests),first_chunk_identical=True,
        timed_repeats_deterministic=True,cross_gpu_replicas_identical=True,profile_latents_identical=True,boundary_checks=6,timing_quality_latents_identical=True,accepted_baseline_reproduced=True,quality_cases=16,actual_attention_blocks=144))
    lines=['# SlimAV Opt5：消除 logits，保留可复用的 FP16 权重','',
        '**本轮比较基线是已接受的 Opt4 最终 FP16。完整两遍重算和权重复用都已实现、实测；最终采用权重复用。每个 query 独立路由，预算、原型数量与训练权重不变，未重训。**','',
        '## 端到端结果','',
        '单张 A100-SXM4-40GB；45帧、480×832、16fps；4 chunks×4去噪步。每个配置完整预热一次后测三次。K32/V32新版本各有跨GPU replica，表中列主GPU结果；全部逐次记录见CSV。计时包括上下文重建，总计另含VAE和RGB回传；加载、编译、LPIPS与MP4编码不计。测速全部结束后才运行质量评估。','',
        '|方案|生成 / s|VAE及回传 / s|总计 / s|峰值 allocated / GiB|','|---|---:|---:|---:|---:|']
    for name in ['Dense','Opt4-K32','Opt5-K32','Opt4-V32','Opt5-V32']:
        r=speed[name];lines.append(f"|{name}|{r['generation_s']:.3f}|{r['decode_s']:.3f}|{r['total_s']:.3f}|{r['peak_allocated_GiB']:.2f}|")
    for kind in ['K32','V32']:
        a=speed['Opt4-'+kind];b=speed['Opt5-'+kind]
        lines+=['',f"{kind}：生成耗时减少 {1-b['generation_s']/a['generation_s']:.2%}，生成速度 {a['generation_s']/b['generation_s']:.3f}×；含VAE总时长减少 {1-b['total_s']/a['total_s']:.2%}。"]
    lines+=['','仍慢于Dense。首chunk、layer0、上下文重建、FFN、投影、交叉注意力与VAE不受15%保留率直接缩减；路由仍扫描全部页。','',
        '## 计算过程与省掉的中间量','',
        '对每个query、每一页b，原型p的logit为 `l_bp = dot(Q投影,K_bp投影)/sqrt(128)+log(count_bp)`；`s_b=logsumexp_p(l_bp)`，`w_bp=softmax_p(l_bp)`，页向量`c_b=Σ_p w_bp V_bp投影`。K32的内容投影来自K，V32来自V。','',
        '1. 在一个GPU线程块内处理64个query、4页，每页原型补齐16个。QK使用FP16输入、FP32累加；logit、softmax、score计算保持片上，不输出完整logits。补齐仅改变执行布局，原P16/P8/P4/P2的有效原型与count不变。',
        '2. 同一内核在片上计算`c_b`，只输出标量`e_b=||c_b||²`；不再构造Gram矩阵。保存`cost_b*w_bp`的一份FP16权重、FP32页score与energy，不保存全页32维content。',
        '3. 一个融合内核读取score，在寄存器中计算均值、标准差、初始mask、候选排序距离和确定性hash，输出mask及8删+8加候选。省去全页distance/hash数组；128-query hash边界和最小索引破同分规则保留。',
        '4. 均值GEMM读取权重时应用逐query mask，直接累加全部页与已选页的两套均值。候选向量从同一FP16权重与原FP32内容原型计算，免去候选QK重算与第二次softmax。',
        '5. 用页energy与均值恢复集合方差（非负截断，单页已选方差为0），运行原81动作head，输出逐query最终mask。稀疏attention内核与Opt4相同。','',
        '`μ_S=Σ_{b∈S}cost_b*c_b / Σ_{b∈S}cost_b`；`variance_S=Σ_{b∈S}cost_b*e_b / Σ_{b∈S}cost_b − ||μ_S||²`。',
        '64只是GPU计算tile，每个query仍拥有自己的score、mask、候选和动作。这里没有Q64共享选页。','',
        '代表层12 heads×4680 queries×296页×2220原型，下表为整层累计产生的逻辑张量（十进制MB），不是同时驻留峰值或实测DRAM流量。实际按512 queries分批。','',
        '|中间量|Opt4 FP16|Opt5|','|---|---:|---:|','|完整原型logits|498.7 MB|0|','|原型权重|249.4 MB|249.4 MB|',
        '|全页32维content|0|0|','|页score / energy|各66.5 MB|各66.5 MB|','|候选距离 / hash|各66.5 MB|0|','|候选向量 / 统计|约133 MB|约133 MB|','',
        'logits与权重两项合计减少约三分之二，但仍有一份FP16权重落显存。新的kernel也引入原型补齐、布局与寄存器压力；不能把减少的逻辑张量字节数直接称为净DRAM节省或整模型显存降低。','',
        '## 为什么没采用完全两遍重算','',
        f"共保留{len(probes)}组实测配置。代表真实块的512-query批次：初始两遍版约0.916ms，原FP16约0.771ms。候选QK用补零16行的Tensor Core计算效率低；改用SIMT向量归约后，候选重算约0.23→0.07ms。调优并融合选择的两遍版约0.708ms；最终权重复用版约0.624ms（该微基准准备阶段单独计时，整模型计时包含所有准备工作）。",'',
        '完全重算确实去掉了logits和权重，但需要再次读K、重复QK和exp，且第二遍依赖页score与mask的索引读取。当前A100和这些形状下，复用小精度权重更划算。这不是证明两遍设计永远慢，只是这轮已实现和调优配置的结果。',
        '借鉴了[官方Triton fused attention教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)的片上分块与融合思路；路由有跨全页阈值屏障，不能直接套用标准attention的一遍online-softmax。新路由kernel为本地编写。','',
        '## 精度与视频质量','',
        '新增近似：候选向量复用FP16权重；页energy改为FP16 w×V、FP32累加后取范数；阈值统计采用融合FP32归约。Q投影仍FP32，主干BF16与全局TF32关闭保持原样。FP16输入精度与FP32累加不能称为全流程FP16。','',
        '8个原注册Val prompt×K32/V32，seed0，共16组。指标在未压缩RGB的后36帧计算，所有首9帧与Dense逐bit一致。LPIPS表示与Dense的感知差异，不是绝对画质；bootstrap按8个prompt抽样10000次，不把帧或query当独立质量样本。','',
        '|方案|LPIPS对Dense↓|相对Opt4变化|95% prompt bootstrap区间|保留率|','|---|---:|---:|---|---:|']
    for kind in ['K32','V32']:
        for variant in ['Opt4','Opt5']:
            r=qs[variant+'-'+kind];ci=r['delta_ci95'];lines.append(f"|{variant}-{kind}|{r['lpips']:.6f}|{r['delta_lpips']:+.6f}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|{r['keep']:.3%}|")
    lines+=['',f"跨层/历史/尾块18组检查：{correct['queries']}个head-query中{correct['changed_queries']}个mask变化，注意力增量NMSE {correct['delta_nmse']:.3e}。",'',
        f"实际生成轨迹抽查layer1/24/29、三个稀疏chunk、step3，共{aa['blocks']}个block、{aa['queries']}个head-query：{aa['changed_queries']}个query的mask变化；相对Opt4的同输入注意力增量NMSE **{aa['delta_nmse']:.3e}**。新版本实际attention NMSE对Dense为{aa['nmse_to_dense']:.8f}，相同Q/K/V上的Opt4为{aa['reference_nmse_to_dense']:.8f}。这组误差使用真实attention输出，不是特征拟合误差。",'',
        '额外6组选择边界检查（单页、全同分、单峰、历史/当前同分及随机输入，覆盖17/72/128/512-query形状）全部与参考mask、删除候选、添加候选逐bit相同。','',
        '局部误差小也可能经多层多步放大为视频轨迹差异，不能声称输出无损。']
    for kind in ['K32','V32']:
        r=qs['Opt5-'+kind];lines.append(f"{kind}逐场景最大LPIPS退化为{r['worst_delta']:+.6f}，场景{r['worst_prompt']}。")
    lines+=['','## 模块profile与后续热点','',
        'layer24、第4chunk、step3；单独profile，不污染计时；观察器已验证不改变latent。每个GPU kernel按runtime/driver correlation唯一归属，CUDA Graph节点归到graph launch，表为kernel累计时间。Opt4列使用上一轮同位置profile，端到端速度表是本轮重新测量。','',
        '|模块 / ms|Opt4 FP16|Opt5|','|---|---:|---:|']
    for key,label in [('route_graph_copy_and_replay','路由及辅助数据'),('sparse_attention_including_union','稀疏attention与页并集'),('ffn','FFN'),('cross_attention','交叉attention'),('current_prototypes','当前原型'),('prototype_projection','原型投影')]:
        lines.append('|'+label+'|'+ '|'.join(f'{modules[n][key]:.3f}'for n in ['Opt4','Opt5'])+'|')
    lines.append('|整层合计|'+'|'.join(f"{profiles[n]['total_kernel_ms']:.3f}"for n in ['Opt4','Opt5'])+'|')
    lines+=['','路由主要kernel：','','|kernel|累计 ms|','|---|---:|']
    for k,v in sorted(kernels.items(),key=lambda r:-r[1]):
        if len(k)<80:lines.append(f"|{'FP32 Q投影 GEMM'if k=='ampere_sgemm_128x128_tn'else k}|{v:.3f}|")
    lines+=['','后续可继续优化候选/81动作head的公共特征计算、第一层归一化折叠与寄存器布局，以及稀疏attention的实际页并集。若改候选数或采用更便宜的静态内容统计，再重算部署精度下的特征/标签并训练head；loss及best继续依据实际硬动作attention AV NMSE。当前轮没有重训，不能宣称训练已补偿误差。','',
        '## 产物与入口','',
        '- [K32全部8场景](ALL8_Dense_Opt4_Opt5_K32.mp4)、[V32全部8场景](ALL8_Dense_Opt4_Opt5_V32.mp4)：左Dense、中已接受Opt4-FP16、右Opt5；360帧、22.5秒。',
        '- [速度/质量/profile图](speed_quality_profile.png)、[差异场景抽帧](worst_case_frames.jpg)。',
        '- [逐次计时](timing_samples.csv)、[逐场景质量](quality_cases.csv)、[实际attention审计](attention_audit.csv)、[全部微基准](microbenchmarks.csv)、[汇总](summary.json)。',
        '- [模块profile](profile/modules.csv)、[kernel明细](profile/kernels.csv)、[trace](profile/trace.json.gz)。',
        '- [代码和先前实验冻结记录](manifest.json)、[最终核验](verification.json)。','',
        '```python','from migration.unified_router.opt5.network import load_flash','net, config = load_flash(experiment_directory)',
        'net.set_mode("V32-Set")  # 或 K32-Set','```','',
        '代码目录：`migration/unified_router/opt5/`；入口`network.py`，第一阶段`kernels.py`，权重复用`weight_route.py`，融合选择`select.py`，精简辅助数据`prepare.py`。完全重算消融保留在`route.py`、`candidate_simt.py`。原Opt4和模型权重未覆盖。',
        '限定当前batch1、12heads、128维、4680-token chunk、64-token页与顺序CUDA Graph推理。首chunk和layer0 Dense；后续首chunk作为普通历史，当前P16、历史P8/P4/P2；β和约15%预算保持。状态：`./v10_optimization5_progress.sh --once`。']
    lines += ['', '抽帧可见海豹场景的头部朝向与位置变化；三列视频保留所有8个场景。视频顺序：森林生物、SUV、海豹、Mumbai机器人、列车、教堂、Johannesburg机器人、水滴；每个场景2.8125秒。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Export plots as standalone artifacts.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    names=['Dense','Opt4-K32','Opt5-K32','Opt4-V32','Opt5-V32'];x=np.arange(len(names))
    axes[0].bar(x,[speed[n]['generation_s']for n in names],label='Generation')
    axes[0].bar(x,[speed[n]['decode_s']for n in names],bottom=[speed[n]['generation_s']for n in names],label='VAE + transfer')
    axes[0].set_xticks(x,names,rotation=35,ha='right');axes[0].set_ylabel('Seconds');axes[0].set_title('Matched end-to-end timing');axes[0].legend(fontsize=8)
    for i,kind in enumerate(['K32','V32']):
        rr=[r for r in quality if r['variant']=='Opt5'and r['mode']==kind]
        axes[1].scatter([i]*8,[r['delta_lpips_vs_opt4']for r in rr],alpha=.7)
        ci=qs['Opt5-'+kind]['delta_ci95'];mean=qs['Opt5-'+kind]['delta_lpips']
        axes[1].errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],color='black',fmt='s',capsize=5)
    axes[1].axhline(0,color='gray',lw=1);axes[1].set_xticks([0,1],['K32','V32']);axes[1].set_ylabel('Delta LPIPS to Dense vs Opt4');axes[1].set_title('8 paired Val prompts; lower is better')
    keys=['route_graph_copy_and_replay','sparse_attention_including_union','ffn'];labels=['Route','Sparse attention','FFN'];x=np.arange(3)
    for i,n in enumerate(['Opt4','Opt5']):axes[2].bar(x+(i-.5)*.34,[modules[n][k]for k in keys],width=.34,label=n)
    axes[2].set_xticks(x,labels,rotation=20);axes[2].set_ylabel('Kernel milliseconds');axes[2].set_title('Layer24 / chunk4 / step3');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'speed_quality_profile.png',dpi=180);fig.savefig(out/'speed_quality_profile.pdf');plt.close(fig)
    # Concatenate every prompt in registry order, retain native three-column size.
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();pids=read(root/'e2e_manifest.json')['prompt_ids']
    for kind in ['K32','V32']:
        playlist=out/f'concat_{kind}.txt';playlist.write_text(''.join(f"file '{out/'quality'/f'{pid}.seed0'/f'Dense_Opt4_Opt5_{kind}.mp4'}'\n"for pid in pids))
        subprocess.run([ffmpeg,'-y','-v','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy',str(out/f'ALL8_Dense_Opt4_Opt5_{kind}.mp4')],check=True)
    video_metadata={}
    import imageio.v2 as imageio
    for kind in ['K32','V32']:
        path=out/f'ALL8_Dense_Opt4_Opt5_{kind}.mp4';reader=imageio.get_reader(path);meta=reader.get_meta_data();frames=reader.count_frames();reader.close()
        assert frames==360 and meta['fps']==16 and tuple(meta['size'])==(2496,480),(kind,frames,meta)
        video_metadata[kind]=dict(frames=frames,fps=meta['fps'],size=meta['size'],duration=meta['duration'],bytes=path.stat().st_size)
    write(out/'video_metadata.json',video_metadata)
    # Inspect the highest measured per-prompt degradation for each head.
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    panels=[]
    for kind in ['K32','V32']:
        prompt=qs['Opt5-'+kind]['worst_prompt'];reader=imageio.get_reader(out/'quality'/prompt/f'Dense_Opt4_Opt5_{kind}.mp4')
        im=Image.fromarray(reader.get_data(40));reader.close();im.thumbnail((1248,240))
        panel=Image.new('RGB',(1248,268),'white');panel.paste(im,(0,28));ImageDraw.Draw(panel).text((8,8),kind+' / '+prompt+' / frame40',fill='black');panels.append(panel)
    canvas=Image.new('RGB',(1248,536),'white')
    for i,im in enumerate(panels):canvas.paste(im,(0,i*268))
    canvas.save(out/'worst_case_frames.jpg',quality=92)
    write(out/'analysis_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__).resolve(),Path(__file__).with_name('opt5_boundary_check.py').resolve(),Path(__file__).with_name('opt5_final_status.py').resolve(),base/'v10_optimization5_progress.sh']})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
