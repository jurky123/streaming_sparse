"""Opt6: preserve raw measurements, source checks, videos, and a concise report."""
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
    for name in ['Opt5-K32','Opt6-K32','Opt6-V32']:assert hashes[name]==hashes[name+'-replica']
    import torch
    pid=read(root/'e2e_manifest.json')['prompt_ids'][0]
    for kind in ['K32','V32']:
        timed=torch.load(out/f'benchmark/Opt6-{kind}/output.pt',map_location='cpu',weights_only=True)['latent']
        validated=torch.load(out/f'quality/{pid}.seed0/Opt6-{kind}.latent.pt',map_location='cpu',weights_only=True)
        assert torch.equal(timed,validated),('timing/quality mismatch',kind)
        before=torch.load(out/f'benchmark/Opt5-{kind}/output.pt',map_location='cpu',weights_only=True)['latent']
        saved=torch.load(root/f'optimization5/quality/{pid}.seed0/Opt5-{kind}.latent.pt',map_location='cpu',weights_only=True)
        assert torch.equal(before,saved),('accepted baseline mismatch',kind)
    boundary=read(out/'boundary_checks.json');assert boundary['status']=='passed'and len(boundary['cases'])==7
    quality=[];audit=[];cases=sorted((out/'quality').glob('*/result.json'));assert len(cases)==8
    for file in cases:
        doc=read(file)
        for kind in ['K32','V32']:
            old=read(root/'optimization5/quality'/file.parent.name/f'Opt5-{kind}.json')
            new=doc['variants']['Opt6-'+kind];assert new['first_chunk_identical']
            for variant,row in [('Opt5',old),('Opt6',new)]:
                met=row['quality_vs_dense']['sparse_frames_only'];prior=old['quality_vs_dense']['sparse_frames_only']
                quality.append(dict(prompt=file.parent.name,variant=variant,mode=kind,lpips=met['lpips'],psnr_db=met['psnr_db'],
                    weighted_keep=row['weighted_keep'],delta_lpips_vs_opt5=met['lpips']-prior['lpips'],
                    lpips_to_opt5=new['quality_vs_opt5']['sparse_frames_only']['lpips']if variant=='Opt6'else 0.))
            audit.extend(dict(prompt=file.parent.name,mode=kind,**r)for r in new['attention_audit'])
    assert all(np.isfinite(r[k])for r in quality for k in ['lpips','psnr_db','weighted_keep'])
    rng=np.random.default_rng(20261024);qs={}
    for variant in ['Opt5','Opt6']:
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
    assert read(out/'profile/modules.json')['observer_latents_bit_identical']
    profiles={'Opt5':read(root/'optimization5/profile/kernel_attribution.json'),'Opt6':attribution(out/'profile')}
    modules={k:{r['module']:r['kernel_ms']for r in p['modules']}for k,p in profiles.items()}
    kernels=collections.defaultdict(float)
    for r in profiles['Opt6']['kernels']:
        if r['module']=='route_graph_copy_and_replay':kernels[r['kernel']]+=r['milliseconds']
    checks=read(out/'checks.json');assert checks['finite_nonempty']and len(checks['rows'])==18
    correct=dict(queries=sum(r['queries']for r in checks['rows']),changed_queries=sum(r['changed_queries']for r in checks['rows']),
        delta_nmse=sum(r['attention_delta_sse']for r in checks['rows'])/sum(r['reference_energy']for r in checks['rows']))
    probes=[]
    for f in (out/'probes').glob('*.json'):
        if f.name.endswith('.status.json'):continue
        r=read(f);probes.append(dict(name=r['name'],reference_route_ms=r['timings_ms']['reference_route'],route_ms=r['timings_ms']['route'],reference_head_ms=r['timings_ms']['reference_head'],head_ms=r['timings_ms']['head'],
            changed_queries=r['changed_queries'],attention_nmse=r['attention_delta_nmse']))
    csv_write(out/'microbenchmarks.csv',probes);csv_write(out/'timing_samples.csv',samples);csv_write(out/'speed_summary.csv',list(speed.values()))
    csv_write(out/'quality_cases.csv',quality);csv_write(out/'attention_audit.csv',audit)
    summary=dict(selected_entry='migration.unified_router.opt6.network.load_head',options=seal['options'],speed=speed,quality=qs,
        actual_attention_audit=aa,correctness=correct,profiles={k:dict(total_kernel_ms=p['total_kernel_ms'],modules=p['modules'])for k,p in profiles.items()},
        baseline='Opt5 accepted FP16 route',retrained=False,test_used=False,source_hashes_verified=len(digests),boundary_checks=7,timing_quality_latents_identical=True)
    summary['latent_identical_cases']=sum(read(f)['latent_bit_identical']for f in(out/'quality').glob('*.seed0/Opt6-*.json'))
    summary['head_prediction_max_nmse']=max(r['head_prediction_nmse']for r in checks['rows'])
    summary['head_prediction_max_abs']=max(r['head_prediction_max_abs']for r in checks['rows'])
    write(out/'summary.json',summary);write(out/'verification.json',dict(status='passed',source_hashes=len(digests),first_chunk_identical=True,
        timed_repeats_deterministic=True,cross_gpu_replicas_identical=True,profile_latents_identical=True,boundary_checks=7,timing_quality_latents_identical=True,accepted_baseline_reproduced=True,quality_cases=16,actual_attention_blocks=144))
    lines=['# SlimAV Opt6：分解81动作head，复用候选计算','',
        '**以已接受的Opt5为基线；只优化动作head，保持24维特征、81动作、原权重、原预算及逐query路由。未重训。**','',
        '## 端到端速度','',
        '单张A100-SXM4-40GB，45帧480×832、16fps，4 chunks×4去噪步。同一Val prompt/seed0，完整预热一次后测三次；K32/V32新版本均有跨GPU replica。表列主GPU均值，所有样本见CSV。生成含上下文重建，总计含VAE及RGB回传，加载/编译/LPIPS/视频编码不计。全部测速后才运行质量评估与profile。','',
        '|方案|生成 / s|VAE及回传 / s|总计 / s|峰值allocated / GiB|','|---|---:|---:|---:|---:|']
    for name in ['Dense','Opt5-K32','Opt6-K32','Opt5-V32','Opt6-V32']:
        r=speed[name];lines.append(f"|{name}|{r['generation_s']:.3f}|{r['decode_s']:.3f}|{r['total_s']:.3f}|{r['peak_allocated_GiB']:.2f}|")
    for kind in ['K32','V32']:
        old=speed['Opt5-'+kind];new=speed['Opt6-'+kind]
        lines+=['',f"{kind}生成耗时减少{1-new['generation_s']/old['generation_s']:.2%}，速度{old['generation_s']/new['generation_s']:.3f}×；总时长减少{1-new['total_s']/old['total_s']:.2%}。"]
    lines+=['','仍慢于Dense。只优化head的收益受它在整个生成中的占比限制；路由打分、选候选、均值、稀疏attention及主干计算仍要执行。','',
        '## 具体如何改写','',
        '旧head逐query为81个动作（GPU补齐128行）计算几何量、拼24维特征、做24→16→1的MLP。删除候选和添加候选各只有8个；同一候选在多个交换动作中反复使用，但旧kernel反复加载向量并重算范数/点积。','',
        '第一层的24维输入可以分成：11维query公共特征、6维只依赖删除候选的特征、6维只依赖添加候选的特征、1维删除/添加之间的交互点积。对归一化后的特征与原W1，数学上可写为：','',
        '`y(i,j) = shared(query) + remove(i) + add(j) + cross(i,j) * W1[:,5]`','',
        '1. 对8个删除和8个添加候选各计算一次范数、与集合误差向量的点积、标准化分数、成本和来源。保留一个“不删除/不添加”槽。',
        '2. 将6维删除、6维添加特征各自归一化并转FP16，用小矩阵乘法算出各候选的16维第一层贡献；11维公共项只算一次。原均值/标准差与权重保持不变。',
        '3. 8×8交互点积采用局部TF32x3、FP32累加，再按原方式归一化并转FP16；这是高精度Tensor Core路径，全局PyTorch TF32仍关闭。',
        '4. 第二个kernel读取公共项、删除项、添加项和交互项，组合原81动作的第一层输出，执行原erf GELU、末层线性评分、合法性检查与最小动作编号破同分规则，直接更新mask。',
        '5. 所有query仍独立，候选、初始mask和注意力kernel均沿用Opt5。','',
        '原FP16特征和FP16第一层权重的乘法仍然保留；主要差别是FP32求和顺序及交互点积的TF32x3归约。没有把归一化折入权重，因为这样会改变FP16取整位置。本轮不做checkpoint重选，也没有减少动作空间。','',
        '仅计第一层的理想标量乘法：直接评估约81×24×16=31,104次；复用后公共项11×16、两侧各9×6×16、交互项81×16，合计约3,200次。这不是实际GPU指令数：Tensor Core分块补齐、交互点积、GELU、末层、访存与启动开销仍存在，所以不能预期整个head或端到端达到约10倍加速。','',
        '分解版每query额外写一份384个FP32槽的临时缓冲，约1.5KB；512-query批次×12heads约9.4MB，整层累计约86.3MB。旧head的完整特征本来在寄存器中，并非落显存；本轮收益来自减少重复几何/MLP工作和改善执行布局，不能宣称消除了旧版全特征HBM缓冲。','',
        '## 实测消融','',
        f"共{len(probes)}组配置，代表真实数据512-query批次：原head约0.154ms；仅重排特征的版本约0.132ms；两阶段分解约0.090ms。相应整条路由约0.624→0.557ms。微基准只作选型，端到端表包含真实构造、复制与全部调用开销。",'',
        '完全融合成一个kernel已实现，但将9×9合法组合补齐成16×16网格，增加GELU/归约工作与寄存器压力，实测慢于两阶段版本，未采用。FP16交互点积也试过，与高精度TF32x3版差距很小，因此最终选TF32x3。',
        '本机Triton版本缺少新版gather接口；特征重排消融使用join构造布局，最终分解版使用小矩阵和紧凑缓冲。参考：[Triton join文档](https://triton-lang.org/main/python-api/generated/triton.language.join.html)、[dot精度文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)。所有新kernel为本地实现。','',
        '## 正确性与端到端视频','',
        f"3层×3历史长度×K32/V32共18组，覆盖17/72/128查询及尾页：{correct['queries']}个head-query中{correct['changed_queries']}个最终mask变化。逐动作预测值也核对过，最大case NMSE {summary['head_prediction_max_nmse']:.3e}，最大绝对差{summary['head_prediction_max_abs']:.3e}；检查包括原始全部81动作分数，不仅比较最终argmax。",'',
        '另外7组边界：无候选、只删、只加、单页已选、全页已选、零head、全部有效动作同为正分。最终mask与参考全部一致，保留非空集合与原破同分规则。',
        '视频评估为全部8个注册Val prompt×K32/V32、seed0。LPIPS与PSNR取未压缩RGB后36帧，首9帧与Dense完全一致。LPIPS衡量与Dense的感知差异；95%区间按prompt bootstrap10000次，不将帧或query当独立质量样本。','',
        '|方案|LPIPS对Dense↓|相对Opt5变化|95% prompt bootstrap区间|KV保留率|','|---|---:|---:|---|---:|']
    for kind in ['K32','V32']:
        for version in ['Opt5','Opt6']:
            r=qs[version+'-'+kind];ci=r['delta_ci95'];lines.append(f"|{version}-{kind}|{r['lpips']:.6f}|{r['delta_lpips']:+.6f}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|{r['keep']:.3%}|")
    lines+=['',f"16组完整latent中，{summary['latent_identical_cases']}组与Opt5逐bit相同。所有计时重复及跨GPU replica一致；计时输出与质量评估输出一致；本轮重新运行的Opt5与上轮已接受版本一致。",'',
        f"在新轨迹的同一Q/K/V上抽查layer1/24/29、三个稀疏chunk、step3：{aa['blocks']}个block、{aa['queries']}个head-query，{aa['changed_queries']}个最终mask变化。实际attention输出相对Opt5的增量NMSE为 **{aa['delta_nmse']:.3e}**；实际attention NMSE对Dense为{aa['nmse_to_dense']:.9f}，相同输入Opt5为{aa['reference_nmse_to_dense']:.9f}。这与head预测分数误差分开统计。"]
    for kind in ['K32','V32']:
        r=qs['Opt6-'+kind];lines+=['',f"{kind}逐场景最大LPIPS退化：{r['worst_delta']:+.6f}，场景{r['worst_prompt']}。"]
    lines+=['','## 最终profile','',
        'layer24、第4chunk、step3；单独profile，已验证观察器不改变latent。每个kernel按CUDA runtime/driver correlation唯一归属，graph节点归到graph launch。表为kernel累计时间；Opt5列使用上轮同位置profile，端到端速度表是本轮重新测量。','',
        '|模块 / ms|Opt5|Opt6|','|---|---:|---:|']
    for key,label in [('route_graph_copy_and_replay','路由及辅助数据'),('sparse_attention_including_union','稀疏attention及并集'),('ffn','FFN'),('cross_attention','交叉attention'),('current_prototypes','当前原型'),('prototype_projection','原型投影')]:
        lines.append('|'+label+'|'+'|'.join(f'{modules[n][key]:.3f}'for n in ['Opt5','Opt6'])+'|')
    lines+=['|整层合计|'+'|'.join(f"{profiles[n]['total_kernel_ms']:.3f}"for n in ['Opt5','Opt6'])+'|','',
        '|路由kernel|累计 ms|','|---|---:|']
    for k,v in sorted(kernels.items(),key=lambda r:-r[1]):
        if len(k)<80:lines.append(f'|{k}|{v:.3f}|')
    oldhead=sum(r['milliseconds']for r in profiles['Opt5']['kernels']if r['kernel']=='_candidate_actions')
    newhead=kernels['_factor_prepare']+kernels['_factor_finish']
    summary['head_profile']=dict(reference_ms=oldhead,new_ms=newhead,reduction=1-newhead/oldhead)
    lines+=['',f"完整层head从{oldhead:.3f}ms降至{newhead:.3f}ms，减少{1-newhead/oldhead:.2%}。",'',
        '下一步应针对当前profile中的打分/页统计、均值聚合与候选选择继续优化。若改特征或候选集合，需要重新生成部署精度下的训练标签并适配小head，loss和best仍按实际硬动作AV NMSE；当前改写没有这样的重训需求证据。','',
        '## 产物与复现','',
        '- [K32全部8场景](ALL8_Dense_Opt5_Opt6_K32.mp4)、[V32全部8场景](ALL8_Dense_Opt5_Opt6_V32.mp4)：左Dense、中Opt5、右Opt6，360帧、22.5秒。',
        '- [速度/质量/profile图](speed_quality_profile.png)、[差异场景抽帧](worst_case_frames.jpg)、[汇总JSON](summary.json)。',
        '- [逐次计时](timing_samples.csv)、[逐场景质量](quality_cases.csv)、[真实attention审计](attention_audit.csv)、[微基准消融](microbenchmarks.csv)、[跨层及动作分数检查](checks.json)、[动作边界检查](boundary_checks.json)。',
        '- [模块profile](profile/modules.csv)、[kernel明细](profile/kernels.csv)、[trace](profile/trace.json.gz)、[冻结记录](manifest.json)、[最终核验](verification.json)。','',
        '代码：`migration/unified_router/opt6/factor.py`为最终两阶段kernel；`route.py`与`network.py`为推理入口。`packed.py`与`fused.py`保留消融；`debug.py`用于输出动作分数检查，不在正式推理路径中。','',
        '```python','from migration.unified_router.opt6.network import load_head','net, config = load_head(experiment_directory)','net.set_mode("V32-Set")  # 或 K32-Set','```','',
        '状态：`./v10_optimization6_progress.sh --once`。适用当前batch1、12heads、128维、4680-token chunk、64-token页与顺序CUDA Graph推理。首chunk、layer0 Dense；后续首chunk按普通历史处理；当前P16、历史P8/P4/P2；β与约15%预算不变。Test未使用，权重未修改，Opt5保留。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'summary.json',summary)
    # Export plots as standalone artifacts.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    names=['Dense','Opt5-K32','Opt6-K32','Opt5-V32','Opt6-V32'];x=np.arange(len(names))
    axes[0].bar(x,[speed[n]['generation_s']for n in names],label='Generation')
    axes[0].bar(x,[speed[n]['decode_s']for n in names],bottom=[speed[n]['generation_s']for n in names],label='VAE + transfer')
    axes[0].set_xticks(x,names,rotation=35,ha='right');axes[0].set_ylabel('Seconds');axes[0].set_title('Matched end-to-end timing');axes[0].legend(fontsize=8)
    for i,kind in enumerate(['K32','V32']):
        rr=[r for r in quality if r['variant']=='Opt6'and r['mode']==kind]
        axes[1].scatter([i]*8,[r['delta_lpips_vs_opt5']for r in rr],alpha=.7)
        ci=qs['Opt6-'+kind]['delta_ci95'];mean=qs['Opt6-'+kind]['delta_lpips']
        axes[1].errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],color='black',fmt='s',capsize=5)
    axes[1].axhline(0,color='gray',lw=1);axes[1].set_xticks([0,1],['K32','V32']);axes[1].set_ylabel('Delta LPIPS to Dense vs Opt5');axes[1].set_title('8 paired Val prompts; lower is better')
    keys=['route_graph_copy_and_replay','sparse_attention_including_union','ffn'];labels=['Route','Sparse attention','FFN'];x=np.arange(3)
    for i,n in enumerate(['Opt5','Opt6']):axes[2].bar(x+(i-.5)*.34,[modules[n][k]for k in keys],width=.34,label=n)
    axes[2].set_xticks(x,labels,rotation=20);axes[2].set_ylabel('Kernel milliseconds');axes[2].set_title('Layer24 / chunk4 / step3');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'speed_quality_profile.png',dpi=180);fig.savefig(out/'speed_quality_profile.pdf');plt.close(fig)
    # Concatenate every prompt in registry order, retain native three-column size.
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();pids=read(root/'e2e_manifest.json')['prompt_ids']
    for kind in ['K32','V32']:
        playlist=out/f'concat_{kind}.txt';playlist.write_text(''.join(f"file '{out/'quality'/f'{pid}.seed0'/f'Dense_Opt5_Opt6_{kind}.mp4'}'\n"for pid in pids))
        subprocess.run([ffmpeg,'-y','-v','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy',str(out/f'ALL8_Dense_Opt5_Opt6_{kind}.mp4')],check=True)
    video_metadata={}
    import imageio.v2 as imageio
    for kind in ['K32','V32']:
        path=out/f'ALL8_Dense_Opt5_Opt6_{kind}.mp4';reader=imageio.get_reader(path);meta=reader.get_meta_data();frames=reader.count_frames();reader.close()
        assert frames==360 and meta['fps']==16 and tuple(meta['size'])==(2496,480),(kind,frames,meta)
        video_metadata[kind]=dict(frames=frames,fps=meta['fps'],size=meta['size'],duration=meta['duration'],bytes=path.stat().st_size)
    write(out/'video_metadata.json',video_metadata)
    # Inspect the highest measured per-prompt degradation for each head.
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    panels=[]
    for kind in ['K32','V32']:
        prompt=qs['Opt6-'+kind]['worst_prompt'];reader=imageio.get_reader(out/'quality'/prompt/f'Dense_Opt5_Opt6_{kind}.mp4')
        im=Image.fromarray(reader.get_data(40));reader.close();im.thumbnail((1248,240))
        panel=Image.new('RGB',(1248,268),'white');panel.paste(im,(0,28));ImageDraw.Draw(panel).text((8,8),kind+' / '+prompt+' / frame40',fill='black');panels.append(panel)
    canvas=Image.new('RGB',(1248,536),'white')
    for i,im in enumerate(panels):canvas.paste(im,(0,i*268))
    canvas.save(out/'worst_case_frames.jpg',quality=92)
    write(out/'analysis_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__).resolve(),base/'v10_optimization6_progress.sh']})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
