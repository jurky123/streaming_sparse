"""Merge the same complete Val observations and pair uncertainty by prompt."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import numpy as np
from streaming_sparse.v08.common import atomic_json,sha,source_identity,HERE
from streaming_sparse.v08.metrics import summarize


MODES=['Dense','mass','K200','V200','Kbest','Kpaused']


def paired(a,b):
    keys=sorted(a);assert keys==sorted(b)
    difference=np.array([b[k]-a[k]for k in keys]);draw=np.random.default_rng(20261019).integers(0,len(keys),size=(5000,len(keys)))
    return dict(delta_V_minus_K=float(difference.mean()),prompt_paired_95CI=np.quantile(difference[draw].mean(-1),[.025,.975]).tolist(),
        prompt_count=len(keys),per_prompt_delta=dict(zip(keys,difference.tolist())),bootstrap_repeats=5000,bootstrap_seed=20261019,
        note='One trained Router seed only; prompt resampling does not establish training-seed robustness.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--partial',action='store_true');args=p.parse_args();root=Path(args.directory).resolve()
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['source_sha256']==source_identity()
    output=dict(manifest=manifest,offline={},videos={},status='running',updated_at=datetime.now(timezone.utc).isoformat());evidence={str(root/'manifest.json'):sha(root/'manifest.json')}
    all_rows=[]
    for mode in MODES:
        paths=[root/f'rank{i}/{mode}_offline.json'for i in range(8)]
        if not all(p.exists()for p in paths):continue
        rows=[row for path in paths for row in json.loads(path.read_text())['rows']]
        rows.sort(key=lambda r:(r['prompt_id'],r['generation_seed'],r['history_tokens'],r['step']))
        assert len(rows)==192 and len({r['sample_id']for r in rows})==192
        expected={(pid,g,h,s)for pid in manifest['prompt_ids']for g in [0,1]for h in [4680,9360,14040]for s in range(4)}
        assert {(r['prompt_id'],r['generation_seed'],r['history_tokens'],r['step'])for r in rows}==expected
        output['offline'][mode]=dict(**summarize(rows),actual_keep=float(np.mean([r['actual_history_keep']for r in rows])))
        all_rows.extend(dict(mode=mode,**row)for row in rows)
        evidence.update({str(path):sha(path)for path in paths})
    if all(m in output['offline']for m in ['K200','V200']):
        output['offline_paired']=paired(output['offline']['K200']['prompts'],output['offline']['V200']['prompts'])
    if all_rows:
        with(root/'offline_rows.csv').open('w')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        atomic_json(root/'offline_summary.json',output['offline'])
    paths=[root/f'rank{i}/video_rows.json'for i in range(8)]
    video_rows=[row for path in paths if path.exists()for row in json.loads(path.read_text())]
    if len(video_rows)==16:
        assert {(r['prompt_id'],r['generation_seed'])for r in video_rows}=={(pid,g)for pid in manifest['prompt_ids']for g in [0,1]}
        for mode in MODES[1:]:
            values=[r['metrics'][mode]['quality']for r in video_rows];item={}
            for scope in ['full','sparse_frames_only']:
                mse=float(np.mean([v[scope]['mse_rgb01']for v in values]));lp=float(np.mean([v[scope]['lpips']for v in values]))
                item[scope]=dict(mse_rgb01=mse,psnr_db=-10*math.log10(mse)if mse else None,lpips=lp)
            item['chunks']=[]
            for i in range(4):
                mse=float(np.mean([v['chunks'][i]['mse_rgb01']for v in values]));lp=float(np.mean([v['chunks'][i]['lpips']for v in values]))
                item['chunks'].append(dict(chunk=i+1,mse_rgb01=mse,psnr_db=-10*math.log10(mse)if mse else None,lpips=lp))
            item['prompts']={pid:float(np.mean([r['metrics'][mode]['quality']['sparse_frames_only']['lpips']for r in video_rows if r['prompt_id']==pid]))for pid in manifest['prompt_ids']}
            counts=[c for r in video_rows for c in r['metrics'][mode]['sparsity']]
            kept=sum(c['retained_query_tokens']for c in counts);historical=sum(c['routed_queries']*c['history_tokens']for c in counts)
            item['historical_connection_keep']=kept/historical
            item['all_layer_denoising_connection_drop']=(1-kept/historical)*.6*.1
            output['videos'][mode]=item
        output['videos_paired_sparse_lpips']=paired(output['videos']['K200']['prompts'],output['videos']['V200']['prompts'])
        atomic_json(root/'video_summary.json',dict(summary=output['videos'],rows=video_rows,paired_sparse_lpips=output['videos_paired_sparse_lpips']))
        evidence.update({str(path):sha(path)for path in paths})
    finished=all(json.loads((root/f'rank{i}/status.json').read_text()).get('status')=='completed'for i in range(8))
    complete=len(output['offline'])==6 and(len(video_rows)==16 or not manifest['videos'])and finished
    if complete:
        for i in range(8):
            st=json.loads((root/f'rank{i}/status.json').read_text());assert st['status']=='completed'and st['backbone_versions_unchanged']and st['routers_unchanged']
        assert sha(HERE/'runs/seed0/F-K/recovery.pt')==manifest['checkpoints']['Kpaused']['sha256'],'Paused training checkpoint changed'
        assert json.loads((HERE/'runs/seed0/F-K/USER_PAUSE.json').read_text())['status']=='paused'
        output['status']='completed';output['passed']=True
    elif not args.partial:raise RuntimeError('Evaluation is not complete')
    output['video_cases_completed']=len(video_rows);output['evidence']=evidence
    assets=root/'asset_identity.json'
    if assets.exists():output['assets']=json.loads(assets.read_text());evidence[str(assets)]=sha(assets)
    atomic_json(root/'summary.json',output)
    text=['# 冻结 Backbone：K-only 与 V-aware 质量评估','',
        f"状态：{output['status']}。8个Val Prompt ×2生成Seed，离线192完整输入；训练Seed=0。当前480×832、4 chunks、每chunk3 latent frames，共45 RGB帧，16fps。",'',
        '骨干与LoRA全部冻结；所有checkpoint的LoRA张量一致且B=0。原训练停在1039步，本轮没有训练或恢复训练。只在原层4/14/24的12 Heads稀疏，当前段全保留，历史保留20%；其余27层Dense，历史写入也Dense。','',
        '## 对照身份','',
        '- K200 / V200：共享校准起点、同一数据/采样顺序、同样200次head warm-start；Sketch固定，只学习残差评分头。这是本轮唯一训练预算配对的K/V对照。',
        '- mass：校准后的LogZ排序；V-aware warm-start按Val选出的best实际是step0，已核对其全部Router张量等于该初始化。',
        '- Kbest / Kpaused：现有K-only主训练分支的Val最佳200步/暂停1039步；此前还承接旧数据640步。这两项训练更多，不能拿来单独归因V-aware是否有效。',
        '- 未运行封存Test，也未训练F-V主分支。因此结果是现有checkpoint的开发诊断，不能作为只训Router的质量上限或正式多Seed结论。','',
        '## 离线外部输入 x0 误差','',
        '固定公共Replay输入，以Dense Teacher保存的x0为参考。每输入NMSE后Prompt等权，越低越好。',
        '', '| 模型 | NMSE | P90 | P99 | 输入等权实际历史保留率 |','|---|---:|---:|---:|---:|']
    for name,d in output['offline'].items():text.append(f"| {name} | {d['prompt_equal_error']:.9f} | {d['input_error']['p90']:.9f} | {d['input_error']['p99']:.9f} | {d['actual_keep']:.4%} |")
    if 'offline_paired'in output:
        d=output['offline_paired'];text+=['',f"V200−K200：{d['delta_V_minus_K']:.9g}；按Prompt配对bootstrap 95%CI：{d['prompt_paired_95CI']}。负值才表示V200误差更低。"]
    text+=['','## 自生成闭环视频','',
        '每个分支使用相同Prompt/Seed/原生噪声调用与采样器，各自生成历史；RGB经相同原生BF16 VAE解码。PSNR使用压缩前[0,1] RGB合并MSE，LPIPS使用预训练AlexNet v0.1、全分辨率逐帧均值。它们衡量与同噪声Dense视频的相似度，不是视频审美或文本一致性的绝对评分。',
        '首chunk本身Dense，所有分支首9帧必须逐像素一致；主视频比较额外排除这9帧，只统计后36帧。',
        '', '| 模型 | 全视频 PSNR dB | 全视频 LPIPS | 后36帧 PSNR dB | 后36帧 LPIPS | 第4chunk LPIPS |','|---|---:|---:|---:|---:|---:|']
    for name,d in output['videos'].items():text.append(f"| {name} | {d['full']['psnr_db']:.4f} | {d['full']['lpips']:.6f} | {d['sparse_frames_only']['psnr_db']:.4f} | {d['sparse_frames_only']['lpips']:.6f} | {d['chunks'][3]['lpips']:.6f} |")
    if output['videos']:
        d=output['videos_paired_sparse_lpips'];text+=['',f"后36帧LPIPS的V200−K200：{d['delta_V_minus_K']:.9g}；按Prompt配对bootstrap 95%CI：{d['prompt_paired_95CI']}。负值表示V200更接近Dense；本次区间跨零。"]
        d=output['videos']['K200'];text+=['',f"闭环历史连接加权实际保留 {d['historical_connection_keep']:.4%}。由于只有3/30层稀疏、当前段全保留，整段全层去噪逻辑QK连接删除率仅 {d['all_layer_denoising_connection_drop']:.4%}；这不是全30层80%稀疏的质量结果，也不是实测FLOPs或加速比。"]
        text+=['','### 逐chunk（PSNR dB / LPIPS）','','| 模型 | chunk1 | chunk2 | chunk3 | chunk4 |','|---|---|---|---|---|']
        for name,d in output['videos'].items():
            cells=['∞ / 0'if c['mse_rgb01']==0 else f"{c['psnr_db']:.3f} / {c['lpips']:.5f}"for c in d['chunks']]
            text.append('| '+name+' | '+' | '.join(cells)+' |')
        text+=['','### 对照视频','', '每个视频横向三个画面；左侧Dense，中/右按文件名标注。完整16对Prompt/Seed均提供，未按表现挑选。','']
        for row in sorted(video_rows,key=lambda r:(r['prompt_id'],r['generation_seed'])):
            case=f"{row['prompt_id']}.seed{row['generation_seed']}"
            text.append(f"- {case}：[Dense / K200 / V200](videos/{case}/paired_Dense_K200_V200.mp4)；[Dense / Kbest / Kpaused](videos/{case}/current_Dense_Kbest_Kpaused.mp4)")
    text+=['','全部数值：summary.json、offline_rows.csv、video_summary.json；checkpoint/数据/源码身份：manifest.json。RGB帧与质量指标取自压缩前像素，MP4仅供目视查看。','']
    if complete and output['videos']:
        a=output['offline']['K200']['prompt_equal_error'];b=output['offline']['V200']['prompt_equal_error']
        x=output['videos']['K200']['sparse_frames_only']['lpips'];y=output['videos']['V200']['sparse_frames_only']['lpips']
        conclusions=['## 当前结论','',
            f'同预算200步下，V-aware离线NMSE高 {(b/a-1)*100:.2f}%，闭环后36帧LPIPS低 {(1-y/x)*100:.2f}%；两项Prompt配对置信区间都跨零，未显示稳定优势。视频LPIPS仅3/8个Prompt均值改善，平均收益受个别Prompt影响较大。',
            '校准Mass的闭环LPIPS均值最好；复杂残差评分头尚未证明其额外计算值得。Kbest离线误差最好，但闭环LPIPS未优于Mass。',
            '所有稀疏分支误差均随chunk累积：第2chunk LPIPS约0.022，第4chunk约0.108–0.116。当前Kpaused在离线与闭环上都没有超过Kbest，不能把多训练步数当作质量提高。',
            '以上属于一个Router训练Seed、8个开发Val Prompt上的现有权重比较。V-aware缺少与当前K-only等预算的主训练，尚未检验其充分训练后的表现；未测全30层稀疏或更长视频。',
            '', '[质量曲线](quality_comparison.png) · [PDF](quality_comparison.pdf)','']
        text=text[:4]+conclusions+text[4:]
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(dict(status=output['status'],offline_modes=list(output['offline']),video_cases_completed=len(video_rows),report=str(root/'REPORT.md'))),flush=True)


if __name__=='__main__':main()
