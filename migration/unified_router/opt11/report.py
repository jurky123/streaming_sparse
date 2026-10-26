"""Seal both fine-tuning stages, native video metrics, paired timing and provenance."""
import argparse,json,hashlib,statistics,subprocess
from pathlib import Path
import torch
from ..common import atomic_json,sha,save

def read(p):return json.loads(Path(p).read_text())
def digest(z):return hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();m=read(root/'experiment.json');base=Path(m['parent_experiment']);pids=read(base/'e2e_manifest.json')['prompt_ids'];variants=['Compact','Aggressive'];offline={};exports={}
 for v in variants:
  rows=[];heads={}
  for layer in m['layers']:
   first=root/f'training/layer{layer:02d}/{v}';second=root/f'strong_training/layer{layer:02d}/{v}';initial=read(first/'initial_validation.json');best1=read(first/'best_validation.json');best2=read(second/'best_validation.json');assert best1['nmse']<=initial['nmse']and best2['nmse']<=best1['nmse']+1e-10
   state=torch.load(second/'best.pt',map_location='cpu',weights_only=True);assert state['best_metric']=='actual_deployment_attention_nmse'and state['best_value']==best2['nmse'];heads[str(layer)]=state['head'];rows.append(dict(layer=layer,initial=initial['nmse'],stage1=best1['nmse'],stage2=best2['nmse'],best_step_stage1=best1['updates'],best_step_stage2=best2['updates'],initial_keep=initial['keep'],final_keep=best2['keep']))
  offline[v]=dict(initial_nmse=statistics.mean(r['initial']for r in rows),stage1_nmse=statistics.mean(r['stage1']for r in rows),final_nmse=statistics.mean(r['stage2']for r in rows),improved_layers=sum(r['stage2']<r['initial']for r in rows),layers=rows)
  f=root/f'exports/{v}.best_heads.pt';save(f,dict(heads=heads,variant=v,parent=str(base),backbone_and_prototypes_included=False,best_metric='actual_deployment_attention_nmse',layers=m['layers']));exports[v]=dict(path=str(f),sha256=sha(f))
 names=['Dense','CompactBefore','CompactAfter','AggressiveBefore','AggressiveAfter'];per_prompt=[];quality={};checked=0
 locks=[read(root/f'quality/rank{i}_checkpoint_hashes.json')for i in range(8)];assert all(d==locks[0]for d in locks)
 for path,h in locks[0].items():assert sha(path)==h
 for index,pid in enumerate(pids):
  folder=root/f'paired/prompt{index}';assert read(folder/'status.json')['status']=='completed';timed=read(folder/'timings.json');assert len(timed['records'])==20;per_prompt.append(dict(prompt_id=pid,timing=read(folder/'summary.json')))
  expected={}
  for n in names:
   z=torch.load(base/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)['Dense']if n=='Dense'else torch.load(root/f'quality/{pid}.seed0/{n}.latent.pt',map_location='cpu',weights_only=True);expected[n]=digest(z)
  for r in timed['records']:assert r['latent_sha256']==expected[r['variant']],(pid,r['variant']);checked+=1
 timing={n:{k:statistics.mean(r['timing'][n][k]for r in per_prompt)for k in ['generation_seconds','decode_seconds','total_seconds']}for n in names}
 for n in names[1:]:
  rows=[read(root/f'quality/{pid}.seed0/{n}.json')for pid in pids];assert all(r['first_chunk_identical']and r['repeat_identical']for r in rows);audits=[a for r in rows for a in r['attention_audit']]
  quality[n]=dict(lpips_mean=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['lpips']for r in rows),psnr_mean=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['psnr_db']for r in rows),keep_mean=statistics.mean(r['weighted_keep']for r in rows),audited_actual_attention_nmse=sum(r['sparse_to_dense_sse']for r in audits)/sum(r['dense_energy']for r in audits),max_masked_implementation_nmse=max(r['masked_implementation_nmse']for r in audits),per_prompt=[dict(prompt_id=pid,used_for_head_selection=pid in m['splits']['val'],lpips=r['quality_vs_dense']['sparse_frames_only']['lpips'],psnr=r['quality_vs_dense']['sparse_frames_only']['psnr_db'],keep=r['weighted_keep'])for pid,r in zip(pids,rows)])
  if n=='CompactBefore':quality[n]['bit_identical_to_prior_opt10_prompts']=sum(r.get('matches_opt10',False)for r in rows)
  for split,in_monitor in [('head_selection_val',True),('additional_val',False)]:
   selected=[r for pid,r in zip(pids,rows)if(pid in m['splits']['val'])==in_monitor];quality[n][split]=dict(prompts=len(selected),lpips=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['lpips']for r in selected))
 improvements={v:dict(offline_nmse_reduction=1-offline[v]['final_nmse']/offline[v]['initial_nmse'],lpips_delta=quality[v+'After']['lpips_mean']-quality[v+'Before']['lpips_mean'],generation_seconds_delta=timing[v+'After']['generation_seconds']-timing[v+'Before']['generation_seconds'],lpips_improved_prompts=sum(a['lpips']<b['lpips']for a,b in zip(quality[v+'After']['per_prompt'],quality[v+'Before']['per_prompt'])))for v in variants}
 video_info={}
 for v in variants:
  listing=root/f'quality/concat_{v}.txt';listing.write_text(''.join("file '"+str(root/f'quality/{pid}.seed0/Dense_{v}_Before_After.mp4')+"'\n"for pid in pids));dest=root/f'ALL8_Dense_{v}_Before_After.mp4'
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(dest)],check=True)
  d=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(dest)]))['streams'][0];assert int(d['nb_read_frames'])==360 and d['width']==2496 and d['height']==480;video_info[v]=d
 original=read(root/'parent_lock.json')
 for path,h in {**original['source'],**original['parameters']}.items():assert sha(path)==h,path
 training_lock=read(root/'training_implementation_lock.json')
 for path,h in training_lock.items():assert sha(path)==h,path
 repo=Path(__file__).resolve().parents[3];source={str(f):sha(f)for f in sorted((repo/'migration/unified_router/opt11').glob('*.py'))};source[str(repo/'v10_finetune11_progress.sh')]=sha(repo/'v10_finetune11_progress.sh')
 result=dict(scope='29 layers, two variants, two 200-update stages; 192 Train events; 96 selection-Val events; 8 native Val videos seed0; Test unused',offline=offline,quality=quality,timing=timing,per_prompt_timing=per_prompt,improvements=improvements,exports=exports,backbone_and_prototypes_frozen=True)
 atomic_json(root/'final_summary.json',result);atomic_json(root/'final_verification.json',dict(timed_outputs_checked=checked,all160_timed_outputs_match_quality=checked==160,quality_outputs=32,repeated_trajectories_identical=32,first_chunk_dense_identical_all=True,parent_source_unchanged=len(original['source']),parent_parameter_files_unchanged=len(original['parameters']),training_source_lock_unchanged=True,head_checkpoints_locked_before_quality=58,checkpoint_hashes=locks[0],videos=video_info,test_used=False))
 atomic_json(root/'manifest.json',dict(code=source,parent_lock=original,training_lock=training_lock,evaluated_heads=locks[0],exports=exports))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,3,figsize=(16,4.8));colors=['#538fbc','#c77b42']
 for v,color in zip(variants,colors):
  curves=[]
  for layer in m['layers']:
   first=read(root/f'training/layer{layer:02d}/{v}/curve.json');second=read(root/f'strong_training/layer{layer:02d}/{v}/curve.json');vals=[r['nmse']for r in first]+[r['nmse']for r in second[1:]];running=np.minimum.accumulate(vals);curves.append(running)
  average=np.mean(curves,axis=0);axes[0].plot(np.arange(9)*50,(1-average/average[0])*100,marker='.',label=v,color=color)
 axes[0].axvline(200,color='#999999',ls='--');axes[0].set_xlabel('Updates (stage 2 starts at 200)');axes[0].set_ylabel('Reduction of mean layer NMSE (%)');axes[0].set_title('Actual hard attention: best so far');axes[0].legend()
 ns=names[1:];axes[1].bar(ns,[quality[n]['lpips_mean']for n in ns],color=['#a5c3dc','#538fbc','#e4ba94','#c77b42']);axes[1].set_ylabel('LPIPS distance to Dense');axes[1].set_title('8 native Val videos; 36 sparse frames');axes[1].tick_params(axis='x',rotation=35)
 axes[2].bar(names,[timing[n]['generation_seconds']for n in names],color=['#666666','#a5c3dc','#538fbc','#e4ba94','#c77b42']);axes[2].set_ylabel('Generation seconds');axes[2].set_title('Same GPU: mean of prompt medians');axes[2].tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(root/'results.png',dpi=170);fig.savefig(root/'results.pdf');plt.close(fig)
 lines=['# Opt11：优化组合与实际 AV 对齐微调','', f"本轮只能确认有限回收：Compact 的平均 LPIPS 下降 {-improvements['Compact']['lpips_delta']:.6f}，Aggressive 下降 {-improvements['Aggressive']['lpips_delta']:.6f}，两组均只有 4/8 视频改善；额外 4 个 Val 上 Compact 均值略差，尚无稳定恢复质量的证据。激进版仍显著差于 Compact，微调未补回大部分额外损失。", '', '已完成两组完整优化路径、两阶段微调、8 个 Val 视频和同卡测速。微调对象是 29 层的 AV 动作头，backbone、prototype、投影和归一化冻结。最终 head 依据实际部署硬动作 attention NMSE 选择，包含初始 checkpoint 回退。','', '## 实际注意力误差：离线 Val','', '16 个 Train prompt 的 192 个事件用于微调；4 个选择-Val prompt 的全部 96 事件、每事件 256 个原始分组 query 用于选 best。下表为各层的事件均值再对 29 层平均，不能直接当作闭环视频质量。','','| 路径 | 微调前 NMSE | 1e-4 阶段 best | 1e-3 阶段 best | 总下降 | 改善层数 |','|---|---:|---:|---:|---:|---:|']
 for v in variants:
  d=offline[v];lines.append(f'| {v} | {d["initial_nmse"]:.8f} | {d["stage1_nmse"]:.8f} | {d["final_nmse"]:.8f} | {improvements[v]["offline_nmse_reduction"]*100:.3f}% | {d["improved_layers"]}/29 |')
 lines+=['', '第一阶段 200 updates、学习率 1e-4 的回收很小，因此又在各层各分支 best 上做 200 updates、学习率 1e-3 的检验。两阶段原始记录分别保留，最后选择第二阶段 best；其第 0 步就是上一阶段 best。选 best 用真正更新后的整组 mask、量化和在线跳过输出，不用训练 loss 最低代替。','', '## 闭环视频：实际是否拿回损失','', '固定 8 个 Val prompt、seed0，四个 native chunks × 四步、45 帧、480×832、16fps。LPIPS 统计后 36 个稀疏帧；首 chunk latent 和前 9 帧像素全部与 Dense 逐位一致。每种新输出另跑了一次完整轨迹验证一致性。','','| 路径 | LPIPS 前 | LPIPS 后 | 后−前（负值更接近 Dense） | 改善样例 | 保留率后 |','|---|---:|---:|---:|---:|---:|']
 for v in variants:
  b=quality[v+'Before'];d=quality[v+'After'];r=improvements[v];lines.append(f'| {v} | {b["lpips_mean"]:.6f} | {d["lpips_mean"]:.6f} | {r["lpips_delta"]:+.6f} | {r["lpips_improved_prompts"]}/8 | {d["keep_mean"]*100:.3f}% |')
 lines+=['', 'LPIPS 是对 Dense 输出的距离，不是绝对视觉质量评分。小幅离线误差下降不保证后续生成更接近 Dense；这里明确区分两者。', '', f"本轮重新生成的 CompactBefore 中，{quality['CompactBefore']['bit_identical_to_prior_opt10_prompts']}/8 与上一轮 Opt10 latent 逐位相同，其余存在输出差异，根因尚未单独定位。所有本轮 Before/After 都通过了各自重复轨迹校验；本文使用同一环境下重新生成的 Before 作配对基线，不混用上一轮 LPIPS。",'', '8 个视频中 4 个 prompt 用于 head 选择，另 4 个是额外 Val；均不是 Test：','','| 路径 | 选择-Val LPIPS 前→后 | 额外-Val LPIPS 前→后 |','|---|---:|---:|']
 for v in variants:
  b=quality[v+'Before'];d=quality[v+'After'];lines.append(f'| {v} | {b["head_selection_val"]["lpips"]:.6f} → {d["head_selection_val"]["lpips"]:.6f} | {b["additional_val"]["lpips"]:.6f} → {d["additional_val"]["lpips"]:.6f} |')
 lines+=['', '## 生成速度','', '每个 prompt 的 5 种模式在同一张 A100-SXM4-40GB 上预热后正反交替，各重复 3 次。下表为逐 prompt 中位数的均值；generation 包括原样上下文重建和 latent 回传，total 另包括 VAE/像素回传。首次编译不计入。160 个计时输出（含预热和 Dense 对照）全部与独立质量结果一致。','','| 模式 | 生成/s | 含 VAE/s |','|---|---:|---:|']
 for n in names:lines.append(f'| {n} | {timing[n]["generation_seconds"]:.3f} | {timing[n]["total_seconds"]:.3f} |')
 lines+=['', '微调保留相同推理结构，速度变化主要与产生的 mask 和测量波动有关；没有换回较慢的高精度 attention。','', '## 实现与边界','', '- Compact 包含 Opt10 已采用的路由/访存优化、FP16 头、tanh GELU、二阶矩统计、真实 P 评分及 INT8 QK。Aggressive 另加 Skip0 在线 PV 跳过和 FP16 PV。','- 原始 32-query 分组和 128-query 候选 hash 规则均保留；没有 Q64 mask 共享、没有 Fast 的去噪步复用。当前 P16，历史 P8/P4/P2，首 chunk/layer0 Dense，beta=1.09。','- 训练前向 logits 直接来自部署 Triton；反向用因式分解 FP16/tanh surrogate。实际优化 attention 在 no_grad 下产生标签；本轮没有为 attention/backbone 实现 backward。','- 标签是真实内核的单 query 条件动作误差：同组其它 31 行固定为原模型策略。训练 loss 是这些硬动作误差的概率加权期望，并非联合改变所有 query 后的精确策略期望。验证会按当前头更新全部 256 行并执行真实内核。','- 每个 query 仍只做一次增删/交换，因此只能在现有 81 个动作里找补偿。固定 prototype 与 backbone 的限制不能靠增加动作头训练步数保证消除。Train 条件 oracle 诊断见 `conditional_oracle_bounds.json`，它不是全局最优的证明。','- 第一阶段的 4 个选择-Val prompt 参与了 best 选择；最终 8 个视频又用于本轮开发评估，不能当作未接触的 Test。','', '## 文件与复现','', '[Compact：Dense / Before / After](ALL8_Dense_Compact_Before_After.mp4) · [Aggressive：Dense / Before / After](ALL8_Dense_Aggressive_Before_After.mp4)','', '[曲线和结果图](results.png) · [PDF](results.pdf) · [全部数字](final_summary.json) · [校验](final_verification.json) · [实验方案](PLAN.md)','', '[Compact 最终 head 参数](exports/Compact.best_heads.pt) · [Aggressive 最终 head 参数](exports/Aggressive.best_heads.pt)。仅包含动作头；加载时仍使用父实验的冻结 backbone/prototype。新入口 `migration.unified_router.opt11.network` 默认 `SLIMAV_OPT11=Aggressive`、`SLIMAV_OPT11_TRAINED=1`，读取 `strong_training/` 的 best；显式设置 TRAINED=0 可恢复对应微调前对照。','', f'已校验 {len(original["source"])} 个父版本源码和 {len(original["parameters"])} 个父参数文件不变。训练前向、两阶段脚本和最终评估源码封存在 `manifest.json`。','', '```bash','./v10_finetune11_progress.sh --once', 'SLIMAV_OPT11=Aggressive SLIMAV_OPT11_TRAINED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\', '  TORCHINDUCTOR_COMPILE_THREADS=2 streaming_sparse/.venv/bin/python -u \\', '  -m migration.runtime.bootstrap migration.unified_router.opt11.network \\', f'  --directory {root} --output {root}/reproduce_Aggressive --mode K32-Set','```','']
 (root/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(offline=offline,improvements=improvements,timing=timing),indent=2))
if __name__=='__main__':main()
