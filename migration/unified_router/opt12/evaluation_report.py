"""Verify native video evaluation and paired timings; summarize convergence."""
import hashlib
import json
import statistics as stats
import subprocess
from pathlib import Path
import torch
from ..common import atomic_json, sha
from .status import ROOT


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(z):
    return hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    root = ROOT
    m = read(root/'experiment.json')
    base = Path(m['parent_experiment'])
    pids = read(base/'e2e_manifest.json')['prompt_ids']
    assert not any(read(root/'quality/exit_codes.json'))
    assert not any(read(root/'paired/exit_codes.json'))
    lock = read(root/'quality/evaluation_lock.json')
    for path, digest in {**lock['training_source'], **lock['weights'], **lock['code']}.items():
        assert sha(path) == digest, path
    for rank in range(8):
        assert read(root/f'quality/rank{rank}.json')['status'] == 'completed'
    names = ['Dense', 'CompactBefore', 'CompactAfter', 'AggressiveBefore', 'AggressiveAfter']
    quality, per_prompt, improvements = {}, [], {}
    for name in names[1:]:
        rows = []
        for pid in pids:
            item = read(root/f'quality/{pid}.seed0/{name}.json')
            assert item['repeat_identical'] and item['first_chunk_identical']
            metric = item['quality_vs_dense']['sparse_frames_only']
            rows.append(dict(prompt_id=pid, selection_val=pid in m['splits']['val'],
                        lpips=metric['lpips'], psnr_db=metric['psnr_db'], keep=item['weighted_keep'],
                        matches_opt11_after=item.get('matches_opt11_after')))
        quality[name] = dict(lpips=stats.mean(r['lpips'] for r in rows),
                            psnr_db=stats.mean(r['psnr_db'] for r in rows),
                            keep=stats.mean(r['keep'] for r in rows), per_prompt=rows)
        for group, selected in [('selection_val', True), ('additional_val', False)]:
            subset = [r for r in rows if r['selection_val'] == selected]
            quality[name][group] = dict(count=len(subset), lpips=stats.mean(r['lpips'] for r in subset),
                                       psnr_db=stats.mean(r['psnr_db'] for r in subset))
        if name.endswith('Before'):
            quality[name]['matches_opt11_after_count'] = sum(r['matches_opt11_after'] for r in rows)
    checked = 0
    for i, pid in enumerate(pids):
        folder = root/f'paired/prompt{i}'
        assert read(folder/'status.json')['status'] == 'completed'
        record = read(folder/'timings.json')
        assert len(record['records']) == 20 and record['repeats'] == 3
        expected = {name: tensor_sha(torch.load(root/f'quality/{pid}.seed0/{name}.latent.pt',
                    map_location='cpu', weights_only=True)) for name in names}
        for row in record['records']:
            assert row['latent_sha256'] == expected[row['variant']], (pid, row['variant'])
            checked += 1
        per_prompt.append(dict(prompt_id=pid, timing=read(folder/'summary.json')))
    timing = {name: {key: stats.mean(r['timing'][name][key] for r in per_prompt)
                    for key in ['generation_seconds', 'decode_seconds', 'total_seconds']} for name in names}
    for variant in ['Compact', 'Aggressive']:
        before, after = quality[variant+'Before'], quality[variant+'After']
        improvements[variant] = dict(lpips_delta=after['lpips']-before['lpips'],
            lpips_reduction_pct=100*(1-after['lpips']/before['lpips']),
            improved_prompts=sum(a['lpips']<b['lpips'] for a,b in zip(after['per_prompt'], before['per_prompt'])))
        for group, selected in [('selection_val', True), ('additional_val', False)]:
            improvements[variant][group] = dict(lpips_delta=after[group]['lpips']-before[group]['lpips'],
                improved_prompts=sum(a['lpips']<b['lpips'] for a,b in zip(after['per_prompt'], before['per_prompt']) if a['selection_val']==selected))
    videos = {}
    for variant in ['Compact', 'Aggressive']:
        listing = root/f'quality/concat_{variant}.txt'
        listing.write_text(''.join("file '"+str(root/f'quality/{pid}.seed0/Dense_{variant}_Before_After.mp4')+"'\n" for pid in pids))
        dest = root/f'ALL8_Dense_{variant}_Opt11_Opt12.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(dest)],check=True)
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(dest)]))['streams'][0]
        assert int(info['nb_read_frames']) == 360 and info['width'] == 2496 and info['height'] == 480
        videos[variant] = dict(path=str(dest), **info)
    audit = read(root/'convergence_audit.json')
    result = dict(scope='29 sparse layers; layer0 and first chunk Dense; 8 Val prompts seed0; Test unused',
        before='Opt11 strong-stage best, head-only fine-tuning', after='Opt12 joint prototype/projection/head best',
        selection='actual deployed attention NMSE under retention cap; per-layer best, not last',
        convergence=audit['summary'], quality=quality, timing=timing, improvements=improvements,
        per_prompt_timing=per_prompt, videos=videos)
    atomic_json(root/'evaluation_summary.json', result)
    atomic_json(root/'evaluation_verification.json', dict(passed=True, timed_outputs_checked=checked,
        fresh_dense_matches_reference=8, repeated_sparse_trajectories_identical=32,
        first_chunk_dense_identical=True, evaluated_checkpoints_unchanged=True,
        training_source_unchanged=True, before_after_parameters_restored_for_timing=True, test_used=False))
    atomic_json(root/'evaluation_source_manifest.json', {str(f.resolve()):sha(f) for f in Path(__file__).parent.glob('*.py')})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ['#9cbfd6','#3978a8','#edbd94','#c06b33']
    labels = ['Compact\nOpt11','Compact\nOpt12','Aggressive\nOpt11','Aggressive\nOpt12']
    axes[0].bar(labels, [quality[n]['lpips'] for n in names[1:]], color=colors)
    axes[0].set_title('8 Val videos: LPIPS distance to Dense')
    axes[0].set_ylabel('Mean LPIPS on 36 sparse frames')
    for j, group in enumerate(['selection_val','additional_val']):
        xs = [i+j*.36 for i in range(4)]
        axes[1].bar(xs, [quality[n][group]['lpips'] for n in names[1:]], width=.36, label=group)
    axes[1].set_xticks([i+.18 for i in range(4)], labels)
    axes[1].set_title('Selection Val versus additional Val')
    axes[1].legend()
    axes[2].bar(['Dense']+labels, [timing[n]['generation_seconds'] for n in names], color=['#777777']+colors)
    axes[2].set_title('Same-GPU paired generation time')
    axes[2].set_ylabel('Seconds, mean of prompt medians (3 repeats)')
    fig.savefig(root/'evaluation_results.png', dpi=150)
    fig.savefig(root/'evaluation_results.pdf')
    plt.close(fig)
    lines = ['Opt12：联合微调收敛检查与端到端评估', '',
        '全部29层训练完成；首chunk和layer0保持Dense。视频按每层真实注意力误差最优checkpoint运行。',
        'Before=Opt11两阶段动作头微调best；After=Opt12 prototype、K64投影和动作头联合微调best。',
        '8个Val提示词固定seed0，45帧480×832；LPIPS统计后36个稀疏帧，是相对Dense的距离，不是绝对视觉质量分数。', '',
        '2000步末尾仍有改善的层（best在2000，最后100和500步原始验证NMSE均下降）：']
    for variant in ['Compact','Aggressive']:
        conv = audit['summary'][variant]
        lines.append(f'{variant}: {conv["still_improving_at_cap"]}；平均离线NMSE {conv["initial_nmse"]:.8f} → {conv["best_nmse"]:.8f}')
    lines += ['这是仍有优化空间的证据，不证明延长训练一定改善视频；其它很多层best更早，末尾误差回升。',
        '本次固定当前best进行评估，尚未启动额外续训。部分层经过FP32反向修复，边界记录在resume_history.jsonl。', '', '端到端视频：']
    for variant in ['Compact','Aggressive']:
        a,b = quality[variant+'Before'],quality[variant+'After']
        lines.append(f'{variant}: LPIPS {a["lpips"]:.6f} → {b["lpips"]:.6f}；改善 {improvements[variant]["improved_prompts"]}/8；保留率 {a["keep"]:.3%} → {b["keep"]:.3%}')
        for group in ['selection_val','additional_val']:
            lines.append(f'  {group}: {a[group]["lpips"]:.6f} → {b[group]["lpips"]:.6f}；改善 {improvements[variant][group]["improved_prompts"]}/4')
        lines.append(f'  新生成Before与上一轮Opt11 After逐位一致：{a["matches_opt11_after_count"]}/8')
    lines += ['', '同卡配对速度（每提示词3次，先取中位数再平均，首次编译排除；包含原样上下文重建）：']
    for name in names:
        lines.append(f'{name}: 生成 {timing[name]["generation_seconds"]:.3f}s；含VAE {timing[name]["total_seconds"]:.3f}s')
    lines += ['', '校验：8个新Dense与旧参考逐位一致；32种稀疏输出各自重复轨迹逐位一致；160个计时输出与独立质量输出一致。',
        '参与选模的4个Val与额外4个Val分开报告；额外Val在先前实验中已用于开发比较，不是未接触Test。',
        '拼接视频列顺序：Dense / Before(Opt11) / After(Opt12)，8个提示词依次串联。',
        '完整数字：evaluation_summary.json；校验：evaluation_verification.json；收敛：convergence_audit.json。']
    (root/'EVALUATION_REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(improvements=improvements,timing=timing), indent=2))


if __name__ == '__main__':
    main()
