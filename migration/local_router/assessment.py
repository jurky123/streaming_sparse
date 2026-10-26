"""Summarize the complete Dense0 / sparse1-29 end-to-end video experiment."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from streaming_sparse.v08.common import atomic_json, source_identity
from streaming_sparse.v08.metrics import summarize
from migration.frozen_quality.report import paired


MODES = ['mass', 'Klast', 'Vlast', 'Kbest', 'Vbest']


def assess(directory):
    experiment = Path(directory).resolve()
    root = experiment / 'network_check/dense0/quality'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['source_sha256'] == source_identity()
    assert manifest['sparse_layers'] == list(range(1, 30))
    assert manifest['dense_layers'] == [0] and manifest['keep'] == .2
    assert manifest['checkpoints']['Kbest']['router_sha256'] == manifest['checkpoints']['Klast']['router_sha256']
    expected = {(p, s) for p in manifest['prompt_ids'] for s in manifest['generation_seeds']}
    cases = [row for rank in range(8)
             for row in json.loads((root / f'rank{rank}/video_rows.json').read_text())]
    assert len(cases) == len(expected) == 16
    assert {(r['prompt_id'], r['generation_seed']) for r in cases} == expected
    for rank in range(8):
        status = json.loads((root / f'rank{rank}/status.json').read_text())
        assert status['status'] == 'completed'
        assert status['backbone_versions_unchanged'] and status['routers_unchanged']
    offline = {}
    for mode in ['Dense', *MODES]:
        rows = [r for rank in range(8)
                for r in json.loads((root / f'rank{rank}/{mode}_offline.json').read_text())['rows']]
        assert len(rows) == 192
        assert {(r['prompt_id'], r['generation_seed'], r['history_tokens'], r['step']) for r in rows} == {
            (p, s, h, step) for p, s in expected for h in (4680, 9360, 14040) for step in range(4)}
        offline[mode] = summarize(rows)
    video = {}
    flat = []
    for mode in MODES:
        counts = []
        qualities = []
        for case in cases:
            data = case['metrics'][mode]
            quality = data['quality']
            assert len(quality['per_frame']) == 45
            assert all(r['identical'] for r in quality['per_frame'][:9])
            assert len(data['sparsity']) == 12
            assert len(data['diagnostics']) == 16
            assert {(c['history_tokens'], c['step']) for c in data['sparsity']} == {
                (h, step) for h in (4680, 9360, 14040) for step in range(4)}
            for item in data['sparsity']:
                assert item['routed_queries'] == 29 * 12 * 4680
                assert item['history_tokens'] in (4680, 9360, 14040)
                assert item['keep'] <= .2 + 1e-9
            counts.extend(data['sparsity'])
            qualities.append(quality)
            value = quality['sparse_frames_only']
            flat.append(dict(mode=mode, prompt_id=case['prompt_id'],
                             generation_seed=case['generation_seed'], **value))
        item = {}
        for scope in ['full', 'sparse_frames_only']:
            mse = float(np.mean([q[scope]['mse_rgb01'] for q in qualities]))
            item[scope] = dict(mse_rgb01=mse, psnr_db=-10 * math.log10(mse),
                               lpips=float(np.mean([q[scope]['lpips'] for q in qualities])))
        item['chunks'] = []
        for i in range(4):
            mse = float(np.mean([q['chunks'][i]['mse_rgb01'] for q in qualities]))
            item['chunks'].append(dict(chunk=i + 1, mse_rgb01=mse,
                psnr_db=-10 * math.log10(mse) if mse else None,
                lpips=float(np.mean([q['chunks'][i]['lpips'] for q in qualities]))))
        item['prompts'] = {p: float(np.mean([
            c['metrics'][mode]['quality']['sparse_frames_only']['lpips']
            for c in cases if c['prompt_id'] == p])) for p in manifest['prompt_ids']}
        item['historical_connection_keep'] = (
            sum(c['retained_query_tokens'] for c in counts) /
            sum(c['routed_queries'] * c['history_tokens'] for c in counts))
        video[mode] = item
    comparisons = {
        'last_V_minus_K': paired(video['Klast']['prompts'], video['Vlast']['prompts']),
        'best_V_minus_K': paired(video['Kbest']['prompts'], video['Vbest']['prompts']),
    }
    mass_comparisons = {mode: paired(video['mass']['prompts'], video[mode]['prompts'])
                        for mode in ['Klast', 'Vlast']}
    for value in mass_comparisons.values():
        value['delta_candidate_minus_mass'] = value.pop('delta_V_minus_K')
    mass = video['mass']['sparse_frames_only']['lpips']
    for mode in MODES[1:]:
        video[mode]['lpips_relative_improvement_vs_mass'] = (
            1 - video[mode]['sparse_frames_only']['lpips'] / mass)
    result = dict(status='completed', verified_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_identity(), manifest=manifest, video=video, offline=offline,
        paired_lpips=comparisons, paired_vs_mass=mass_comparisons, video_cases=16, all29_layers_routed=True,
        first9_frames_pixel_identical=True, test_used=False)
    merged = json.loads((root / 'summary.json').read_text())
    for mode in MODES:
        assert math.isclose(video[mode]['sparse_frames_only']['lpips'],
                            merged['videos'][mode]['lpips'], rel_tol=1e-12)
        assert math.isclose(video[mode]['sparse_frames_only']['psnr_db'],
                            merged['videos'][mode]['psnr_db'], rel_tol=1e-12)
    atomic_json(root / 'end_to_end_assessment.json', result)
    with (root / 'video_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot(root, video, comparisons)
    write_report(root, result, cases)
    print(json.dumps(dict(status='completed', cases=16,
        report=str(root / 'END_TO_END_REPORT.md'))), flush=True)


def plot(root, video, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = ['mass', 'Klast', 'Vlast', 'Vbest']
    labels = ['Initial Mass', 'K-only last/best', 'V-aware last', 'V-aware best']
    colors = ['#8d99ae', '#277da8', '#e99635', '#66a061']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, metric, title in [(axes[0, 0], 'psnr_db', 'PSNR vs Dense (higher is better)'),
                              (axes[0, 1], 'lpips', 'LPIPS vs Dense (lower is better)')]:
        values = [video[m]['sparse_frames_only'][metric] for m in modes]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt='%.4f', padding=3)
        ax.set_title(title)
        ax.tick_params(axis='x', labelrotation=14)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    ax = axes[1, 0]
    for mode, label, color in zip(modes, labels, colors):
        ax.plot(range(1, 5), [v['lpips'] for v in video[mode]['chunks']],
                marker='o', label=label, color=color)
    ax.set(title='Closed-loop drift by chunk', xlabel='Chunk (3 latent frames)', ylabel='LPIPS')
    ax.set_xticks(range(1, 5))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    delta = comparisons['last_V_minus_K']['per_prompt_delta']
    keys = sorted(delta)
    ax.barh([p[:8] for p in keys], [delta[p] for p in keys],
            color=['#66a061' if delta[p] < 0 else '#ce6262' for p in keys])
    ax.axvline(0, color='black', linewidth=.8)
    ax.set(title='Paired LPIPS: V-last minus K-last', xlabel='Negative favors V-aware', ylabel='Prompt ID')
    ax.grid(axis='x', alpha=.2)
    fig.suptitle('Dense layer 0; sparse layers 1–29 | 20% history | 8 prompts × 2 seeds\n'
                 'Video averages exclude the identical first 9 frames; 45 frames total', fontsize=12)
    fig.savefig(root / 'end_to_end_comparison.png', dpi=180)
    fig.savefig(root / 'end_to_end_comparison.pdf')
    plt.close(fig)


def write_report(root, result, cases):
    v = result['video']
    lines = ['# 第0层Dense、第1–29层稀疏：端到端视频质量', '',
        '已完成8个Val Prompt×2个生成Seed，共16组闭环视频；480×832、45帧、16fps。'
        '每个分支独立生成自己的历史，执行全部4个原生去噪step；骨干与LoRA冻结。', '',
        '各稀疏分支全部29层×12个Heads、4680个当前Query均实际参与路由；历史预算20%、Page64，'
        '当前chunk完整保留，context-write保持Dense。首chunk的9帧已逐像素验证一致，主比较统计后36帧。', '',
        'PSNR和LPIPS衡量与相同Prompt、相同噪声Dense视频的差异；指标来自压缩前RGB。'
        'PSNR由合并MSE换算，LPIPS为AlexNet v0.1；它们不能单独代表审美或文本一致性。', '',
        '| 模型 | 后36帧PSNR↑ | 后36帧LPIPS↓ | LPIPS相对Mass降低 | 全45帧LPIPS↓ | x0 NMSE↓ |',
        '|---|---:|---:|---:|---:|---:|',
        '| Dense参考 | ∞ | 0 | — | 0 | 0 |']
    for mode in MODES:
        s = v[mode]['sparse_frames_only']
        gain = v[mode].get('lpips_relative_improvement_vs_mass')
        gain_text = '—' if gain is None else f'{gain:.2%}'
        lines.append(f"| {mode} | {s['psnr_db']:.4f} | {s['lpips']:.6f} | {gain_text} | "
                     f"{v[mode]['full']['lpips']:.6f} | {result['offline'][mode]['prompt_equal_error']:.9f} |")
    lines += ['', 'Mass是本轮共同初始Router，残差评分头为零：原4/14/24层迁移旧校准权重，'
              '其余26层全新初始化，各层使用固定Train归一化统计。与Mass比较反映本轮整体训练收益；'
              'Value信息的增益应看同预算K/V配对，不能把整体训练收益全部归因于Value残差头。', '',
              'Kbest与Klast的全部Router张量相同；Vbest与Vlast仅按各层局部Val选择，'
              '没有按本轮视频指标重新挑选权重。各层实际停止点为350–800更新，'
              'checkpoint上的800表示最大层计数，不能解释为所有层均训练了800步。', '',
              '## 配对比较', '']
    for name, d in result['paired_lpips'].items():
        better = sum(x < 0 for x in d['per_prompt_delta'].values())
        lines.append(f"- {name}：LPIPS差 {d['delta_V_minus_K']:.7f}，Prompt配对bootstrap "
                     f"95%CI {d['prompt_paired_95CI']}；V在{better}/8个Prompt均值上更低。")
    for name, d in result['paired_vs_mass'].items():
        lines.append(f"- {name}−Mass：LPIPS差 {d['delta_candidate_minus_mass']:.7f}，Prompt配对bootstrap "
                     f"95%CI {d['prompt_paired_95CI']}；负值表示训练后更接近Dense。")
    lines += ['', '同一个Prompt的两个生成Seed先取均值，再以Prompt为单位重采样5000次。'
              '仅一个Router训练Seed，使用开发Val，封存Test未访问。', '',
              '## 随历史累积的误差', '',
              '| 模型 | chunk1 LPIPS | chunk2 LPIPS | chunk3 LPIPS | chunk4 LPIPS |',
              '|---|---:|---:|---:|---:|']
    for mode in MODES:
        lines.append('| ' + mode + ' | ' + ' | '.join(f"{c['lpips']:.6f}" for c in v[mode]['chunks']) + ' |')
    lines += ['', f"闭环历史连接加权保留率：{v['Klast']['historical_connection_keep']:.4%}。"
              '这是实际选页预算检查，不能当作实测加速比。', '',
              '![端到端对照](end_to_end_comparison.png)', '']
    if (root / 'visual_checks/first_prompt_seed0.png').exists():
        lines += ['固定排序第一个Prompt、seed0的逐chunk帧对照，左Dense、中K-only、右V-aware；'
                  '该样例按Prompt顺序选取，不能代表全部16组的平均表现。', '',
                  '![固定样例逐chunk对照](visual_checks/first_prompt_seed0.png)', '']
    lines += ['## 全部视频', '',
              '每行均为固定Prompt/Seed，未按效果挑选；左右标签写入视频。', '']
    for row in sorted(cases, key=lambda r: (r['prompt_id'], r['generation_seed'])):
        case = f"{row['prompt_id']}.seed{row['generation_seed']}"
        lines.append(f'- {case}：[Dense / Klast / Vlast](videos/{case}/paired_Dense_Klast_Vlast.mp4)；'
                     f'[Dense / Kbest / Vbest](videos/{case}/current_Dense_Kbest_Vbest.mp4)')
    lines += ['', '原始指标：video_comparison.csv；完整分析：end_to_end_assessment.json；'
              '权重及数据身份：manifest.json；各层更新数：../../../assembled.json。', '']
    (root / 'END_TO_END_REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    directory = Path(args.directory)
    if args.watch:
        while True:
            state = json.loads((directory / 'status.json').read_text())
            if state.get('status') == 'completed':
                break
            if state.get('status') in ('failed', 'paused'):
                raise RuntimeError(f'Evaluation did not complete: {state}')
            time.sleep(30)
    assess(directory)


if __name__ == '__main__':
    main()
