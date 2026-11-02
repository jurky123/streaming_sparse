"""CPU-only reaggregation of saved pre-encoding RGB metrics; no inference edits."""
import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

METHODS = ['dense', 'dense_host', 'mass_var', 'route_fast', 'scatter_fast', 'pilot_anchor', 'pilot_candidate', 'pilot_swap']
WINDOWS = {'pre_sparse': (0, 21), 'early_primary': (21, 101), 'next_5s': (101, 181), 'next_10s': (181, 261), 'tail_diagnostic': (400, 480)}

def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value)
    os.replace(tmp, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/rollout_improve_20261030'))
    args = parser.parse_args()
    rows, sources = [], {}
    for folder in sorted((args.root / 'acceptance').iterdir()):
        if not folder.is_dir():
            continue
        for method in METHODS:
            p = folder / f'{method}.quality.json'
            if not p.exists() or not (folder / f'{method}.rep0.json').exists():
                continue
            raw = p.read_bytes()
            data = json.loads(raw)['per_frame']
            for key in ['mse', 'lpips', 'ssim']:
                assert len(data[key]) == 480
                assert all(math.isfinite(x) for x in data[key])
            sources[str(p)] = hashlib.sha256(raw).hexdigest()
            result = {}
            for name, (start, end) in WINDOWS.items():
                mse = statistics.mean(data['mse'][start:end])
                result[name] = dict(lpips=statistics.mean(data['lpips'][start:end]), ssim=statistics.mean(data['ssim'][start:end]), mse=mse, psnr_db=100. if mse == 0 else -10 * math.log10(mse))
            rows.append(dict(condition=folder.name, method=method, windows=result))
    conditions = sorted({r['condition'] for r in rows})
    common = [c for c in conditions if all(any(r['condition'] == c and r['method'] == m for r in rows) for m in METHODS)]
    aggregates = {}
    for method in METHODS:
        group = [r for r in rows if r['condition'] in common and r['method'] == method]
        if group:
            aggregates[method] = {w: {k: statistics.mean(r['windows'][w][k] for r in group) for k in ['lpips', 'ssim', 'psnr_db']} for w in WINDOWS}
    doc = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol='User-requested early-quality reanalysis; start chosen from runtime activation, not metric values. Zero-based half-open frame ranges. 16fps. Original uncompressed RGB per-frame metrics; no MP4 remeasurement. One quality sample per condition/method, timing repeats are not independent quality samples. PSNR averages are per-condition dB means.', windows=WINDOWS, complete_quality_conditions=common, rows=rows, aggregates=aggregates, source_sha256=sources)
    write(args.root / 'early_quality.json', json.dumps(doc, indent=2))
    lines = ['# 前段质量重新评估', '', f"更新时间：{doc['created_utc']}。共有 {len(common)} 组完整质量条件，{len(rows)} 份已提交质量结果。", '', '主指标采用首次实际稀疏选择后的5秒：0起始帧区间 [21,101)，播放时间 [1.3125,6.3125) 秒。runtime在历史页数 p>72 时才选择；chunk 0/1不丢页，chunk 2有144页历史并选72页。VAE首chunk输出9帧，后续每chunk12帧，因此第三chunk起始为第21帧。该起点由实现决定，未根据结果挑选。', '', '直接重汇总压制前RGB已有逐帧指标，不重读有损视频，不运行GPU。LPIPS AlexNet和SSIM沿用256×384分辨率，MSE/PSNR沿用512×768。PSNR先平均窗口MSE再转dB；跨条件表为各条件dB等权平均。旧全片报告保留为历史口径，本报告是新的主要质量口径。', '', '只对所有8种方法均已有首轮质量的条件计算总平均；不要求完成3次速度重复，不把重复计时当质量样本。尚未齐全的条件仅列明细。前段窗口仍包含自回归反馈，并非单步因果误差；后段LPIPS只保留为诊断，不作为主验收门槛。不自动套用旧全片阈值。', '', '## 完整条件等权平均', '', '| 方法 | 前段LPIPS↓ | 前段SSIM↑ | 前段PSNR dB↑ | 随后5秒LPIPS | 再后5秒LPIPS |', '|---|---:|---:|---:|---:|---:|']
    for m, a in aggregates.items():
        e = a['early_primary']
        lines.append(f"| {m} | {e['lpips']:.6f} | {e['ssim']:.6f} | {e['psnr_db']:.3f} | {a['next_5s']['lpips']:.6f} | {a['next_10s']['lpips']:.6f} |")
    lines += ['', '完整条件：' + '、'.join(common), '', '## 每条件前段指标', '', '| 条件 | 方法 | LPIPS↓ | SSIM↑ | PSNR dB↑ | 稀疏前LPIPS（诊断） |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        e = r['windows']['early_primary']
        lines.append(f"| {r['condition']} | {r['method']} | {e['lpips']:.6f} | {e['ssim']:.6f} | {e['psnr_db']:.3f} | {r['windows']['pre_sparse']['lpips']:.8f} |")
    lines += ['', '稀疏前诊断不强制为零：分组重排及数值路径可能已引入微小差异。所有窗口和源文件SHA256见 [early_quality.json](early_quality.json)。', '', '重新执行（CPU）：`python3 -m local_experiments.improve_early_quality`。不修改正在运行的生成程序或旧指标。']
    write(args.root / 'early_quality.zh.md', '\n'.join(lines) + '\n')
    print(json.dumps(dict(complete_conditions=common, aggregates=aggregates), indent=2))

if __name__ == '__main__':
    main()
