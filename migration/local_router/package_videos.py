"""Copy completed comparison videos into a portable local review bundle."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import time
import zipfile


TITLES = ['森林蓝色小兽', '红杉山路越野车', '水族馆海豹', '孟买雨中机器人',
          '东京列车窗景', '海岸教堂航拍', '约翰内斯堡机器人', '水龙头液滴']


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def package(experiment, previous=None):
    source = experiment / 'network_check/dense0/quality'
    review = experiment / 'video_review'
    (review / 'videos').mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['sparse_layers'] == list(range(1, 30)) and manifest['dense_layers'] == [0]
    cases = [r for f in source.glob('rank*/video_rows.json') for r in json.loads(f.read_text())]
    assert len({(r['prompt_id'], r['generation_seed']) for r in cases}) == len(cases)
    by_case = {(r['prompt_id'], r['generation_seed']): r for r in cases}
    records = []
    metrics = []
    cards = []
    for index, pid in enumerate(manifest['prompt_ids'], 1):
        title = TITLES[index - 1]
        for seed in manifest['generation_seeds']:
            row = by_case.get((pid, seed))
            label = f'{index:02d}_{title}_seed{seed}'
            body = '<p class="pending">生成中，刷新目录可查看新视频。</p>'
            if row is not None:
                paths = {}
                for group, prefix in [('main', 'paired_Dense_Klast_Vlast'),
                                      ('best', 'current_Dense_Kbest_Vbest')]:
                    original = source / 'videos' / f'{pid}.seed{seed}' / f'{prefix}.mp4'
                    filename = f'videos/{label}_{"Dense_Konly_Vaware" if group == "main" else "Dense_Kbest_Vbest"}.mp4'
                    dest = review / filename
                    if not dest.exists():
                        tmp = dest.with_suffix('.tmp')
                        shutil.copy2(original, tmp)
                        tmp.replace(dest)
                    assert original.stat().st_size == dest.stat().st_size
                    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
                    paths[group] = filename
                    records.append(dict(number=index, title=title, prompt_id=pid,
                        generation_seed=seed, group=group, file=filename, sha256=digest,
                        source=str(original), bytes=dest.stat().st_size))
                for mode in ['mass', 'Klast', 'Vlast', 'Kbest', 'Vbest']:
                    metrics.append(dict(number=index, title=title, prompt_id=pid,
                        generation_seed=seed, mode=mode,
                        **row['metrics'][mode]['quality']['sparse_frames_only']))
                main = html.escape(paths['main'], quote=True)
                best = html.escape(paths['best'], quote=True)
                body = (f'<p>左 Dense · 中 K-only last · 右 V-aware last</p>'
                    f'<video controls preload="none" src="{main}"></video>'
                    f'<p><a href="{main}">打开 / 下载主对照</a></p>'
                    f'<details><summary>最佳权重对照：左 Dense · 中 Kbest · 右 Vbest</summary>'
                    f'<video controls preload="none" src="{best}"></video>'
                    f'<p><a href="{best}">打开 / 下载最佳权重对照</a></p></details>')
            cards.append(f'<section><h2>{index:02d} · {html.escape(title)} · seed {seed}</h2>{body}</section>')
    state = json.loads((experiment / 'status.json').read_text())
    complete = len(cases) == 16 and state['status'] == 'completed' and (source / 'END_TO_END_REPORT.md').exists()
    info = dict(updated_at=datetime.now(timezone.utc).isoformat(), status='completed' if complete else 'generating',
        completed_cases=len(cases), expected_cases=16, video_count=len(records),
        sparse_layers=manifest['sparse_layers'], dense_layers=[0], history_keep=.2,
        frames=45, fps=16, resolution=[480, 832], first9_frames_identical=True, videos=records)
    write_json(review / 'manifest.json', info)
    if metrics:
        with (review / 'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    shutil.copy2(source / 'manifest.json', review / 'evaluation_manifest.json')
    shutil.copy2(experiment / 'assembled.json', review / 'assembled.json')
    for name in ['REPORT.md', 'END_TO_END_REPORT.md', 'end_to_end_comparison.png', 'end_to_end_comparison.pdf',
                 'end_to_end_assessment.json', 'video_comparison.csv']:
        path = source / name
        if path.exists():
            if path.suffix == '.md':
                text = path.read_text()
                for record in records:
                    original = Path(record['source'])
                    text = text.replace(str(original.relative_to(source)), record['file'])
                text = text.replace('visual_checks/first_prompt_seed0.png', 'first_prompt_seed0.png')
                text = text.replace('../../../assembled.json', 'assembled.json')
                text = text.replace('manifest.json', 'evaluation_manifest.json')
                (review / name).write_text(text)
            else:
                shutil.copy2(path, review / name)
    preview = source / 'visual_checks/first_prompt_seed0.png'
    if preview.exists():
        shutil.copy2(preview, review / 'first_prompt_seed0.png')
    (review / 'README.txt').write_text(
        f'端到端视频对照：{len(cases)}/16组，{len(records)}个MP4。\n'
        '打开 index.html 观看；视频文件也可直接用播放器打开。\n'
        '主对照：左Dense，中K-only末尾权重，右V-aware末尾权重。\n'
        '最佳权重对照：左Dense，中Kbest，右Vbest。Kbest与Klast张量相同。\n'
        '所有稀疏分支：第0层Dense，第1–29层全部12Heads稀疏；历史保留20%，当前段完整保留。\n'
        '每个视频480×832×3横向拼接、45帧、16fps；首9帧相同。\n'
        'metrics.csv记录后36帧相对同噪声Dense的PSNR/LPIPS，来自压缩前RGB。\n'
        '文件按固定Prompt顺序和seed编号，未按效果筛选。\n')
    report_links = ('<p><a href="END_TO_END_REPORT.md">完整评估报告</a> · '
                    '<a href="end_to_end_comparison.png">质量曲线</a> · '
                    '<a href="end_to_end_comparison.pdf">曲线 PDF</a></p>'
                    if (review / 'END_TO_END_REPORT.md').exists() else '')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>29层稀疏 · 视频对照</title><style>'
        'body{font:16px/1.6 system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1440px;color:#182330;background:#f4f6f9}'
        'h1{margin-bottom:8px}h2{font-size:20px}section{background:white;padding:20px;border-radius:12px;margin:20px 0}'
        'video{width:100%;background:#111;display:block}a{color:#165ca6}summary{cursor:pointer}.pending{color:#68778a}'
        '</style><h1>第0层 Dense · 第1–29层稀疏</h1>'
        f'<p>已完成 {len(cases)}/16 组 Prompt / seed，{len(records)} 个视频。'
        '历史保留20%；45帧、16fps。主对照：左 Dense、中 K-only、右 V-aware。</p>'
        '<p>首9帧相同，质量指标主要比较后36帧。视频保留全部案例，按固定顺序排列。'
        '<a href="metrics.csv">逐视频指标 CSV</a> · <a href="manifest.json">文件清单</a></p>'
        + report_links + ''.join(cards) + '</html>')
    (review / 'index.html').write_text(page)
    key = (len(cases), complete)
    if key != previous:
        dest = experiment / 'video_comparison_bundle.zip'
        tmp = dest.with_suffix('.tmp')
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(review.rglob('*')):
                if path.is_file():
                    archive.write(path, 'video_review/' + str(path.relative_to(review)))
        tmp.replace(dest)
        print(json.dumps(dict(cases=len(cases), videos=len(records), complete=complete,
            directory=str(review), bundle=str(dest)), ensure_ascii=False), flush=True)
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    experiment = Path(args.directory).resolve()
    previous = None
    while True:
        previous = package(experiment, previous)
        if previous[1] or not args.watch:
            return
        state = json.loads((experiment / 'status.json').read_text())
        if state['status'] in ('failed', 'paused'):
            raise RuntimeError(f'Evaluation interrupted: {state}')
        time.sleep(30)


if __name__ == '__main__':
    main()
