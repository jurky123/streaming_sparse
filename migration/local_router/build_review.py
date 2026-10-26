"""Build a small, allowlisted v0.8 review archive with no model/data payloads."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_file(name):
    p = ROOT.joinpath(*name.split('.'))
    for candidate in [p.with_suffix('.py'), p / '__init__.py']:
        if candidate.is_file():
            return candidate
    return None


def source_closure(seeds):
    pending = list(seeds)
    seen = set()
    while pending:
        path = Path(pending.pop())
        if path in seen:
            continue
        seen.add(path)
        parts = list(path.relative_to(ROOT).with_suffix('').parts)
        package = '.'.join(parts[:-1])
        for i in range(1, len(parts)):
            init = ROOT.joinpath(*parts[:i], '__init__.py')
            if init.is_file() and init not in seen:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    base = importlib.util.resolve_name('.' * node.level + base, package)
                names = [base] + [base + '.' + a.name for a in node.names if a.name != '*']
            for name in names:
                target = module_file(name)
                if target is not None and target not in seen:
                    pending.append(target)
    return sorted(seen)


def build(experiment, destination):
    experiment = Path(experiment).resolve()
    quality = experiment / 'network_check/dense0/quality'
    assert json.loads((experiment / 'status.json').read_text())['status'] == 'completed'
    assessment = json.loads((quality / 'end_to_end_assessment.json').read_text())
    assert assessment['video_cases'] == 16 and assessment['all29_layers_routed']
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    evidence = {}

    def copy(source, name, category):
        source = Path(source)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        assert source.is_file() and not source.is_symlink(), source
        shutil.copy2(source, target)
        evidence[name] = dict(category=category, source=str(source.relative_to(ROOT)),
                              source_sha256=sha(source), transformed=False)

    def write(name, value, category, source=None):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        evidence[name] = dict(category=category, transformed=source is not None)
        if source is not None:
            evidence[name].update(source=str(Path(source).relative_to(ROOT)), source_sha256=sha(source))

    seeds = list((ROOT / 'streaming_sparse/v08').glob('*.py'))
    for folder in ['streaming_sparse/v08/execution', 'streaming_sparse/v08/launcher',
                   'migration/local_router', 'migration/runtime']:
        seeds += list((ROOT / folder).glob('*.py'))
    for name in ['migration/paired_router/runtime.py', 'migration/paired_router/quality.py',
                 'migration/paired_router/evaluation.py', 'migration/frozen_quality/worker.py',
                 'migration/frozen_quality/report.py']:
        seeds.append(ROOT / name)
    sources = source_closure(seeds)
    for source in sources:
        copy(source, 'code/' + str(source.relative_to(ROOT)), 'source')
    for name in ['streaming_sparse/v08/configs/protocol.json', 'configs/default_config.yaml',
                 'configs/self_forcing_dmd.yaml', 'wan_models/Wan2.1-T2V-1.3B/config.json']:
        copy(ROOT / name, 'code/' + name, 'configuration')
    copy(ROOT / 'LICENSE', 'LICENSE', 'license')
    copy(ROOT / 'streaming_sparse/v08/README.txt', 'context/v08_original_README.txt', 'historical_context')
    for name in ['requirements.lock.txt', 'environment.freeze.txt', 'model_revisions.json']:
        copy(ROOT / 'migration' / name, 'metadata/environment/' + name, 'environment')
    copy(ROOT / 'requirements.txt', 'metadata/environment/base_requirements.txt', 'environment')
    original = experiment / 'experiment.json'
    config = json.loads(original.read_text())
    rows = config.pop('rows')
    config['sample_counts'] = {s: sum(r['split'] == s for r in rows) for s in ['train', 'val']}
    config['omitted_fields'] = ['rows (training/validation replay input manifests and query-index lists)']
    write('metadata/experiment_config.json', config, 'configuration', original)
    registry = ROOT / 'streaming_sparse/v08/configs/prompt_registry.json'
    reg = json.loads(registry.read_text())
    write('metadata/evaluation_prompts.json', {
        'registry_sha256': sha(registry), 'prompt_ids': assessment['manifest']['prompt_ids'],
        'texts': {p: reg['texts'][p] for p in assessment['manifest']['prompt_ids']},
        'training_and_test_prompt_texts': 'excluded'}, 'evaluation_protocol', registry)
    for name in ['assembled.json', 'status.json', 'execution_code.json', 'collection_summary.json',
                 'training_started_verified.json', 'resume_verification.json', 'active_resumption.json']:
        copy(experiment / name, 'metadata/' + name, 'provenance')
    resumption = Path(json.loads((experiment / 'active_resumption.json').read_text())['directory'])
    copy(resumption / 'resumption.json', 'metadata/resumption.json', 'provenance')
    for name in ['audit.json', 'benchmark.json']:
        copy(experiment / name, 'audits/local/' + name, 'current_audit')
    for name in ['phase0_unit.json', 'phase0_real.json', 'execution/benchmark.json']:
        copy(ROOT / 'streaming_sparse/v08/reports' / name, 'audits/v08/' + name, 'foundation_audit')
    for stage in ['audit', 'collection', 'training', 'assembly', 'quality', 'quality_merge']:
        name = f'{stage}_workers/launcher_status.json'
        copy(experiment / name, 'metadata/launchers/' + stage + '.json', 'execution_status')
    for stage in ['calibration', 'warmstart_K', 'warmstart_V']:
        for name in ['completed.json', 'validation_curve.json']:
            copy(ROOT / f'streaming_sparse/v08/runs/seed0/{stage}/{name}',
                 f'context/initialization/{stage}/{name}', 'initialization_provenance')
    for name in ['manifest.json', 'summary.json', 'end_to_end_assessment.json',
                 'video_comparison.csv', 'end_to_end_comparison.png', 'end_to_end_comparison.pdf']:
        copy(quality / name, 'results/quality/' + name, 'evaluation_result')
    for rank in range(8):
        for name in ['status.json', 'video_rows.json', *[m + '_offline.json'
                for m in ['Dense', 'mass', 'Klast', 'Vlast', 'Kbest', 'Vbest']]]:
            copy(quality / f'rank{rank}' / name, f'results/quality/rank{rank}/' + name, 'raw_metric_records')
    report_source = quality / 'END_TO_END_REPORT.md'
    report = report_source.read_text().split('固定排序第一个Prompt')[0].rstrip()
    report += ('\n\n## 评审包中的结果记录\n\n'
        '本包按要求不包含视频、latent、checkpoint或训练数据，也不包含视频帧截图。'
        '逐帧指标、路由计数、逐输入离线误差保留于rank0–rank7目录；'
        '原记录中的视频/权重路径仅用于身份追溯，不表示文件包含在本包中。\n\n'
        '汇总：end_to_end_assessment.json；逐视频表：video_comparison.csv；'
        '评估身份：manifest.json；各层更新数：../../metadata/assembled.json。\n')
    write('results/quality/REPORT.md', report, 'review_report', report_source)
    layer_rows = []
    for layer in range(1, 30):
        path = experiment / f'training/layer{layer:02d}/validation.json'
        d = json.loads(path.read_text())
        assert d['done']
        copy(path, f'results/training/layer{layer:02d}/validation.json', 'validation_curve')
        layer_rows.append(dict(layer=layer, done=d['done'], updates=d['updates'],
            K_best_step=d['best_step']['K'], V_best_step=d['best_step']['V'],
            initial_nmse=d['curve']['K'][0]['prompt_equal_nmse'],
            K_last_nmse=d['curve']['K'][-1]['prompt_equal_nmse'],
            V_last_nmse=d['curve']['V'][-1]['prompt_equal_nmse'],
            K_best_nmse=d['best']['K'], V_best_nmse=d['best']['V']))
    out = destination / 'results/training/layer_summary.csv'
    with out.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    evidence[str(out.relative_to(destination))] = dict(category='derived_training_summary', transformed=True)
    metrics_path = destination / 'results/training/optimizer_metrics.csv'
    metric_count = 0
    metric_sources = []
    replayed_iterations = []
    with metrics_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'log_record', 'iteration', 'iteration_occurrence', 'layer', 'kind',
            'attention_nmse_train', 'gradient_norm', 'lr', 'rank_update_seconds'])
        writer.writeheader()
        for rank in range(8):
            path = experiment / f'training/rank{rank}.metrics.jsonl'
            metric_sources.append(dict(source=str(path.relative_to(ROOT)), source_sha256=sha(path)))
            iterations = {}
            for log_record, line in enumerate(path.read_text().splitlines(), 1):
                record = json.loads(line)
                prior = iterations.setdefault(record['iteration'], [])
                if prior:
                    replayed_iterations.append(dict(rank=rank, iteration=record['iteration'],
                        log_record=log_record, occurrence=len(prior) + 1,
                        losses_and_sampling_identical=all(record[k] == prior[0][k]
                            for k in ['losses', 'sample_ids', 'query_slices'])))
                prior.append(record)
                for layer, values in record['losses'].items():
                    for kind in ['K', 'V']:
                        writer.writerow(dict(rank=rank, log_record=log_record, iteration=record['iteration'],
                            iteration_occurrence=len(prior), layer=layer,
                            kind=kind, attention_nmse_train=values['loss'][kind],
                            gradient_norm=values['gradient_norms'][kind], lr=values['lr'],
                            rank_update_seconds=record['seconds']))
                        metric_count += 1
    evidence[str(metrics_path.relative_to(destination))] = dict(category='training_metric_records',
        transformed=True, omitted_fields=['sample_ids', 'query_slices'], sources=metric_sources)
    write('metadata/training_log_replays.json', dict(records=replayed_iterations,
        policy='Preserve every logged observation. Duplicate iteration numbers are marked by log_record and iteration_occurrence; do not count them as extra optimizer updates.'), 'log_provenance')
    copy(ROOT / 'migration/local_router/verify_review.py', 'verify_review.py', 'review_verifier')
    v = assessment['video']
    table = '\n'.join(f"| {name} | {v[name]['sparse_frames_only']['psnr_db']:.4f} | "
        f"{v[name]['sparse_frames_only']['lpips']:.6f} |" for name in ['mass', 'Klast', 'Vlast', 'Vbest'])
    write('README.md', f'''# CAVIR v0.8 评审包

评审对象：`local_attn_29_20261023`，冻结Self-Forcing/Wan 1.3B骨干与LoRA，独立训练第1–29层的K-only/V-aware Router；第0层Dense。
本包包含关键代码、审计和数值结果。视频、视频帧、checkpoint、公共模型权重、训练数据、QKV缓存、Replay payload和latent均不包含。

## 先读这些文件

1. [评审导读](REVIEW_GUIDE.md)：实际执行范围、代码入口与结论边界。
2. [端到端报告](results/quality/REPORT.md)：完整表格、配对区间、逐chunk曲线。
3. [质量曲线](results/quality/end_to_end_comparison.png)；[PDF](results/quality/end_to_end_comparison.pdf)。
4. [各层停止点](results/training/layer_summary.csv)与[逐次优化指标](results/training/optimizer_metrics.csv)。
5. `audits/`：原始KV输出、Mask VJP、K-only信息边界及执行等价性检查。

## 本轮端到端结果

8个开发Val Prompt×2个生成Seed，16组45帧视频；首9帧Dense且逐像素一致，主指标使用后36帧。当前段完整保留，历史预算20%，所有4个去噪step均执行，context-write Dense。

| 模型 | PSNR↑ | LPIPS↓ |
|---|---:|---:|
{table}

Kbest与Klast张量相同。末尾权重V−K的LPIPS差为+0.0005766，Prompt配对95%CI为[-0.0012721, 0.0021815]；最佳权重配对区间也跨零。本轮未显示稳定V-aware优势。
训练后K/V相对共同初始Mass的LPIPS约降低10%；这反映整体训练收益，不能单独归因Value信息。

## 在没有模型和数据的机器上复核

解压后运行：

```bash
python verify_review.py
```

标准库即可校验包内SHA256、科学源码身份、排除项、192×6条离线记录、16组逐帧指标及29层完成状态。
环境中如有NumPy，还会重算5000次Prompt配对bootstrap并核对区间。包内没有权重和输入张量，不能直接重新生成视频或重新训练。

`MANIFEST.json`记录每个文件来源、原始SHA256及是否转换；`SHA256SUMS`覆盖所有交付文件（不含校验表自身）。
原JSON中出现的绝对路径和checkpoint/video文件名仅用于身份追溯；对应资产已明确排除。

生成时间：{datetime.now(timezone.utc).isoformat()}。
''', 'review_documentation')
    write('REVIEW_GUIDE.md', '''# 评审导读

## 实际执行范围

- 本轮训练目标只有逐层、拼接12个Heads的Attention输出NMSE；训练过程不优化网络x0或视频LPIPS。
- Dense冻结骨干产生固定输入，各事件128个独立Query，16个Train/4个局部Val Prompt，各2个生成Seed；本包仅保留配置、划分ID、指纹及汇总数，不带输入清单或张量。
- 每个实际Layer×Head独立参数；K/V共享初始化、事件和Query顺序，每层配对停止，实际350–800步。`800`是最大层计数。
- 推理执行全部4680个当前Query。当前chunk全保留，历史Page64、8原型、预算20%，第1–29层各12Heads稀疏，其余第0层Dense。
- 每个视频分支独立生成历史，全部4个原生去噪step，4个chunk/12个latent frames，对应45个RGB帧；不是在固定Dense历史上渲染的伪闭环。
- 只有一个Router训练Seed0。端到端使用8个开发Val Prompt×2个生成Seed；封存Test未访问。

## 代码入口（均在code目录）

| 审查问题 | 关键代码 |
|---|---|
| K-only是否真正不读取V；K/V容量是否匹配 | `streaming_sparse/v08/router.py`、`migration/local_router/core.py` |
| 真实KV页统计、Attention输出及连续Mask梯度 | `migration/local_router/core.py`、`streaming_sparse/v08/attention.py`、`streaming_sparse/v05/primitives.py` |
| 采集信息边界与128个独立Query | `migration/local_router/prepare.py`、`collect.py` |
| 训练Loss、学习率、采样和配对早停 | `migration/local_router/train.py` |
| 各层参数身份与最终权重组装 | `migration/local_router/finish.py`、`migration/paired_router/runtime.py` |
| 稀疏层选择、真实QKV、context-write | `streaming_sparse/v08/backbone.py`、`cache.py`、`wan/modules/causal_model.py` |
| 同噪声四step闭环与各自历史 | `streaming_sparse/v08/rollout.py` |
| 压缩前RGB PSNR/LPIPS、首chunk相同检查 | `migration/frozen_quality/worker.py`；当前标签适配在`migration/paired_router/quality.py` |
| Prompt等权、bootstrap、最终汇总 | `migration/local_router/assessment.py`、`streaming_sparse/v08/metrics.py`、`migration/frozen_quality/report.py` |
| 启动、依赖导入缓存、执行优化 | `migration/local_router/controller.py`、`migration/runtime/bootstrap.py`、`router_fast.py`、`streaming_sparse/v08/execution/accelerate.py` |

源文件依赖由AST遍历补齐，保留原仓库相对路径；模型架构代码与JSON配置保留，模型权重不包含。
`context/v08_original_README.txt`描述早期三层稀疏及F/A/J×K/V计划，不能把这些计划当成本轮已完成矩阵。
当前层范围及Loss以`metadata/experiment_config.json`和本导读为准。

## 对照与结果边界

- Mass是本轮共同初始Router、残差头为零。仅4/14/24迁移旧校准，其余26层全新初始化，使用Train固定归一化统计。
- K/V的Assignment、校准器、内容投影及残差头均可学习；骨干和LoRA冻结。
- Kbest=Klast；Vbest只在第2层选择600步而不是末尾700步，其余各层相同；选择依据是局部Val，不是视频结果。
- 视频指标衡量与相同噪声Dense输出的一致性；不单独代表审美或文本一致性。PSNR先平均MSE再转dB，LPIPS为AlexNet v0.1逐帧均值。
- 同一Prompt的两个生成Seed先均值，再按8个Prompt配对bootstrap5000次；不要把生成Seed当作独立训练Seed。
- 局部Attention误差下降不能替代端到端结论；闭环误差随chunk累积，V-aware本轮没有稳定优于K-only。
- 当前使用masked SDPA，没有稀疏kernel；逻辑连接保留率不等于实测加速比。

## 原始记录与派生文件

- `results/quality/rank*/`保留6种模式的逐输入SSE/energy/NMSE和逐帧视频指标、路由计数、冻结检查；这些是数值结果，不是输入张量或视频。
- `results/training/layer*/validation.json`保留全部局部Val曲线、最佳点与停止状态。
- `optimizer_metrics.csv`由训练日志提取Loss、梯度、LR、rank总更新耗时，删除sample_ids和query_slices；耗时不是单层或单视频延迟。同一步的重复日志全部保留，由log_record/iteration_occurrence标记；rank2第482步、rank5第652步各有一次同Loss与同采样的重放记录，详见`metadata/training_log_replays.json`，不能计为额外更新。
- `metadata/experiment_config.json`删除原始rows字段，只保留配置、split ID、指纹及样本总数。`evaluation_prompts.json`仅含评估Val文本，不含Train/Test文本。
- `audits/local/`是本轮重跑审计；`audits/v08/`是匹配科学源码身份的基础数学/真实网络/执行等价性审计，二者适用范围不同。
- 10月24日从断点恢复：缓存重采集，原优化器/采样器续接，完成rank跳过；细节和前后SHA见`metadata/resumption.json`，首批次与AdamW状态验证见`resume_verification.json`。
- 运行源码SHA与本包交付源码SHA分别可追溯，原路径保留为引用；排除资产没有被替换成伪造占位文件。
''', 'review_documentation')
    write('metadata/bundle_scope.json', dict(
        experiment=experiment.name, scientific_source_sha256=config['source_sha256'],
        excluded=['video files', 'video frame screenshots', 'checkpoints and public weights',
                  'training input data', 'QKV cache', 'Replay payload tensors', 'latent tensors',
                  'virtualenv', 'dependency caches', 'training sample/query-index manifests'],
        included_training_metric_rows=metric_count, included_source_files=len(sources),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()), 'bundle_scope')
    banned = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.pkl', '.pickle', '.npy', '.npz',
              '.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif', '.zip'}
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    for path in files:
        rel = path.relative_to(destination)
        assert path.suffix.lower() not in banned, rel
        assert not {'data', 'checkpoints', '__pycache__', '.venv'} & set(rel.parts), rel
        if path.suffix in ['.png', '.pdf']:
            assert path.stem == 'end_to_end_comparison'
    package_manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scientific_source_sha256=config['source_sha256'],
        files=[dict(path=str(p.relative_to(destination)), bytes=p.stat().st_size, sha256=sha(p),
                    **evidence[str(p.relative_to(destination))]) for p in files])
    (destination / 'MANIFEST.json').write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2) + '\n')
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(destination)}\n' for p in files))
    archive = destination.with_suffix('.zip')
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(p for p in destination.rglob('*') if p.is_file()):
            bundle.write(path, destination.name + '/' + str(path.relative_to(destination)))
    archive.with_suffix('.zip.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    result = dict(directory=str(destination), archive=str(archive), bytes=archive.stat().st_size,
                  files=len(files) + 1, source_files=len(sources), training_metric_rows=metric_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.directory, args.output)


if __name__ == '__main__':
    main()
