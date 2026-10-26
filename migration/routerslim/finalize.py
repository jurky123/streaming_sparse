"""Freeze reviewable evidence before enabling the R1 training entry."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args();root=Path(args.directory).resolve()
    evidence={};records={}
    def load(name):
        path=root/name;value=json.loads(path.read_text());assert value['passed'],(name,value.get('error'))
        evidence[str(path)]=sha(path);records[name]=value;return value
    router=load('router_audit.json');batch=load('batch8_audit.json')
    for h in (4680,9360,14040):
        for prefix in ('full','train'):
            check=load(f'{prefix}_{h}.json');assert len(check['cases'])==4
            assert check['r1_sha256']==router['r1_sha256']
    base=load('matched_baseline_profile/profile_breakdown.json');new=load('matched_training_profile/profile_breakdown.json')
    detailed=load('baseline_profile/profile_breakdown.json')
    for variant in ('variance','projection','skip','optimized','training'):load(f'{variant}_profile/profile_breakdown.json')
    assert router['r1_sha256']==batch['r1_sha256']==sha(ROOT/'migration/routerslim/r1.py')
    assert router['checkpoint_sha256']==batch['checkpoint_sha256']==sha(root/'baseline.pt')
    assert batch['world_size']==8 and batch['global_batch']==8 and batch['update']==820
    measurements=[]
    for a,b in zip(base['samples'],new['samples']):
        assert a['sample_id']==b['sample_id']and a['history_tokens']==b['history_tokens']
        assert b['median_seconds']<a['median_seconds']
        measurements.append(dict(history_tokens=a['history_tokens'],baseline_median=a['median_seconds'],new_median=b['median_seconds'],
            baseline_p90=a['p90_seconds'],new_p90=b['p90_seconds'],speedup=a['median_seconds']/b['median_seconds']))
    report=dict(passed=True,source_sha256=router['source_sha256'],r1_sha256=router['r1_sha256'],entry_sha256=sha(ROOT/'migration/routerslim/train_module.py'),
        checkpoint_sha256=router['checkpoint_sha256'],features=batch['features'],evidence=evidence,
        baseline_update=819,global_batch=8,world_size=8,measurements=measurements,
        created_at=datetime.now(timezone.utc).isoformat(),scope='Frozen F-K/seed0 continuation; all-parameter router gradients; 480p offline network replay, not streaming inference or end-to-end video speed.',
        source_files={str(path):sha(path)for path in sorted((ROOT/'migration/routerslim').glob('*.py'))})
    (root/'deployment_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# R1 训练优化：Profile 与验收','',
        '已在冻结的第 819 次更新上完成检查。正式训练接续原 optimizer、sampler、学习率、验证曲线与 2000 更新预算，使用 8 卡、batch 8。',
        '', '**范围：480p、Self-Forcing 1.3B、F-K/seed0 离线网络输出训练；不是 720p 流式生成推理加速报告。**','',
        '## 实际启用','',
        '- 有效原型的等权方向方差提前聚合；不按 count 加权。',
        '- 更新 ≥501 时停止权重已经为零的 Attention 辅助分支；校准辅助项、原 8×16 Query 抽样与归一化保留。日志记 null/停算，避免当作 loss 降到零。',
        '- 连续 Soft Gate、FP64 VJP、全维 K、完整残差范数、当前段与历史预算保持。边界分组回退到原公式和原 16 Query 计算形状。',
        '- 内容投影前移已实现并通过等价检查，但当前训练实测变慢，默认关闭。没有进行低维近似、修改 MLP 结构或引入推理专用 Hard-only 路径。','',
        '## 同一卡完整输入前后向计时','',
        '物理 GPU 4，OMP=4，同 checkpoint/同输入。每例一次冷启动后 5 次同步计时；中位数/P90 不含 profiler、首次 CUDA graph 捕获、数据加载、优化器更新和多卡归约。每个输入都重新构建历史/摘要，未使用跨参数版本缓存。',
        '', '| 历史 Tokens | 原版中位/P90（秒） | 启用后中位/P90（秒） | 加速比 |','|---:|---:|---:|---:|']
    for m in measurements:text.append(f"| {m['history_tokens']} | {m['baseline_median']:.3f} / {m['baseline_p90']:.3f} | {m['new_median']:.3f} / {m['new_p90']:.3f} | {m['speedup']:.3f}× |")
    text+=['','八卡实际更新速度另见 metrics.jsonl；不能把上表当作完整训练吞吐或视频生成收益。','',
        '## 原版 CUDA 分项','',
        '来自 GPU 0 单次带注释 trace；只用于成本定位，不用于前后速度比较。统计包含前向、checkpoint 重算与反向，每个 CUDA kernel 只计一次。CUDA 时间之和与同步墙钟时间不同，不能直接相加。完整前后向/历史/当前分解见 profile_breakdown.csv。',
        '', '| 分项 CUDA 秒 | 历史 4680 | 历史 14040 |','|---|---:|---:|']
    names=set();totals=[]
    for item in detailed['samples']:
        d=defaultdict(float)
        for row in item['profile']['components']:d[row['component']]+=row['gpu_seconds'];names.add(row['component'])
        totals.append(d)
    for name in sorted(names,key=lambda n:-sum(d[n]for d in totals)):
        text.append(f"| {name} | {totals[0][name]:.4f} | {totals[1][name]:.4f} |")
    text+=['',
        'native_masked_forward 在反向阶段含必要的 native Attention 重算；native_qkv_backward 为其原生 QKV 导数，其余连续门导数单列 gate_vjp。原生融合 Attention 不能从此 trace 精确拆成 QK、Softmax、AV 三个独立延迟。辅助 Attention 的部分前向和未注释的骨干/搬运在 backbone.other；unattributed 明示无法匹配 launch 的少量 kernel，不将其冒归某模块。',
        '', '## 等价性与更新验证','',
        f"FP64 方差/投影/梯度最大误差：{router['algebra']['variance_maxabs']:.3e} / {router['algebra']['projection_maxabs']:.3e} / {router['algebra']['gradient_maxabs']:.3e}。",'',
        '独立 Router：6 个 Layer×Head 对、3 种历史，256 独立 Query；Hard Mask 逐位一致，K-only 更换 V 后评分不变。',
        '完整网络：三种历史×四个去噪步骤，分别检验全部候选与最终启用候选；所有 x0 逐位一致；所有 Router 参数的梯度存在性一致，atol=5e-5、rtol=2e-4，未放宽旧容差。AdamW 使用第819步保存的真实 moments。',
        f"最终候选第820步的真实下一批8输入：梯度最大差 {batch['gradient_maxabs']:.3e}，AdamW 参数最大差 {batch['adamw_parameter_maxabs']:.3e}，moment 最大差 {batch['adamw_moment_maxabs']:.3e}；anchor 已计入。",'',
        '未修改科学源码、冻结骨干、训练数据/Query 清单与 optimizer/sampler 状态。所有证据 SHA 见 deployment_audit.json；checkpoint 和 Router SHA 可追溯。',
        '', '## 复现与查看','',
        '运行 `./progress.sh` 看实时进度；运行 `./start_training.sh` 从最新完整 checkpoint 恢复。独立 tmux 后台运行，退出终端不终止训练。',
        '原版回退：保留最新恢复点，将 migration/routerslim_deployment.json 移出该路径，再用原启动入口恢复；不要在运行中改执行代码。',
        '完整 trace：baseline_profile/baseline_history{4680,14040}.trace.json；CSV：baseline_profile/profile_breakdown.csv；分项消融：*_profile/profile_breakdown.json。', '']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(dict(audit=str(root/'deployment_audit.json'),report=str(root/'REPORT.md'),measurements=measurements)),flush=True)


if __name__=='__main__':main()
