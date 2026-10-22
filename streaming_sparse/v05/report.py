"""Explicit Phase1 stability gate and final three-seed review summaries."""
import argparse
import json
from pathlib import Path
import statistics
from .common import HERE,source_identity,atomic_json,protect_output


def phase1_gate(root):
    root=Path(root);results=[];lines=['v0.5 Phase1：固定历史前端/分区/Pre-Q H，R1/R2/R3逐Query诊断','']
    for seed in (0,1):
        run=root/f'phase1/seed{seed}'
        result=json.loads((run/'phase1_result.json').read_text())
        curve=json.loads((run/'validation_curve.json').read_text())
        if result['source_sha256']!=source_identity():raise ValueError('Stale phase1 result')
        initial=curve[0];best=min(curve,key=lambda row:row['joint_validation'])
        improved=best['joint_validation']<=initial['joint_validation']*.99
        bounded=all(best['arms'][k]<=initial['arms'][k]*1.25 for k in ('R1','R2','R3'))
        passed=result['passed'] and improved and bounded
        result.update(seed=seed,gate_passed=passed,one_percent_joint_improvement=improved,all_arms_validation_bounded=bounded)
        results.append(result)
        lines.append(f"seed{seed}：{'通过' if passed else '未通过'}；best step={best['updates']}；joint Val {initial['joint_validation']:.6f} -> {best['joint_validation']:.6f}")
        for arm in ('R1','R2','R3'):
            lines.append(f"  {arm}：Val {initial['arms'][arm]:.6f} -> {best['arms'][arm]:.6f}；soft={best['soft'][arm]:.6f}；Mass={best['mass']:.6f}")
    passed=all(x['gate_passed'] for x in results)
    lines+=['','本检查只判断数值稳定、学习信号和全部8tiles覆盖，不是架构胜负判断。',
             '通过后自动启动完整数据R2/R3的三个Router seeds；失败则保留日志并停止扩展。']
    report=protect_output(root/'reports/phase1_gate.json');report.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(report,dict(passed=passed,source_sha256=source_identity(),results=results))
    (report.parent/'PHASE1_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    if not passed:raise RuntimeError('Phase1 stability/learning gate failed; do not launch full training')


def final(root):
    root=Path(root);collected={};lines=['v0.5 完整实验结果（3 Router seeds；mean ± sample std）','']
    for split in ('train','val','test'):
        seeds=[json.loads((root/f'phase2/seed{s}/evaluation_{split}/summary.json').read_text()) for s in (0,1,2)]
        for value in seeds:
            if value['identity']['source_sha256']!=source_identity():raise ValueError('Stale formal evaluation')
        for keep in (.3,.2,.1):
            key=f'keep={keep}';group={}
            reference=set(seeds[0]['metrics'])
            if any(set(x['metrics'])!=reference for x in seeds):raise ValueError('Unmatched seed slices')
            for slice_name in sorted(reference):
                if not (slice_name==key or slice_name.startswith(key+'/')):continue
                methods=[name for name,value in seeds[0]['metrics'][slice_name].items() if isinstance(value,dict) and 'global_attention_error' in value]
                values={}
                for name in methods:
                    errors=[x['metrics'][slice_name][name]['global_attention_error'] for x in seeds]
                    values[name]=dict(mean=statistics.mean(errors),std=statistics.stdev(errors),router_seed_values=errors)
                    for level in ('tile_relative_error','query_relative_error'):
                        values[name][level]={}
                        for tail in ('mean','median','p90','p99'):
                            a=[x['metrics'][slice_name][name][level][tail] for x in seeds]
                            values[name][level][tail]=dict(mean=statistics.mean(a),std=statistics.stdev(a))
                for field in ('global_delta_R2_minus_R3','relative_improvement','query_win_rate_R3','tile_win_rate_R3'):
                    a=[x['metrics'][slice_name]['paired'][field] for x in seeds]
                    values[field]=dict(mean=statistics.mean(a),std=statistics.stdev(a))
                group[slice_name]=values
            collected[f'{split}/{key}']=group
            top=group[key]
            lines.append(f"{split} keep={keep:.0%}："+'；'.join(f"{name}={top[name]['mean']:.6f}±{top[name]['std']:.6f}" for name in ('R2','R3','Mass','SpatialMass')))
            lines.append(f"  R3相对R2改善={top['relative_improvement']['mean']:.2%}；逐Query胜率={top['query_win_rate_R3']['mean']:.2%}")
    curves={str(s):json.loads((root/f'phase2/seed{s}/validation_curve.json').read_text()) for s in (0,1,2)}
    lines+=['','R0引用v0.4历史结果，前端与Norm不同，不能据此隔离逐Query的单一因果效应。',
             'R1/R2/R3比较与Norm监测没有单独Norm消融；Norm的因果收益尚未独立识别。',
             'Signed关系来自学习，不等同真实Attention Error交互；positive/absolute为冻结干预，不能替代重新训练消融。',
             '30%与10%只作预算泛化压力测试；训练预算仅20%。',
             '有限seed与最大训练预算不构成显著性或收敛证明；先审阅Validation轨迹。',
             '本轮未训练Backbone、未用Diffusion/Oracle监督、未测视频质量或内核速度。']
    out=protect_output(root/'reports');out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'final_three_seed_summary.json',dict(source_sha256=source_identity(),summaries=collected,validation_curves=curves,std_ddof=1))
    (out/'RESULTS_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    previous=HERE.parent/'v04/reports/final_three_seed_summary.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        atomic_json(out/'R0_v04_historical_reference.json',dict(note='Historical reference only: different frontend/normalization/query granularity; not a matched causal control',
                    source_sha256=old['source_sha256'],results=old['summaries']['train_keep=0.2/test']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase1-gate',action='store_true')
    parser.add_argument('--root',default=str(HERE));args=parser.parse_args()
    (phase1_gate if args.phase1_gate else final)(args.root)


if __name__=='__main__':main()
