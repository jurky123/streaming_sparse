"""Paired local/early-video ablation report; no speed claims or invented results."""
import json
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/review_v1_20261102/ablation')
def main():
    lines=['# 消融实验（2026-11-03）','','固定512×768、338历史页、16 probes、10%精确预算；骨干与各候选checkpoint冻结。没有为去除模块重新训练，结果表示当前模型对该模块的依赖。共享GPU不作速度结论。','','## 同状态局部消融','','固定Q/K/V与物理页。均值/细节/无补偿及fixed_mask训练对照使用完全相同mask；路由对照使用相同摘要和预算。局部结果不包含重排分组或闭环状态变化。','','|场景|对照|事件数|平均NMSE|能量加权NMSE|相对full变差事件|','|---|---|---:|---:|---:|---:|'];records=[]
    for case in ('horse_field','traffic_crossing'):
        rows=[json.loads(p.read_text()) for p in sorted((ROOT/'local'/case).glob('c*_l*_s*.json'))]
        names=sorted({n for row in rows for n in row['rows']})
        for name in names:
            values=[x['rows'][name] for x in rows];n=len(values);macro=sum(x['nmse'] for x in values)/n;micro=sum(x['squared_error'] for x in values)/sum(x['target_energy'] for x in values);worse=sum(x['rows'][name]['nmse']>x['rows']['full']['nmse'] for x in rows)
            lines.append(f'|{case}|{name}|{n}|{macro:.6f}|{micro:.6f}|{worse}/{n}|');records.append(dict(case=case,variant=name,events=n,macro_nmse=macro,energy_weighted_nmse=micro,worse=worse))
    lines+=['','## 前段闭环消融','','主窗口RGB[21,101)，seed101。full与旧版复用此前同输入的结果。每个消融自由演化自身轨迹；去均值/细节不改该次路由使用的完整统计，分组关闭仍保持原有partial-chunk淘汰页顺序。','','|场景|对照|LPIPS↓|SSIM↑|PSNR↑ (dB)|相对full LPIPS变化|','|---|---|---:|---:|---:|---:|'];video=[]
    base=Path('artifacts/review_v1_20261102/video')
    for case in ('horse_field','traffic_crossing'):
        full=json.loads((base/f'{case}_s101/candidate/result_9.json').read_text())['quality']['early']
        paths=[('full',base/f'{case}_s101/candidate/result_9.json'),('accepted',base/f'{case}_s101/accepted/result_9.json')]+[(p.parent.name,p) for p in sorted((ROOT/'video'/f'{case}_s101').glob('*/result_9.json'))]
        for name,p in paths:
            if not p.exists():continue
            q=json.loads(p.read_text())['quality']['early'];delta=q['lpips']-full['lpips'];lines.append(f"|{case}|{name}|{q['lpips']:.5f}|{q['ssim']:.5f}|{q['psnr']:.3f}|{delta:+.5f}|");video.append(dict(case=case,variant=name,quality=q,lpips_delta=delta))
    lines+=['','projected_mean仅将完整均值替换为当前basis和旧center的rank8重建，保留相同K、细节与mask，并非完整旧kernel重现。no_new_training保留新结构但不做本轮训练；no_downstream保留摘要预训练。mass_route保持同预算且强制own页。no_grouping仅在视频中评估。','', '相关事件不作为独立统计样本；两个场景的消融不能支持普遍质量或速度结论。缺失组仍待运行，不填充推测结果。']
    Path('docs/review_v1_20261102/ablation_results.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(local=records,video=video))
if __name__=='__main__':main()
