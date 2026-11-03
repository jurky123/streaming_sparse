"""Report router-only heldout video comparisons; no speed claim."""
import json
from pathlib import Path
from .learned_cache.io import atomic
BASE=Path('artifacts/review_v1_20261102')
def main():
    lines=['# 下游监督路由：独立消融','', '其他权重、KV上限、10%精确页预算、16 probes均固定。训练只更新6层×3 heads的路由θ，其余层/head保持原值。训练city_walk、reef_fish；验收horse_field、traffic_crossing，seed101，前段RGB[21,101)。','', '每条标签只改变一个head，在72个query tile各替换一个未受保护的边界页，比较真实两步x₀误差。训练监督是整组替换的偏好，不冒充单页独立贡献或完整视频梯度。','', '|场景|路由|LPIPS↓|SSIM↑|PSNR↑ (dB)|','|---|---|---:|---:|---:|'];rows=[]
    for case in ['horse_field','traffic_crossing']:
        for name,path in [('原风险路由',BASE/'video'/f'{case}_s101'/'candidate'/'result_9.json'),('注意力质量路由',BASE/'ablation/video'/f'{case}_s101'/'mass_route'/'result_9.json'),('下游监督路由',BASE/'route_downstream/video'/f'{case}_s101'/'full'/'result_9.json')]:
            if not path.exists():lines.append(f'|{case}|{name}|待完成|待完成|待完成|');continue
            data=json.loads(path.read_text());q=data['quality']['early'];rows.append(dict(case=case,method=name,quality=q))
            lines.append(f"|{case}|{name}|{q['lpips']:.6f}|{q['ssim']:.6f}|{q['psnr']:.3f}|")
    lines+=['','共享GPU不作速度验收；下游监督不增加推理特征或参数规模。仅两个场景单seed，尚不足以证明泛化。训练集pair准确率不能替代闭环验收。']
    Path('docs/review_v1_20261102/route_downstream.md').write_text('\n'.join(lines)+'\n')
    atomic(BASE/'route_downstream/summary.json',rows)
if __name__=='__main__':main()
