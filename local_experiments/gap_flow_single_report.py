import json
from pathlib import Path
ROOT=Path('artifacts/flow_single_20261104')
def main():
    lines=['# 单步 flow 蒸馏验收','','起点为摘要预训练；冻结骨干和硬路由θ，训练29层的7类稀疏摘要/分组特征参数。两个训练场景，各chunk2、5和四个实际推理时间步，共16次更新。损失为原生flow的逐元素MSE，学生与dense教师共享输入、时间步、文本及历史。这里只验证可微模块的flow蒸馏，未重新采集硬路由排序标签。','','主验收为seed101、RGB[21,101)，同分辨率、KV上限和10%选页预算。共享卡不作速度验收。','','|场景|配置|LPIPS↓|SSIM↑|PSNR↑ dB|','|---|---|---:|---:|---:|']
    for case in ['horse_field','traffic_crossing']:
        for name,path in [('摘要预训练',Path('artifacts/review_v1_20261102/ablation/video')/f'{case}_s101/no_downstream/result_9.json'),('单步flow蒸馏',ROOT/'video'/f'{case}_s101/full/result_9.json')]:
            if not path.exists():lines.append(f'|{case}|{name}|待完成|待完成|待完成|');continue
            q=json.loads(path.read_text())['quality']['early'];lines.append(f"|{case}|{name}|{q['lpips']:.6f}|{q['ssim']:.6f}|{q['psnr']:.3f}|")
    out=Path('docs/flow_single_20261104.md');out.write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
