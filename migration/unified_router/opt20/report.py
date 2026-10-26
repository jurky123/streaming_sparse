import json,statistics,html
from pathlib import Path
from .setup import ROOT
from ..common import atomic_json,sha

def main():
    training=[]
    for layer in range(1,30):
        for variant in ['Tile15','Tile20']:
            p=ROOT/f'training/layer{layer:02d}/{variant}'
            d=json.loads((p/'completed.json').read_text());best=json.loads((p/'best_validation.json').read_text())
            training.append(dict(**d,best_updates=best['updates'],keep=best['keep'],executed=best['executed'],relative_improvement=1-d['best']/d['initial']))
    chosen=json.loads((ROOT/'chosen_checkpoints.json').read_text())
    for f,digest in chosen.items():assert sha(f)==digest,f
    cases=[json.loads(f.read_text()) for f in sorted((ROOT/'e2e_final').glob('*.seed0/result.json'))];assert len(cases)==8
    names=['Tile15_before','Tile15_after','Tile20_before','Tile20_after'];quality={}
    for name in names:
        quality[name]=dict(lpips=statistics.mean(c['variants'][name]['quality_vs_dense']['sparse_frames_only']['lpips'] for c in cases),
            psnr_db=statistics.mean(c['variants'][name]['quality_vs_dense']['sparse_frames_only']['psnr_db'] for c in cases),
            keep=statistics.mean(c['variants'][name]['weighted_keep'] for c in cases),
            executed=statistics.mean(c['variants'][name]['executed_attention_ratio'] for c in cases),
            generation_seconds=statistics.mean(c['timings'][name] for c in cases))
    dense=statistics.mean(c['timings']['Dense'] for c in cases)
    atomic_json(ROOT/'summary.json',dict(training=training,quality=quality,dense_generation_seconds=dense,cases=cases,
        sparse_layers=list(range(1,30)),dense_layers=[0],first_chunk_dense=True,video_cases=8,video_frames=45,resolution=[480,832],
        best_metric='actual_deployment_attention_nmse',video_used_for_checkpoint_selection=False,default_deployment_changed=False))
    lines=['Opt20：全29层实际15%/20% tile微调与完整视频评估','',
      '层1–29使用32-query共享tile；层0/首chunk Dense。候选35%、8代表；历史P8/P4/P2、当前P16。',
      '训练目标是实际Opt19 kernel的attention输出NMSE；不是路由标签、动作头或视频像素损失。',
      '原型分配权重/偏置与key投影联合训练；主干、分组树、预算、INT8 QK/BF16 PV固定。',
      '每层每预算最多2000步；至少1000步后允许验证平台期停止。best仅按实际Val输出误差选取。',
      '每组验证96事件，4 prompt×2 seed×3历史长度×4 step；full4680Q，128采样位置监督。',
      '全视频使用8个Val prompt、seed0、45帧480x832；LPIPS/PSNR统计后36帧；生成时间不含VAE，含历史重建。','',
      f'Dense生成平均 {dense:.4f}s','方案                LPIPS↓    PSNR↑   keep%   execute%  生成秒']
    for name,d in quality.items():lines.append(f"{name:18} {d['lpips']:.5f} {d['psnr_db']:.3f} {100*d['keep']:.2f} {100*d['executed']:.2f} {d['generation_seconds']:.4f}")
    lines+=['','layer variant updates best_step initial_NMSE best_NMSE improvement']
    for d in training:lines.append(f"{d['layer']:02} {d['variant']:7} {d['updates']:4} {d['best_updates']:4} {d['initial']:.6f} {d['best']:.6f} {100*d['relative_improvement']:+.2f}%")
    lines+=['','主干与既有实验文件未修改。视频指标未用于选择checkpoint；没有宣称Test泛化。',
      '若best仍出现在末步，不据此宣称收敛；训练曲线在training/layerXX/Tile15或Tile20/loss_curve.png。',
      '视频浏览：index.html；三列顺序为Dense、微调前、微调后。']
    (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    body=['<!doctype html><meta charset="utf-8"><title>Opt20 全层视频对比</title>',
      '<style>body{font-family:sans-serif;max-width:1500px;margin:24px auto;background:#17191d;color:#eee}video{width:100%;margin-bottom:24px}a{color:#8cf}</style>',
      '<h1>Opt20：全29层 tile 微调</h1><p>视频顺序：Dense / 微调前 / 微调后。层0与首chunk Dense。</p><a href="REPORT.txt">完整指标</a>']
    for case in cases:
        pid=case['prompt_id'];body.append(f'<h2>{html.escape(pid)}</h2>')
        for variant in ['Tile15','Tile20']:
            src=f'e2e_final/{pid}.seed0/{variant}_Dense_before_after.mp4'
            body.append(f'<h3>{variant}</h3><video controls preload="metadata" src="{src}"></video>')
    (ROOT/'index.html').write_text('\n'.join(body))
    atomic_json(ROOT/'completion.json',dict(status='completed',training_jobs=58,sparse_layers=29,video_cases=8))
    print(ROOT/'REPORT.txt')

if __name__=='__main__':main()
