"""Report completed paired quality evidence without inventing a speed result."""
import argparse,json,subprocess
import imageio_ffmpeg
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/review_v1_20261102')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--chunks',type=int,choices=[9,41],default=41);args=ap.parse_args()
    rows=[];missing=[];lines=['# 扩大后的闭环质量验收','','相同512×768、338历史页（layer0为360）、同seed与噪声。主验收看RGB[21,101)，同时记录中后段。共享GPU，不作速度结论。','','|场景|seed|窗口|旧版LPIPS|新版LPIPS|旧版SSIM|新版SSIM|','|---|---|---|---|---|---|---|']
    for case in ('horse_field','traffic_crossing'):
        for seed in ((101,303) if args.chunks==41 else (101,)):
            base=ROOT/'video'/f'{case}_s{seed}';files=[base/m/f'result_{args.chunks}.json' for m in ('dense','accepted','candidate')]
            if not all(p.exists() for p in files):missing.append(f'{case}_s{seed}');continue
            old=json.loads(files[1].read_text());new=json.loads(files[2].read_text())
            comparison=base/('comparison.mp4' if args.chunks==41 else 'comparison_9.mp4')
            if not comparison.exists():
                tmp=base/'comparison.tmp.mp4'
                command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error']
                for method in ('dense','accepted','candidate'):command+=['-i',str(base/method/f'video_{args.chunks}.mp4')]
                command+=['-filter_complex','[0:v]scale=384:256[a];[1:v]scale=384:256[b];[2:v]scale=384:256[c];[a][b][c]hstack=inputs=3[v]','-map','[v]','-c:v','libx264','-crf','18','-threads','4',str(tmp)]
                subprocess.run(command,check=True);tmp.replace(comparison)
            for window in (('early','middle','late') if args.chunks==41 else ('early',)):
                a=old['quality'][window];b=new['quality'][window]
                rows.append(dict(case=case,seed=seed,window=window,lpips_delta=b['lpips']-a['lpips'],ssim_delta=b['ssim']-a['ssim']))
                lines.append(f"|{case}|{seed}|{window}|{a['lpips']:.5f}|{b['lpips']:.5f}|{a['ssim']:.5f}|{b['ssim']:.5f}|")
    early=[x for x in rows if x['window']=='early']
    # No invented tolerance: strict paired check reported as a screening result,
    # not a claim of statistical significance or completed human video review.
    passed=bool(early) and not missing and all(x['lpips_delta']<=0 and x['ssim_delta']>=0 for x in early)
    summary=dict(status='incomplete' if missing else 'numeric_screen_passed_pending_visual_review' if passed else 'quality_tradeoff_requires_analysis',missing=missing,rows=rows,speed_valid=False,production_precision_accepted=False)
    summary['chunks']=args.chunks
    atomic(ROOT/('acceptance.json' if args.chunks==41 else 'early_acceptance.json'),summary)
    lines+=['',f"状态：{summary['status']}。",'','FP32质量参考实现，尚未验收生产TF32路径；LPIPS/SSIM只衡量接近同输入Dense，不能替代运动、主体和闪烁的视频检查。', '', '各case目录comparison.mp4为完整对照，左到右：Dense、旧版、新版。']
    (Path('docs/review_v1_20261102')/('expanded_acceptance.md' if args.chunks==41 else 'early_acceptance.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
