"""Lightweight, readable speed/profile status (stdlib only)."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os
import sys
import time

from status import read,table,duration

ROOT=Path(__file__).resolve().parents[2]
STATES={'loading':'加载中','warmup':'预热中','benchmark':'测速中','waiting_for_benchmarks':'测速完成，等待统一Profile',
        'profiling':'模块Profile中','completed':'完成'}


def render(root):
    out=root/'speed_profile';m=read(out/'manifest.json');state=read(out/'status.json')
    if not m:return '尚无测速任务。'
    rows=[];done=0
    for rank,mode in enumerate(m['assignments']):
        folder=out/f'rank{rank}';s=read(folder/'status.json');d=read(folder/'timings.json');samples=d.get('samples',[])
        done+=len(samples)
        avg=lambda key: f'{sum(x[key]for x in samples)/len(samples):.3f}'if samples else '—'
        rows.append([rank,mode,f'{len(samples)}/{m["repetitions"]}',avg('generation_seconds'),
            avg('decode_and_transfer_seconds'),avg('total_seconds'),STATES.get(s.get('status'),'等待启动')])
    label={'running':'进行中','completed':'全部完成','failed':'失败'}.get(state.get('status'),'准备中')
    elapsed=time.time()-datetime.fromisoformat(m['created_at']).timestamp()
    lines=[f'Slim-AV v0.10 速度/Profile · {label} · 已用{duration(elapsed)}',
        '同Prompt/seed · 单卡A100 · 45帧480×832 · 先预热，全部测速结束后才开Profiler',
        f'纯测速进度 {done}/{len(m["assignments"])*m["repetitions"]} 次（不含预热）','',
        *table(['卡','方案','次数','生成秒','解码秒','总秒','状态'],rows),'',
        '计时含Dense上下文重建；排除文本编码、加载、LPIPS和MP4编码。',
        '报告：'+str(out/'REPORT.md'),'Ctrl+C只退出查看，不影响测量。']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--directory',type=Path)
    a=p.parse_args();root=a.directory or Path(read(ROOT/'migration/unified_router_active.json')['directory'])
    tty=sys.stdout.isatty()and os.environ.get('TERM')!='dumb';watch=a.watch or(tty and not a.once)
    try:
        if tty and watch:sys.stdout.write('\033[?1049h\033[?25l')
        while True:
            sys.stdout.write(('\033[H'if tty and watch else '')+render(root)+('\033[J'if tty and watch else '')+'\n');sys.stdout.flush()
            if not watch:break
            time.sleep(5)
    except KeyboardInterrupt:pass
    finally:
        if tty and watch:sys.stdout.write('\033[?25h\033[?1049l');sys.stdout.flush()


if __name__=='__main__':main()
