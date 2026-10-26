"""Show sealed benchmark progress and final profile/report progress together."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization5'
    while True:
        if not a.once:print('\033[2J\033[H',end='',flush=True)
        subprocess.run([sys.executable,'-m','migration.unified_router.opt5.status','--once'],check=True)
        try:state=json.loads((root/'profile/status.json').read_text())['status']
        except(OSError,ValueError):state='等待视频评估结束'
        print('模块 profile:',state)
        print('报告与对比视频:','完成'if(root/'analysis_manifest.json').exists()else'整理中 / 待开始')
        if a.once:return
        time.sleep(5)

if __name__=='__main__':main()
