"""Start the existing audited resume entry inside an independent tmux server."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOCKET='streaming_sparse'
SESSION='cavir-batch8'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        os.chdir(ROOT)
        python=ROOT/'streaming_sparse/.venv/bin/python'
        os.execv(str(python),[str(python),'-u',str(ROOT/'migration/run_batch8.py')])
    prefix=['tmux','-L',SOCKET]
    exists=subprocess.run(prefix+['has-session','-t',SESSION],capture_output=True).returncode==0
    if exists:
        print(f'后台会话已存在：{SESSION}；运行 ./progress.sh 查看进度。')
        return
    command=shlex.join([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'])
    subprocess.run(prefix+['new-session','-d','-s',SESSION,'-c',str(ROOT),command],check=True)
    process=subprocess.run(prefix+['display-message','-p','-t',SESSION,'#{pid} #{pane_pid}'],
        capture_output=True,text=True,check=True).stdout.strip()
    record=dict(backend='tmux',socket=SOCKET,session=SESSION,server_and_pane_pids=process,
        started_at=datetime.now(timezone.utc).isoformat(),resume_entry=str(ROOT/'migration/run_batch8.py'))
    path=ROOT/'migration/restoration/20261022_133350/detached_session.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(f'已启动独立后台会话：{SESSION}；运行 ./progress.sh 查看进度。')


if __name__=='__main__':main()
