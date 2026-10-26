"""Readable progress for the selected boundary-recomputed head."""
import argparse,json,time
from pathlib import Path

def read(path):
    try:return json.loads(path.read_text())
    except(OSError,ValueError):return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization6'
    while True:
        if not a.once:print('\033[2J\033[H',end='')
        print('SlimAV Opt6 · 分解动作 head + 临界评分回退')
        s=read(root/'guard_status.json');print('阶段:',s.get('stage','微基准'),' 状态:',s.get('status','等待'))
        print('跨层检查:',len(read(root/'checks.json').get('rows',[])),'/ 18；动作边界:',len(read(root/'boundary_checks.json').get('cases',[])),'/ 7')
        for folder in sorted((root/'guard_benchmark').glob('*')):
            v=read(folder/'status.json');rows=read(folder/'timings.json').get('records',[])[1:]
            perf=f" 生成 {sum(r['generation_seconds']for r in rows)/len(rows):.3f}s"if rows else ''
            print('测速',folder.name,v.get('status','等待'),f'{len(rows)}/3'+perf)
        files=list((root/'guard_quality').glob('*.seed0/Guarded-*.json'))
        print('最终视频评估:',len(files),'/ 16；latent与Opt5逐bit一致:',sum(read(f).get('latent_bit_identical',False)for f in files))
        for f in sorted((root/'guard_quality').glob('rank*.json')):
            r=read(f);print(' ',f.stem,r.get('status'),r.get('prompt',''),r.get('mode',''))
        print('模块profile:',read(root/'guard_profile/status.json').get('status','等待'))
        print('最终报告:','完成'if(root/'final_verification.json').exists()else'待完成')
        print('结果目录:',root)
        if a.once:return
        time.sleep(5)

if __name__=='__main__':main()
