import json
from pathlib import Path
ROOT=Path('streaming_sparse/v10/experiments/opt19_tile_kernels_20261025').resolve()

def main():
    print('Opt19 | tile预算与多模块优化')
    for stage,label in [('validation','初筛'),('final','最终组合')]:
        files=sorted((ROOT/stage).glob('*/status.json'))
        docs=[(f.parent.name,json.loads(f.read_text())) for f in files]
        done=sum(d['status']=='completed' for _,d in docs)
        print(f'{label}: {done}/8 组完成')
        for name,d in docs:
            if d['status']!='completed':print(f"  {name}: {d['status']} {d.get('completed',0)}/96")
    summary=ROOT/'summary.json'
    if summary.exists():
        d=json.loads(summary.read_text())
        print('第24层 / Fast35 | keep  executed  NMSE      warm   cold   Dense')
        for r in d['configurations']:
            if r['layer']==24 and r['variant']=='Fast35' and (r['name']=='native32' or r['group']==32):
                print(f"{r['name']:14} {r['keep']*100:5.1f}% {r['executed']*100:5.1f}% {r['nmse']:.6f} {r['warm_ms']:5.3f} {r['cold_ms']:5.3f} {r['dense_ms']:5.3f} ms")
        print('范围：4个试验层；未做整段视频E2E；默认部署未改。')
        print('报告：'+str(ROOT/'REPORT.txt'))
    else:print('最终报告：整理中')

if __name__=='__main__':main()
