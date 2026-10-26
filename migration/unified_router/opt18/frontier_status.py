import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt18_continue_frontier_20261025'

def main():
    path=ROOT/'frontier_queue.json'
    queue=json.loads(path.read_text())if path.exists()else{}
    print('\n计算/精度实测：'+queue.get('status','尚未启动'))
    for rank in range(8):
        path=ROOT/f'frontier/rank{rank}.json'
        if not path.exists():continue
        d=json.loads(path.read_text());print(f'GPU{rank} L{d.get("layer")} {d.get("variant"):10s} {d.get("completed",0):2}/{d.get("total","?")} {d.get("status"):10s} {d.get("config","")}')
        if d.get('error'):print('  '+d['error'])

if __name__=='__main__':main()
