import json
from pathlib import Path
ROOT=Path('streaming_sparse/v10/experiments/opt19_tile_kernels_20261025').resolve()

def main():
    print('Opt19 | tile预算 / KV准备 / 路由 / 高效attention kernel')
    for name in ['probe_Fast35.json','library_flex_reference.json','library_flashinfer_fp16.json']:
        f=ROOT/name
        if not f.exists():print(name,': 尚未产生结果');continue
        rows=json.loads(f.read_text())['rows'];ok=[r for r in rows if 'error' not in r]
        print(f'{name}: {len(ok)}个成功结果，{len(rows)-len(ok)}个失败记录')
        for r in ok[-3:]:
            print(' ',{k:r[k] for k in ['kind','group','budget','mode','warm_ms','nmse'] if k in r})
    for f in sorted((ROOT/'validation').glob('*/status.json')):
        d=json.loads(f.read_text());print(f.parent.name,d)

if __name__=='__main__':main()
