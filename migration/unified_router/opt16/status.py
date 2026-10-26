import json
from .setup import ROOT

def main():
    print('Opt16 双侧分组路由：'+str(ROOT))
    for root in sorted(ROOT.glob('layer*_chunk*')):
        if not root.is_dir():continue
        status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        rows=[]
        for name in ['results.json','low_candidates.json']:
            if (root/name).exists():rows+=json.loads((root/name).read_text())['rows']
        print(f'\n{root.name}: {status.get("status","unknown")}，已记录 {len(rows)} 组')
        print('方案       实际NMSE  token保留  执行比例  历史缓存/当前重建ms  全冷ms')
        for r in rows:
            print(f'{r["variant"]:10s} {r["attention_nmse_dense"]:.5f}  {r["keep_tokens"]:7.2%}  {r["executed_fraction"]:7.2%}       {r["history_cached_current_rebuilt_ms"]:6.3f}       {r["cold_all_kv_ms"]:6.3f}')

if __name__=='__main__':main()
