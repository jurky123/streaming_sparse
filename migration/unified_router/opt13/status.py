import json
from pathlib import Path
root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt13_compact_20261025'
print('Opt13：Compact + query tile 投票（未使用 Aggressive）')
for f in sorted((root/'quality').glob('rank*.json')):
 d=json.loads(f.read_text());print(f"{f.stem:6s} | {d['status']:10s} | {d.get('prompt','-'):12s} | {d.get('variant','-'):10s} | round={d.get('round','-')}")
print('完整视频质量：',len(list((root/'quality').glob('*/Vote8.json'))),'/ 8')
print('成对测速完成：',len(list((root/'quality').glob('*/timing_summary.json'))),'/ 8')
