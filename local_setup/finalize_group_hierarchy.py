import json,time,subprocess
from pathlib import Path
src=Path(__file__).resolve().parents[1];root=Path('/tmp/group_hierarchy_results');py='/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python'
while True:
 try:q=json.loads((root/'queue.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
 if len(list((root/'video').glob('*/*/*/result.json')))==168 and not q['pending'] and not q['active']:break
 if not q['pending'] and not q['active'] and q['failed']:raise RuntimeError(q['failed'])
 time.sleep(5)
for s in ['report_group_hierarchy.py','audit_group_hierarchy.py','visualize_group_hierarchy.py','conclude_group_hierarchy.py']:
 subprocess.run([py,str(src/'local_setup'/s)],cwd=src,check=True)
print('ALL RESULTS AUDITED AND REPORTED',flush=True)
