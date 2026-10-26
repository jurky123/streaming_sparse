import json,html
from .setup import ROOT,VARIANTS
def read(p):
 try:return json.loads(p.read_text())
 except(FileNotFoundError,json.JSONDecodeError):return {}
def main():
 q=read(ROOT/'queue.json');print('Opt32 动作选择 / V-aware：',q.get('stage','waiting_priority_and_opt30'))
 for j in q.get('jobs',[]):
  if j['status']=='running':print('GPU',j['gpu'],j['id'])
 rows=[]
 for name in VARIANTS:
  data=[read(p)for p in(ROOT/'training').glob(f'layer*/{name}/completed.json')]
  valid=[x for x in data if x.get('budget_feasible')]
  row=dict(variant=name,completed_layers=len(data),budget_feasible_layers=len(valid),native_nmse=sum(x['best_native_nmse']for x in valid)/len(valid)if valid else None,module_ms=sum(x['module_ms']for x in valid)/len(valid)if valid else None);rows.append(row)
  if data:print(row)
 ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'summary.json').write_text(json.dumps(dict(stage=q.get('stage'),rows=rows),indent=2))
 lines=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><h1>动作选择与V-aware配对消融</h1>',f'<p>状态：{html.escape(str(q.get("stage")))}</p>','<p>先完成Opt31优先长视频评测和Opt30基础训练，再冻结共同Q64基础。主线KEEP/等成本SWAP保持原物理执行量；动态ADD/DROP分开报告。当前为4层pilot，不能当成全层视频质量。</p>','<table border="1" cellpadding="8"><tr><th>方案</th><th>已完成层</th><th>预算合格层</th><th>实际NMSE</th><th>模块ms</th></tr>']
 for r in rows:lines.append(f'<tr><td>{r["variant"]}</td><td>{r["completed_layers"]}/4</td><td>{r["budget_feasible_layers"]}</td><td>{r["native_nmse"]}</td><td>{r["module_ms"]}</td></tr>')
 lines.append('</table><p>只在同样已完成层之间做配对比较；动态支线不因多读KV获得质量信用。best依据完整96-Val真实native输出误差与执行预算；反向使用原始128维hard-action反事实误差。</p><h2>任务</h2>')
 for j in q.get('jobs',[]):lines.append(f'<p>{html.escape(j["id"])}：{html.escape(j["status"])} <a href="logs/{j["id"]}.log">日志</a></p>')
 (ROOT/'index.html').write_text('\n'.join(lines))
if __name__=='__main__':main()
