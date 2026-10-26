"""Plain-terminal and text-file status; no HTML dependency."""
import json,statistics,math
from pathlib import Path
from .setup import ROOT

def read(p):
 try:return json.loads(p.read_text())
 except(FileNotFoundError,json.JSONDecodeError):return {}

def main(brief=False):
 q=read(ROOT/'queue.json');results=[read(p)for p in(ROOT/'e2e').glob('*/*/*/result.json')];groups={};rows=[]
 for r in results:
  if not r.get('completed'):continue
  for name,d in r['variants'].items():
   if name!='Dense':groups.setdefault((r['duration_seconds'],r['height'],r['width'],name),[]).append((r,d))
 for (duration,height,width,name),items in sorted(groups.items()):
  qual=[d['quality']['sparse_frames_only']for r,d in items];mse=statistics.mean(x['mse_rgb01']for x in qual);secs=statistics.mean(d['seconds']for r,d in items);dense=statistics.mean(r['variants']['Dense']['seconds']for r,d in items)
  total=statistics.mean(d['generation_plus_vae_seconds']for r,d in items);dense_total=statistics.mean(r['variants']['Dense']['generation_plus_vae_seconds']for r,d in items)
  rows.append(dict(seconds=duration,height=height,width=width,method=name,cases=len(items),generation_seconds=secs,speedup=dense/secs,generation_plus_vae_seconds=total,speedup_with_vae=dense_total/total,psnr_db=-10*math.log10(mse)if mse else None,lpips=statistics.mean(x['lpips']for x in qual),last5s_lpips=statistics.mean(d['quality']['final5s']['lpips']for r,d in items)))
 counts={s:sum(j['status']==s for j in q.get('jobs',[]))for s in ['pending','running','done','failed','blocked','superseded']}
 jobs=[j for j in q.get('jobs',[])if j['module']=='evaluate'and j['status']!='superseded'];done=sum(j['status']=='done'for j in jobs)
 summary=dict(stage=q.get('stage','queued'),counts=counts,completed_cases=done,total_cases=len(jobs),results=rows);ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
 lines=[f'Opt31 视频评测 | {summary["stage"]} | 完成 {done}/{len(jobs)} 案例',f'任务状态：{counts}', '分辨率均为 高×宽；Q32=Best，Q64=FastReference。', '共同使用最近3个chunk滚动历史；PSNR/LPIPS相对Dense，剔除首9帧。', '速度分别报告生成、生成+VAE；不含加载/编译/MP4压制。', '', '时长  高×宽       方法    案例  生成秒  生成加速  含VAE加速  PSNR    LPIPS   尾5sLPIPS']
 for r in rows:
  method='Q32'if r['method']=='Best'else'Q64';psnr=f'{r["psnr_db"]:.2f}'if r['psnr_db']is not None else'inf'
  lines.append(f'{r["seconds"]:>2}s   {r["height"]}x{r["width"]:<5}  {method:<6} {r["cases"]}/4   {r["generation_seconds"]:6.2f}  {r["speedup"]:6.3f}x   {r["speedup_with_vae"]:6.3f}x    {psnr:>5}   {r["lpips"]:.4f}  {r["last5s_lpips"]:.4f}')
 lines+=['','正在运行：']
 for j in q.get('jobs',[]):
  if j['status']=='running':
   args=j['args'];status={}
   if j['module']=='evaluate':
    get=lambda k:args[args.index(k)+1];status=read(ROOT/f'e2e/{get("--seconds")}s/{get("--height")}x{get("--width")}/case{get("--case")}/status.json')
   lines.append(f'  GPU{j["gpu"]} {j["id"]} {status}')
 for j in q.get('jobs',[]):
  if j['status']in ['failed','blocked']:lines.append(f'  {j["status"]}: {j["id"]} 日志 {j.get("log", "")}')
 from .review import REVIEW
 lines+=['',f'视频目录：{REVIEW}',f'视频索引：{REVIEW / "视频索引.txt"}']
 content='\n'.join(lines)+'\n';(ROOT/'STATUS.txt').write_text(content);print(content,end='')
if __name__=='__main__':main()
