import json,time
from .setup import ROOT,VARIANTS
from .report import read,generate

def main():
 generate();s=read(ROOT/'summary.json');q=read(ROOT/'queue.json')
 print('Opt30｜Q32×K64对照 + 行为监督Q/K重排 + 均值初始化局部聚合')
 print('阶段：',s.get('stage'),'任务：',s['jobs']);print('权重与原始基线均保留；没有自动重试循环。')
 print('\nGPU  当前任务                         训练进度   best实际NMSE')
 for gpu,id in sorted(q.get('workers',{}).items()):
  job=next(j for j in q['jobs']if j['id']==id);args=job['args'];run=None
  if '--layer'in args and'--variant'in args:
   layer=int(args[args.index('--layer')+1]);name=args[args.index('--variant')+1];run=next((r for r in s['runs']if r['layer']==layer and r['variant']==name),None)
  progress=f'{run["updates"]}/{run["total"]}'if run else'—';best=f'{run["best"]:.6f}'if run and run['best']is not None else'—'
  print(f'{gpu:3}  {id:32} {progress:12} {best}')
 print('\n路线                          已完成层  当前best均值（仅现有层）')
 for name in VARIANTS:
  rows=[r for r in s['runs']if r['variant']==name];valid=[r['best']for r in rows if r['best']is not None]
  if rows:print(f'{name:28} {sum(r["status"]=="completed"for r in rows):2}/{len(rows):2}    '+(f'{sum(valid)/len(valid):.6f}'if valid else'—'))
 failures=[j for j in q.get('jobs',[])if j['status']in['failed','blocked','orphaned']]
 if failures:
  print('\n失败/阻断：')
  for j in failures:print(j['id'],j['status'],j.get('log',j.get('error','')))
 prep=[read(p)for p in(ROOT/'grouping').glob('layer*/completed.json')];print('\n行为树准备：',sum(x.get('version')==3 for x in prep),'/29 层（Train行为蒸馏 + 独立Train真实误差选幅度）')
 print('\n晋级路线：',s.get('promotion',{}).get('selected','待pilot完成'))
 print('已完成视频案例：',len(s['videos']))
 if s.get('video_comparison'):
  print('分辨率    路线                     案例   PSNR↑(dB)   LPIPS↓   生成秒  Dense加速')
  for v in s['video_comparison']:
   psnr=f'{v["psnr_db"]:.3f}'if v['psnr_db']is not None else'∞'
   print(f'{v["height"]:4}p     {v["variant"]:24} {v["cases"]}/4    {psnr:>8}   {v["lpips"]:.5f}  {v["seconds"]:.3f}  {v["speedup_vs_dense"]:.3f}×')
 print('查看：',ROOT/'index.html');print('视频：',ROOT/'VIDEO_INDEX.html')
if __name__=='__main__':main()
