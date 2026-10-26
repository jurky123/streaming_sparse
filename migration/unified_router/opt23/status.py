"""Readable, read-only status of real-QKV capture and scaling benchmarks."""
import json
from .setup import ROOT,CASES,tag

def main():
 print('Opt23 | Tile20 规模扩展 / attention 与路由分项测速')
 print('范围：真实 Dense 轨迹 QKV，4 层 × 6 组；不是高分辨率稀疏视频质量评估。')
 print('分辨率        历史  采集   测速进度  Dense(ms)  Attention(ms)  Router(ms)  模块加速')
 total=0
 for h,w,c in CASES:
  name=tag(h,w,c);capture=ROOT/'capture'/name/'completed.json';folder=ROOT/'bench'/name
  rows=[json.loads(p.read_text()) for p in sorted(folder.glob('layer[0-9][0-9].json'))];total+=len(rows)
  text=f'{h}x{w:<7} {c-1:>3}   '+('完成' if capture.exists() else '进行中')+f'   {len(rows)}/4'
  if rows:
   mean=lambda k:sum(x['median_ms'][k] for x in rows)/len(rows)
   d=mean('dense_flash');a=mean('sparse_attention');r=mean('router');s=mean('sparse_pipeline')
   text+=f'      {d:8.3f}     {a:8.3f}    {r:8.3f}    {d/s:5.2f}x'
  print(text)
  worker=folder/'worker.json'
  if worker.exists() and json.loads(worker.read_text())['status']=='failed':print('  失败详情：',worker.read_text())
 print(f'总进度 {total}/24；已完成组的数字为对应层均值。')
 print('结果目录：',ROOT)
if __name__=='__main__':main()
