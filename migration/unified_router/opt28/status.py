from pathlib import Path
import json,statistics,subprocess
from .setup import ROOT

def main():
 print('Opt28｜硬件 tile 选择 → Query 聚合设计')
 print('当前基线：Opt27 Q32×K64；预算约20%；本轮未更换默认推理或启动训练。')
 for phase,label,total in [('k64_final','最终K64扫描',6),('k128','修正后的K128扫描',6),('confirm','四层 attention 复核',24),('module','初版适配器模块',24),('module_v2','最终完整模块复核',24)]:
  files=sorted((ROOT/phase).glob('*/*.json'));data=[(p,json.loads(p.read_text())) for p in files];done=sum(d.get('completed',False) for _,d in data);errors=sum('error' in r for _,d in data for r in d.get('rows',[]))
  print(f'{label}：{done}/{total} 完成，失败配置 {errors}')
 adapter=ROOT/'adapter.json'
 if adapter.exists():
  d=json.loads(adapter.read_text());print(f'选页转换优化：{len(d["rows"])}/72 组，全部一致='+str(all(r['exact']for r in d['rows'])))
 p=ROOT/'selection_verification.json'
 if p.exists():
  d=json.loads(p.read_text());print(f'独立选页校验：{len(d["rows"])}/24 个块，'+('完成' if d['completed'] else '进行中'))
 print('\n完整模块（四层平均；括号内为已有层数，部分结果仅供参考）')
 print(f'{"分辨率/总chunk":24s} {"tile":12s} {"基线ms":>8s} {"候选ms":>8s} {"加速":>7s} {"Dense NMSE":>12s}')
 for shape in sorted((ROOT/'module_v2').glob('*')):
  rr=[r for p in shape.glob('*.json') for r in json.loads(p.read_text()).get('rows',[]) if 'ms'in r]
  for g,k,w in sorted({(r['options']['q'],r['options']['k'],r['options']['warp_q'])for r in rr}):
   rows=[r for r in rr if (r['options']['q'],r['options']['k'],r['options']['warp_q'])==(g,k,w)];base=statistics.mean(r['baseline_ms']for r in rows);new=statistics.mean(r['ms']for r in rows)
   print(f'{shape.name+" ("+str(len(rows))+"/4)":24s} {str(g)+"×"+str(k):12s} {base:8.3f} {new:8.3f} {base/new:6.3f}× {statistics.mean(r["nmse_vs_dense"]for r in rows):12.5f}')
 result=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
 jobs=[s for s in result.stdout.splitlines() if s.startswith('opt28_')];print('\n仍在运行/等待：'+(', '.join(jobs) if jobs else '无'))
 print('结果目录：'+str(ROOT))
if __name__=='__main__':main()
