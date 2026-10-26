import argparse,json,time,sys
from pathlib import Path
from .setup import ROOT

def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def alive(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
 except OSError:return False
def show():
 s=read(ROOT/'final/status.json');done=len(list((ROOT/'final').glob('case*/completed.json')))
 print(f'Opt22 | 微调后的20%全层方案 | 最终验证 {done}/8 | {s.get("status","准备中")}')
 print('固定20%路由与全部权重；融合调制/残差、保持原归约顺序的RMSNorm、合并QKV投影。')
 print('\nGPU  样例  阶段           对照阶段      方案           重复  进程')
 labels={'loading':'加载','warmup':'预热','timing':'计时','profiling':'热点分析','vae_timing':'VAE计时','completed':'完成'}
 for i in range(8):
  d=read(ROOT/f'final/case{i}/status.json');state=d.get('status','等待')
  process='存活' if alive(d.get('pid')) else '结束' if state=='completed' else '未运行'
  print(f"{i:3}  {i:4}  {labels.get(state,state):<10} {d.get('phase','—'):<12} {d.get('variant','—'):<14} {str(d.get('repeat','—')):>3}  {process}")
 if (ROOT/'REPORT.txt').exists():print('\n报告：'+str(ROOT/'REPORT.txt'))
 print('\n结果目录：'+str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 try:
  while True:
   if a.watch and sys.stdout.isatty():print('\033[2J\033[H',end='')
   show()
   if not a.watch:break
   time.sleep(5)
 except KeyboardInterrupt:pass
if __name__=='__main__':main()
