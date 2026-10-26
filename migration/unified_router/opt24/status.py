"""Read-only experiment overview, with no GPU runtime import."""
import json
from .setup import ROOT

def main():
 print('Opt24 | 复用 attention kernel / query tile 扫描')
 print('冻结 Tile20 权重；Sparge32 保持32-query选页，64/128重分组选页仅作候选测试。')
 for kind,pattern in [('自有kernel扫描','custom/*/layer*.json'),('公开库适配','library/*/*layer*.json'),('同mask大tile','union_*.json'),('Sparge CUDA','sparge_*layer*.json')]:
  paths=[p for p in ROOT.glob(pattern)if 'attempt'not in p.name];data=[json.loads(p.read_text())for p in paths]
  print(f'{kind}: {sum(d.get("completed",False)for d in data)}/{len(data)} 组完成，{sum(len(d.get("rows",[]))for d in data)} 条记录')
 print('\n端到端 8 组：45帧、层1–29稀疏，首chunk与层0 Dense')
 print('Case  状态              Dense(s)  Opt22(s)  Sparge32(s)  加速')
 for i in range(8):
  d=ROOT/'e2e'/f'case{i}';s=d/'status.json';stage=json.loads(s.read_text()).get('status','待启动')if s.exists()else'待启动';line=f'{i:>4}  {stage:<16}'
  if (d/'timing.json').exists():
   tt=json.loads((d/'timing.json').read_text())['results'];fmt=lambda n:f"{tt[n]['seconds']:8.3f}"if n in tt else'       —'
   line+=' '+ ' '.join(fmt(n)for n in ['Dense','Opt22','Sparge32'])
   if 'Sparge32'in tt:line+=f"   {tt['Opt22']['seconds']/tt['Sparge32']['seconds']:.3f}x"
  print(line)
 print('\n结果：',ROOT)
if __name__=='__main__':main()
