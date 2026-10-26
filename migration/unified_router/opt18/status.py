import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt18_continue_frontier_20261025'

def read(p):return json.loads(p.read_text())if p.exists()else{}

def main():
    m=read(ROOT/'experiment.json');print('Opt18 best续训+计算/精度评估 | 4层 × 2预算 | 8卡\n完整Q分组上下文；实际部署输出NMSE；最终保留约15%；backbone冻结。')
    print('复用完整Q/KV；固定best分组树，额外训练最多2000步。')
    print('预检：'+('通过'if read(ROOT/'preflight.json').get('passed')else'未通过/进行中'))
    print('\nGPU  层  配置         步数         状态           Train NMSE    Val best / initial')
    names={'loading':'加载数据','initial_validation':'初始验证','training':'联合训练','validating':'验证','tree_refresh':'分组树更新','budget_validation':'计算预算评估','completed':'完成','failed':'失败','stopped':'停止'}
    for rank in range(8):
        d=read(ROOT/f'training/rank{rank}.json');loss=d.get('losses',{}).get('actual_nmse');best=d.get('best');initial=d.get('initial')
        ls=f'{loss:.6f}'if loss is not None else'-';bs=f'{best:.6f} / {initial:.6f}'if best is not None and initial is not None else'-'
        print(f'{rank:3} {str(d.get("layer","-")):>3} {d.get("variant","-"):10} {d.get("updates",0):4}/{d.get("total",m.get("max_updates",2000))}   {names.get(d.get("status"),"等待"):10} {ls:>12}   {bs}')
        if d.get('error'):print('  '+d['error'])
    completed=list((ROOT/'training').glob('layer*/*/completed.json'));print(f'\n已完成 {len(completed)}/8 任务。')
    print('Fast35：35%候选/8代表；Quality50：50%候选/16代表。')
    print('每100步完整Val选best；固定分组树；每25步保存恢复点。')
    print('训练曲线：training/layerXX/{Fast35,Quality50}/train.jsonl 与 curve.json')
    sweeps=list((ROOT/'frontier').glob('layer*/*/completed.json')); print(f'计算/精度扫描已完成 {len(sweeps)}/8；含候选预算、代表数与64/128/256桶。')
    print('实验目录：'+str(ROOT))

if __name__=='__main__':main()
