import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt17_grouped_joint_20261025'

def read(p):return json.loads(p.read_text())if p.exists()else{}

def main():
    m=read(ROOT/'experiment.json');print('Opt17 分组路由联合微调 | 4层 × 2预算 | 8卡\n完整Q分组上下文；实际部署输出NMSE；最终保留约15%；backbone冻结。')
    codes=read(ROOT/'capture/exit_codes.json');print('完整Q补采：'+('完成（1152份，Train192/Val96）'if codes==[0]*8 else'进行中'))
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
    print('每100步完整Val选best；每250步Train验证分组树提案；每25步保存恢复点。')
    print('训练曲线：training/layerXX/{Fast35,Quality50}/train.jsonl 与 curve.json')
    print('实验目录：'+str(ROOT))

if __name__=='__main__':main()
