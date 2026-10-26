"""Prepare the full29-layer continuation and its predeclared video comparison."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from .common import ROOT,config,atomic_json,sha,history_key


def prepare(parent,directory):
    parent,m=config(parent);root=Path(directory).resolve()
    assert json.loads((parent/'status.json').read_text())['status']=='phase_complete'
    root.mkdir(parents=True,exist_ok=False)
    old_cache=Path(m['cache']);cache=ROOT/'streaming_sparse/v10/data'/root.name
    cache.mkdir(parents=True,exist_ok=False)
    foundation=Path(m['foundation_reused_from'])
    inherited_layers=list(m['layers'])
    m.update(directory=str(root),cache=str(cache),created_at=datetime.now(timezone.utc).isoformat(),
        version='SlimAV-v0.10-full29-keep15-actualAV-e2e',parent_experiment=str(parent),
        layers=list(range(1,30)),groups=['K32-Set','V32-Set'],training_seeds=[0,1],
        train_layer_teams=2,worker_count=8,feature_namespace=root.name,runtime_query_tile=128,
        budget_locked=False,foundation_reused_from=str(foundation),
        end_to_end=dict(enabled=True,training_seed=0,generation_seeds=[0,1],
            modes=['Dense','U-Sol','K32-Set','V32-Set'],prompt_split='all registered Val prompts',
            first_chunk_dense=True,dense_layers=[0],sparse_layers=list(range(1,30)),
            candidates_locked_before_video=True,test_used=False,
            protocol='Four original latent chunks,12 latent frames,45 decoded RGB frames at16fps; pixel metrics on uncompressed RGB.'))
    # New raw-cache directories own all new writes; six captured layers are links.
    for category in ['events','history']:
        for src in (old_cache/category).glob('*/*.pt'):
            dst=cache/category/src.parent.name/src.name
            dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src)
    # The v0.8 cache lacks current raw KV, but its Dense history is reusable:
    # it does not depend on the sampled query indices or the router structure.
    legacy_reuse={}
    for ancestor in Path(m['reference_v08_checkpoint']).parents:
        identity=ancestor/'experiment.json'
        if not identity.exists():continue
        legacy=json.loads(identity.read_text())
        if not legacy.get('cache')or legacy.get('source_sha256')!=m['backbone_source_sha256']:continue
        assert legacy['teacher_sha256']==m['teacher_sha256']and legacy['replay_fingerprints']==m['replay_fingerprints']
        linked=0
        for key in {history_key(row)for row in m['rows']if row['history_tokens']}:
            for layer in m['layers']:
                src=Path(legacy['cache'])/'history'/key/f'layer{layer:02d}.pt';dst=cache/'history'/key/src.name
                if src.exists()and not dst.exists():
                    dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src);linked+=1
        legacy_reuse=dict(source=str(ancestor),linked_history_files=linked,source_identity_checked=True,replay_fingerprints_checked=True)
        break
    for category in ['parameters','initial_parameters']:(root/category).mkdir()
    parameters={}
    for layer in inherited_layers:
        name=f'layer{layer:02d}.pt'
        (root/'parameters'/name).symlink_to(parent/'parameters'/name)
        (root/'initial_parameters'/name).symlink_to(foundation/'initial_parameters'/name)
        parameters[name]=sha(parent/'parameters'/name)
    atomic_json(root/'foundation_inheritance.json',dict(parent=str(parent),parameters=parameters,
        reused_layers=inherited_layers,new_layers=[x for x in m['layers']if x not in inherited_layers],
        action_heads_inherited=False,value_basis_inherited=False))
    atomic_json(root/'legacy_data_reuse.json',legacy_reuse)
    m['P3_extension_rule']='Per layer, all4 selected group/seed runs extend together to800 when any run improves actual96-event Val NMSE >=0.5% in its last3 validations over prior best.'
    m['best_checkpoint_criterion']='Lowest actual hard attention NMSE on all96 nonfirst Val events at the new29-layer Train-calibrated beta; all action heads retrained.'
    m['progression']=['runtime_audit','extend_raw_capture','initialize_and_train_missing_K_layers',
        'Train_only_15pct_calibration','fit29_Value_bases_and_features','paired29_head_training',
        'seal_seed0_best_before_video','all_Val_closed_loop_RGB','report_and_video_index']
    atomic_json(root/'experiment.json',m)
    atomic_json(root/'status.json',dict(status='prepared',stage='N0_runtime_audit'))
    (root/'README.txt').write_text(
        '端到端方案：第0层Dense，第1–29层全部Heads稀疏；首chunk Dense。\n'
        '历史P8/P4/P2、当前P16，Train校准约15%后续chunk总KV；上下文写入Dense。\n'
        '复用六层已训练K原型，补齐23层K原型；29层低预算Value投影和K/V动作头均重新训练，不跨层借权重。\n'
        '两组×两个训练种子；主损失真实AV误差，完整96事件实际硬动作NMSE选best。\n'
        '视频使用预先固定训练seed0的best，所有注册Val提示词×生成seed0/1；Test保持封存。\n'
        '对照Dense、无动作修正U-Sol、K32、V32；四个chunk闭环生成，像素域PSNR/LPIPS与逐chunk漂移。\n'
        '流水线自动完成采集、训练、封存、视频与报告；./v10_progress.sh实时查看。\n')
    atomic_json(ROOT/'migration/unified_router_active.json',dict(directory=str(root)))
    print(json.dumps(dict(directory=str(root),raw_cache=str(cache),sparse_layers=m['layers'],
        new_foundation_layers=23,action_heads=29*2*2,video_training_seed=0)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--prepare-from',required=True)
    a=p.parse_args();prepare(a.prepare_from,a.directory)


if __name__=='__main__':main()
