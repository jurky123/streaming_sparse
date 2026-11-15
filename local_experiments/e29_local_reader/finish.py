from .report import *
import shutil,tarfile

def main():
 assert (E29/'independent_audit.json').exists();gate=json.loads((E29/'test_gate.json').read_text())
 if gate['passed']:assert (E29/'test'/'complete.json').exists(),'Gate passed: finish the authorized once-only Test first'
 audit=json.loads((E29/'independent_audit.json').read_text());stage=json.loads((E29/'stage0.json').read_text());runs=[json.loads((E29/f's{s}'/'training_complete.json').read_text()) for s in [17,29]]
 provenance=dict(protocol_sha=sha(E29/'protocol.json'),stage0=stage,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},runs=runs,B16_fit_sha=sha(OUT/'fit.pt'),dataset_manifest_sha=sha(OUT/'dataset_manifest.json'),test_evaluated=gate['passed'],raw_data_changed=False)
 dump(E29/'provenance.json',provenance)
 summary=dict(action_checks=sum(r['decisions'] for r in audit['counts']),all_count_own_valid=all(r['all_count_own_panel_valid'] for r in audit['counts']),summary_only_exact_actions=sum(r['summary_only_same_action'] for r in audit['checks']),summary_only_checked=len(audit['checks']),max_summary_only_score_difference=max(r['summary_only_score_max_difference'] for r in audit['checks']),max_independent_predicted_gain_error=max(r['predicted_score_reference_error'] for r in audit['checks']),max_true_gain_error=max(r['true_gain_error'] for r in audit['checks']),frozen_C_phi=all(r['frozen_C_exact'] and r['frozen_phi_exact'] for r in runs),training_updates=[r['updates'] for r in runs]);dump(E29/'audit_summary.json',summary)
 pair=864*16*410;cost=dict(page_encoder_parameters=51157,reader_coeff_bytes=170,frozen_phi_coeff_bytes=64,coefficient_cache_total_bytes_per_page=234,optional_B16_projected_R4_values_cache_bytes=128,optional_extra_history_MiB=128*338*12*29/2**20,raw_KV_probe_bytes=0,additional_projected_R4_AV_GFLOPs=pair*2*4*16/1e9,additional_center_coupling_approx_GFLOPs=pair*16*5/1e9,projection_commit_FLOPs_per_page=5120,latency_measured=False,note='Same Reader capacity and candidate budget, not literally free analytic reconstruction. Optional projected cache can be recomputed; excludes existing phi/Reader/exchange and exponentials/IO/launch overhead. No kernel.')
 dump(E29/'cost_estimate.json',cost)
 rows=list(csv.DictReader((E29/'decision_summary.csv').open()));checks=list(csv.DictReader((E29/'checkpoints.csv').open()));curves=list(csv.DictReader((E29/'training_curves.csv').open()))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for a,seed in zip(ax,[17,29]):
  methods=['E27-B','Local-Online','Local-TrueBeta','Local-TrueResponse','E28-LocalLOTO'];vals=[100*float(next(r['R_executed'] for r in rows if int(r['seed'])==seed and r['method']==m and r['policy']=='Set')) for m in methods];a.bar(range(5),vals,color=['#999999','#238b83','#caa44b','#caa44b','#caa44b']);a.set_xticks(range(5),['E27-B','Local\nOnline','True\nbeta','True\nresponse','LOTO\nteacher'],rotation=25);a.axhline(0,color='black',lw=.8);a.set_title(f'Frozen C seed {seed}');a.set_ylabel('Oracle gain recovery (%)')
 fig.tight_layout();fig.savefig(E29/'decision_quality.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4))
 for seed in [17,29]:
  rr=[r for r in checks if int(r['seed'])==seed];ax.plot([int(r['epoch'])+1 for r in rr],[100*float(r['recovery']) for r in rr],'-o',label=f'seed {seed}')
 ax.axhline(20,color='gray',ls='--',label='20% recovery component');ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Epoch');ax.set_ylabel('Development online recovery (%)');ax.legend();fig.tight_layout();fig.savefig(E29/'development_curve.png',dpi=180);plt.close(fig)
 report=E29/'REPORT.md';report.write_text(report.read_text().split('\n\n## 数值审计与成本')[0])
 with report.open('a') as f:
  f.write('\n\n## 数值审计与成本\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n独立审计从Query与摘要重新计算logmass和中心，不接受原始KV、教师质量或Dense作为预测输入；动作与评价路径一致。longdouble混合输出独立复核预测分数及真实动作收益，另对全部主/教师替换动作检查预算、own和合法面板。离线Dense能量仅用于NMSE单位，不改变选页或no-op。\n\nReader仍为51157参数，页面派生系数170B＋冻结phi64B，总计234B/page；并未满足旧192B目标。新的解析路径需要投影摘要V和组合中心：可选缓存四个16维投影代表128B/page（历史29层约14.36MiB），或重算/复用；预计新增低维R4 AV约0.726GFLOPs/层，组合约0.453GFLOPs/层，另有commit投影、指数归一化和IO。原始KV探测为0，但不声称总在线成本完全不变，也未测融合延迟。\n')
 stage=E29/'review';stage.mkdir(exist_ok=True)
 for p in E29.iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.png']:shutil.copy2(p,stage/p.name)
 for seed in [17,29]:
  dd=stage/f's{seed}';dd.mkdir(exist_ok=True)
  for name in ['selected.pt','training_complete.json','analysis_complete.json','local_scale.json'] :shutil.copy2(E29/f's{seed}'/name,dd/name)
 shutil.copytree(Path(__file__).parent,stage/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'));hashes={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'};dump(stage/'SHA256SUMS.json',hashes)
 with tarfile.open(E29/'e29_review.tar.gz','w:gz') as t:t.add(stage,arcname='e29_review')
 print('PACKAGED',E29/'e29_review.tar.gz',flush=True)
if __name__=='__main__':main()
