from .run import *
import shutil,tarfile,csv

def main():
 report=DEST/'REPORT.md';report.write_text(report.read_text().split('\n\n## 独立数值复核与成本')[0])
 audits=json.loads((DEST/'independent_audit.json').read_text());ident=[json.loads((DEST/f's{s}'/'identity_audit.json').read_text()) for s in [17,29]];replays=[json.loads((DEST/f'replay_s{s}.json').read_text()) for s in [17,29]]
 provenance=dict(protocol_sha=sha(DEST/'protocol.json'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},runs=[json.loads((DEST/f's{s}'/'complete.json').read_text()) for s in [17,29]],fit_sha=sha(OUT/'fit.pt'),dataset_manifest_sha=sha(OUT/'dataset_manifest.json'),test_accessed=False,trained=False)
 dump(DEST/'provenance.json',provenance)
 ad=dict(identity_max_relative=max(x['identity_relative_max'] for rr in ident for x in rr),replay_action_mismatches=sum(not x['same_action'] for rr in replays for x in rr),replay_max_gain_difference=max(abs(x['gain_difference']) for rr in replays for x in rr),independent_LS_max_relative=max(x['augmented_LS_relative_error'] for x in audits['checks'] if 'augmented_LS_relative_error' in x),independent_score_max_relative=max(x['relative_error'] for x in audits['checks'] if 'relative_error' in x))
 ad['shared_LS_max_relative']=max(x['relative_coefficient_error'] for x in json.loads((DEST/'shared_audit.json').read_text()))
 ad['shared_LOTO_target_excluded']=all(not x['withheld_event_in_fit'] for x in json.loads((DEST/'shared_audit.json').read_text()) if x['mode']=='LOTO')
 ad['true_action_gain_max_difference']=max(x['true_gain_difference'] for x in audits['checks'] if 'true_gain_difference' in x)
 assert ad['replay_action_mismatches']==0
 dump(DEST/'audit_summary.json',ad)
 pair=12*72*16*410;cost=dict(extra_raw_KV_bytes=0,old_E27_coefficient_bytes_per_page=234,optional_projected_R4_values_bytes_per_page=4*16*2,projection_cache_history_MiB=4*16*2*338*12*29/2**20,project_values_commit_FLOPs_per_page=2*128*16+2*4*8*16,project_values_all_pages_GFLOPs=5120*410*12/1e9,project_values_current_pages_GFLOPs=5120*72*12/1e9,projected_R4_AV_GFLOPs=pair*2*4*16/1e9,analytic_center_and_coupling_approx_GFLOPs=pair*16*5/1e9,optional_projected_cache_logical_scan_GiB=128*410*12*72/2**30,note='Analytic implementation upper-level operation count; excludes existing phi/reader/exchange, normalization, exponentials, synchronization, IO reuse; no kernel or latency measured. Optional cache is NOT required, can recompute/reuse; does not imply free projection.')
 dump(DEST/'cost_estimate.json',cost)
 with (DEST/'REPORT.md').open('a') as f:
  f.write('\n\n## 独立数值复核与成本\n\n'+json.dumps(ad,ensure_ascii=False,indent=2)+'\n\n恒等式在全部576个事件检查；独立增广NumPy LS和longdouble评分抽查两个种子各4事件、3种页面位置。原Centered-Oracle64与E27逐事件核对动作/执行，全部重现。另有同页跨上下文共享/留出解的独立增广LS核验。局部目标对全局中心不依赖是代数事实；本实验只能检验固定phi下跨Query响应系数是否更容易共享，不能证明网络可预测性。\n\n预投影四个V代表到B16可缓存128B/page（可选择重算/复用而不永久缓存），历史29层额外约14.36MiB；原E27派生系数234B/page另计。逻辑全tile扫描约0.042GiB/层，低维R4 AV约0.726GFLOPs/层，中心组合约0.454GFLOPs/层，另有页面投影、指数归一化、原Reader与搜索等成本；不是实测延迟，不宣称总成本仅为少量乘加。\n\n未执行在线系数训练、Test、Kernel或视频实验。LOTO可使用其他去噪步骤或未来chunk的同页教师标签，不能解释成因果在线泛化。\n')
 with (DEST/'REPORT.md').open('a') as f:
  f.write('\n数值敏感性：原Centered系数的单位缩放与E27训练尺度在实数代数上等价，但FP64中6个近并列动作发生变化，其中2个有可见真实收益差。正式结果统一使用E27固定训练尺度，逐事件对照完全重现；全部变化见scale_sensitivity.csv，初版记录保存在before_scale_alignment。未按真实收益打破并列或选择缩放。Local各行在此次尺度对齐中不变。\n')
 stage=DEST/'review';stage.mkdir(exist_ok=True)
 for p in DEST.iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.png']:shutil.copy2(p,stage/p.name)
 for seed in [17,29]:
  dd=stage/f's{seed}';dd.mkdir(exist_ok=True)
  for n in ['complete.json','identity_audit.json','identities.json','conditional.json']:shutil.copy2(DEST/f's{seed}'/n,dd/n)
 shutil.copytree(Path(__file__).parent,stage/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 hashes={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'};dump(stage/'SHA256SUMS.json',hashes)
 with tarfile.open(DEST/'e28_review.tar.gz','w:gz') as tar:tar.add(stage,arcname='e28_review')
 print(json.dumps(ad,indent=2));print('PACKAGED',DEST/'e28_review.tar.gz')
if __name__=='__main__':main()
