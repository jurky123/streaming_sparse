from .train import *
import shutil,tarfile

def table(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def stats(rows):
 old=sum(x['old'] for x in rows);initial=sum(x['initial'] for x in rows);final=sum(x['set'] for x in rows);gain=sum(x['set_gain'] for x in rows);oracle=sum(x['oracle_gain'] for x in rows);rel=np.array([x['set_gain']/x['initial'] for x in rows if x['initial']>1e-12]);rel=rel if len(rel) else np.array([0.]);return dict(n=len(rows),relative_valid_n=sum(x['initial']>1e-12 for x in rows),mean_absolute_swap_gain=gain/len(rows),pooled_swap_gain_pct=100*gain/initial,initial_ratio=initial/old,final_ratio=final/old,mean_swap_pct=float(100*rel.mean()),degraded=sum(x['set_gain']< -1e-12 for x in rows),positive=sum(x['set_gain']>1e-12 for x in rows),neutral_executed=sum(x['execute'] and abs(x['set_gain'])<=1e-12 for x in rows),noop=sum(not x['execute'] for x in rows),p10_pct=float(100*np.quantile(rel,.1)),worst_pct=float(100*rel.min()),pooled_recovery=gain/oracle if oracle else 0,comb_gain=sum(x['independent']-x['set'] for x in rows),independent_final_ratio=sum(x['independent'] for x in rows)/old)
@torch.no_grad()
def audit_rows(bundle,group,rows):
 err=0.;out=[];qs=torch.linspace(0,63,16,device='cuda').round().long()
 for row in rows:
  d=load(OUT/row['event']);ev=torch.load(OUT/row['event'],weights_only=True);m=bundle.layers[d['key']];s=bundle.make(d,group);c=contributions(d['q'],s,m.basis,d['log_scale']);mask,rr,aa=route(d['q'][qs],s,m,d['own'])
  if group!='B':
   cp={n:x[qs] for n,x in c.items()};zz,uu,bb=bundle.reader(d['q'][qs],s,cp);sc=predicted(zz,uu,cp['mass'],mask,rr,aa,bb);assert torch.isfinite(sc).all();j=int(sc.argmax());assert bool(sc[j]>0)==row['set_execute'];assert int(rr[j])==row['set_remove'] and int(aa[j])==row['set_add']
  N=ev['N'].cuda();Z=ev['Z'].cuda();od=ev['dense'].cuda();scale=ev['log_scale'].cuda();q=d['q'].double()
  def ref(summary,x):
   keys=summary.keys[0].double();nu=summary.mean[0].double()[:,None]+torch.einsum('ag,pak,dk->pgd',haar(q).double(),summary.detail[0].double(),m.basis[0].bfloat16().double());w=16*(torch.einsum('qd,pgd->qpg',q,keys)/math.sqrt(128)-scale[:,None,None]).exp();zh=w.sum(-1);nh=torch.einsum('qpg,pgd->qpd',w,nu);o=torch.where(x[None,:,None],N,nh).sum(1)/torch.where(x[None],Z,zh).sum(1)[:,None];return float((o-od).square().sum()/(od.square().sum()+1e-12))
  old=ref(d['old'],d['ev']['old_mask'].cuda());new=ref(s,mask);audit=dict(event=row['event'],old_fp64=old,initial_fp64=new)
  for pol in ['set','independent']:
   x=mask.clone()
   if row.get(pol+'_execute',False):x[row[pol+'_remove']]=False;x[row[pol+'_add']]=True
   assert int(x.sum())==int(mask.sum()) and x[d['own']];val=ref(s,x);audit[pol+'_fp64']=val;err=max(err,abs(val-row[pol]))
  err=max(err,abs(old-row['old']),abs(new-row['initial']));out.append(audit)
 return out,err

def cached_audit(bundle,group,seed,rows):
 path=OUT/'audit_models'/f'{group}_s{seed}.json';ck=OUT/'training'/f'{"A_matched" if group=="A" else group}_s{seed}'/'selected.pt';identity=sha(ck)
 if path.exists():
  value=json.loads(path.read_text());assert value['checkpoint_sha']==identity and value['dataset_sha']==sha(OUT/'dataset_manifest.json');return value['rows'],value['max_error']
 ar,er=audit_rows(bundle,group,rows);dump(path,dict(checkpoint_sha=identity,dataset_sha=sha(OUT/'dataset_manifest.json'),rows=ar,max_error=er));return ar,er

def run():
 assert all((OUT/'training'/f'A_matched_s{s}'/'complete.json').exists() for s in [17,29]),'Wait for matched Frozen-R4 continuation before any test decision'
 torch.set_num_threads(2);development={};alls=[];summaryrows=[];curves=[];audit=[];maxerr=0
 for seed in [17,29]:
  for group in ['A','B','C']:
   dest=OUT/'training'/f'{"A_matched" if group=="A" else group}_s{seed}';done=json.loads((dest/'complete.json').read_text());epoch=done['best']['epoch'];rows=json.loads((dest/f'dev_epoch{epoch}.json').read_text());development[(seed,group)]=rows;st=stats(rows);summaryrows.append(dict(split='development',seed=seed,group=group,**st));alls.extend(dict(seed=seed,group=group,split='development',**r) for r in rows);curves.extend(dict(phase='train',seed=seed,group=group,**r) for r in json.loads((dest/'curves.json').read_text()));
   for ep in [0,1]:curves.append(dict(phase='development',seed=seed,group=group,epoch=ep,**stats(json.loads((dest/f'dev_epoch{ep}.json').read_text()))))
   bundle=Bundle();ck=torch.load(dest/'selected.pt',map_location='cuda',weights_only=False);bundle.layers.load_state_dict(ck['layers']);bundle.reader.load_state_dict(ck['reader']);ar,er=cached_audit(bundle,group,seed,rows);maxerr=max(maxerr,er);audit.extend(dict(group=group,seed=seed,**r) for r in ar)
 gates=[]
 for seed in [17,29]:
  c=development[(seed,'C')];a=development[(seed,'A')];sc=stats(c);sa=stats(a);trajectories={tr:stats([x for x in c if x['trajectory']==tr])['mean_absolute_swap_gain'] for tr in sorted(set(x['trajectory'] for x in c))};ok=sc['initial_ratio']<=1.02 and all(x>0 for x in trajectories.values()) and sc['final_ratio']<sa['final_ratio'] and sc['mean_swap_pct']>sa['mean_swap_pct'] and sc['comb_gain']>0 and sc['pooled_recovery']>=.2;gates.append(dict(seed=seed,passed=ok,trajectory_mean_absolute_swap_gain=trajectories,C=sc,A=sa))
 passed=all(x['passed'] for x in gates);dump(OUT/'test_gate.json',dict(created=time.time(),passed=passed,seeds=gates,test_accessed_for_evaluation=False))
 if passed:
  B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda()
  for seed in [17,29]:
   for group in ['A','B','C']:
    bundle=Bundle();ck=torch.load(OUT/'training'/f'{"A_matched" if group=="A" else group}_s{seed}'/'selected.pt',map_location='cuda',weights_only=False);bundle.layers.load_state_dict(ck['layers']);bundle.reader.load_state_dict(ck['reader']);rows=evaluate(bundle,group,files('test'),B);alls.extend(dict(seed=seed,group=group,split='test',**r) for r in rows);summaryrows.append(dict(split='test',seed=seed,group=group,**stats(rows)));ar,er=audit_rows(bundle,group,rows);maxerr=max(maxerr,er);audit.extend(dict(group=group,seed=seed,**r) for r in ar)
 table('initial_quality.csv',[{k:r[k] for k in ['seed','group','split','event','trajectory','old','initial']} for r in alls]);table('set_decision.csv',alls);table('information_quality.csv',[{k:v for k,v in r.items() if k in ['seed','group','split','event','residual_nmse','beta_nmse','residual_cosine','I_sign_accuracy','B_nmse','rank_spearman','regret','set64_gain']} for r in alls]);general=[]
 for sp in sorted(set(x['split'] for x in alls)):
  for seed in [17,29]:
   for group in ['A','B','C']:
    sub=[x for x in alls if x['split']==sp and x['seed']==seed and x['group']==group]
    for dim in ['trajectory','layer','chunk','step','head']:
     for val in sorted(set(x[dim] for x in sub)):general.append(dict(split=sp,seed=seed,group=group,dimension=dim,value=val,**stats([x for x in sub if x[dim]==val])))
 budgets=[]
 for p in sorted((OUT/'training').glob('*/complete.json')):
  v=json.loads(p.read_text());budgets.append(dict(run=p.parent.name,updates=v['updates'],seconds=v['seconds'],selected_epoch=v['best']['epoch'],role='shared_warmup' if p.parent.name in ['A_s17','A_s29'] else 'main_comparison'))
 table('training_budget.csv',budgets);table('generalization.csv',general);table('training_curves.csv',curves);table('summary.csv',summaryrows);table('fp64_audit.csv',audit)
 costs=dict(cache_bytes_per_page=170,cache_MiB_29layers=170*338*12*29/2**20,original_KV_extra_reads=0,readout_GFLOPs=2*4*17*16*410*864/1e9,swap_norm_GFLOPs=2*16*16*512*864/1e9,query_projection_GFLOPs=2*128*4*16*864/1e9,page_encoder_allpages_GFLOPs=2*(712*64+64*85)*410*12/1e9,page_encoder_current72_GFLOPs=2*(712*64+64*85)*72*12/1e9,derived_cache_naive_read_GiB_layer=170*410*864/2**30)
 costs['readout_affine_phi_mass_GFLOPs']=30*16*410*864/1e9;costs['set_state_accum_GFLOPs']=17*16*(410+41)*864/1e9;costs['swap_full_scalar_arithmetic_GFLOPs']=98*16*512*864/1e9+16*512*864/1e9;costs['main_arithmetic_GFLOPs']=costs['readout_GFLOPs']+costs['query_projection_GFLOPs']+costs['readout_affine_phi_mass_GFLOPs']+costs['set_state_accum_GFLOPs']+costs['swap_full_scalar_arithmetic_GFLOPs'];table('cost_estimate.csv',[costs]);dump(OUT/'audit.json',dict(stage0=json.loads((OUT/'stage0.json').read_text())['passed'],reference_to_independent_FP64_max_abs_nmse=maxerr,masks_checked=len(audit)*2,dataset_trajectories=14,events=2016,train_files=len(files('train')),development_files=len(files('development')),test_files=len(files('test')),test_evaluated=passed,teachers='N/Z FP64 cached, never reader input; refreshed summary-based labels eachupdate',basis_train_only=True,dataset_manifest_sha=sha(OUT/'dataset_manifest.json'),fit_sha=sha(OUT/'fit.pt'),protocol_sha=sha(OUT/'protocol.json')))
 lines=['# E24：联合学习集合感知四代表','', '两份快照链路检查、14条轨迹采集、三组×两个种子的训练与开发集评估完成。测试集'+('已在开发门槛通过后冻结评价。' if passed else '未评价：开发门槛未通过。') ,'','A冻结R4、训练读出；B仅训练pooling、不做集合修正；C从A读出和原始pooling初始化联合训练。先共享训练Frozen-R4读出2304更新作为warm-up；正式A/C从完全相同的warm读出出发各训练2304更新，A冻结摘要、C更新pooling，控制额外训练步数。B从原pooling训练2304更新、不使用读出。A/C各继承同样warm-up成本，B不需要读出预训练；三组正式比较阶段预算与数据顺序相同。原warm-only A保留为辅助结果，不作为C的主对照。物理分组、basis、grouping与Risk参数冻结。规范的精确页面贡献在数据准备与数值修复后缓存，训练更新不重复原KV Attention；每次新摘要都重算监督与Risk。','', '| 划分 | Seed | 组 | 初选/OldRisk | 最终/OldRisk | 平均修正收益 | 退化数 | Oracle汇总恢复 |','|---|---|---|---:|---:|---:|---:|---:|']
 for r in summaryrows:lines.append(f"| {r['split']} | {r['seed']} | {r['group']} | {r['initial_ratio']:.5f} | {r['final_ratio']:.5f} | {r['mean_swap_pct']:+.3f}% | {r['degraded']}/{r['n']} | {r['pooled_recovery']:.2%} |")
 lines+=['','核心判断：C的收益主要来自初选质量，而非可靠的交换。两种子初选汇总NMSE改善约22.61%/17.56%；交换的汇总增益分别为−0.246%/+1.447%，Oracle恢复−1.78%/+11.30%，未达到20%门槛。Set在同信息下优于Independent并不足以说明值得执行：它仍可能比不交换差。这里的汇总增益是NMSE总和之比，不同于逐实例相对改善的均值。正负动作以绝对收益1e-12容差分列，执行但近零的动作另列neutral_executed，避免与no-op混淆。同信息Set/Independent差值、两条开发轨迹各自收益及完整成功条件见test_gate.json。没有依据Test调模型；测试轨迹预先采集不等于已评价。B没有新读出，其最终值等于初选，不能将其no-op当学到执行判断。','', '数据严格按prompt轨迹隔离8/2/4，每轨迹144实例，不能视为2016条独立视频。这些prompt在仓库历史开发中出现过，不能声称全新场景。16维基与固定训练尺度只用1152训练实例。两个快照只用于四步实现检查，不进入正式划分。','', 'Page Encoder为712→64→85，Query128→4，现有R4 logits共享，输出85个BF16系数。阶段0验证真实集合恒等式、pooling梯度和标签刷新；未训练重建对照的开发汇总NMSE/原缓存为0.99999957，单独记录浮点路径误差；单head重建原BF16摘要有少量量化单位差异，保存原缓存作为OldRisk参照。最终独立FP64核验见fp64_audit.csv与audit.json。','', '成本主算术项与缓存见cost_estimate.csv；不包含全部指数、归一化、访存、状态更新、调度和kernel启动，无实测加速声明。170B/page为派生缓存而非新增原始统计，当前页面仍需刷新编码。完整标量主算术计数约1.76GFLOPs/层（另列当前页编码），不计exp/tanh的硬件代价；理论计数不是实测延迟。没有实现kernel或端到端视频评价。','', '数值预检发现部分logit范围超过1300；页面logZ/归一化输出作为规范缓存，线性贡献仅是有限精度公共尺度视图。质量标签用logZ差，FP64集合计算，无实例裁剪。固定尺度为训练事件RMS中位数；接近数值底噪（NMSE≤1e-12）的实例保留在绝对与汇总指标，单例相对百分比单列有效数。初选/修正/信息诊断/分层数据/成本/训练曲线六张表均保存。所有训练曲线与每epoch开发结果位于training/，模型selected.pt可复现。失败或未评价项目按上述边界保留，不把残差回归改进等同于组合决策成功。']
 lines += ['', '补充审计：匹配预算的Frozen-R4延续训练是在本轮执行中、Test评价前补充的公平性对照，并非最初预登记内容。两个种子的全部四个预定Joint检查点均检查了Oracle恢复门槛，最高为14.97%，没有检查点达到20%；详见all_checkpoint_gate_audit.json。主表检查点按初选约束下最低最终开发NMSE选择，因此主表不是按Oracle恢复率挑选的最好结果。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');pkg=OUT/'review';pkg.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.md','.csv','.json']:shutil.copy2(p,pkg/p.name)
 shutil.copy2(OUT/'fit.pt',pkg/'fit.pt')
 for seed in [17,29]:shutil.copy2(OUT/'training'/f'A_s{seed}'/'selected.pt',pkg/f'A_warm_s{seed}_selected.pt')
 for seed in [17,29]:
  for group in ['A','B','C']:shutil.copy2(OUT/'training'/f'{"A_matched" if group=="A" else group}_s{seed}'/'selected.pt',pkg/f'{group}_s{seed}_selected.pt')
 shutil.copytree(ROOT/'local_experiments/e24_joint_representatives',pkg/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(OUT/'provenance.json',dict(torch_version=torch.__version__,cuda=torch.version.cuda,source_hashes={p.name:sha(p) for p in (ROOT/'local_experiments/e24_joint_representatives').glob('*.py')},checkpoints={p.parent.name:sha(p) for p in (OUT/'training').glob('*/selected.pt')},dataset_manifest_sha=sha(OUT/'dataset_manifest.json')))
 shutil.copy2(OUT/'provenance.json',pkg/'provenance.json')
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(OUT/'e24_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e24_review')
 print('E24 COMPLETE test_evaluated',passed)
if __name__=='__main__':run()
