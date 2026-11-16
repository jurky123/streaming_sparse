"""Publish both completed experiments with accessible full-score review archives."""
from .core import *
import shutil,tarfile,ast,hashlib
E47=ROOT/'artifacts/e47_conditional_value_v1'
def finish():
 for seed in [17,29]:
  assert json.loads((D/f's{seed}/audit.json').read_text())['passed'];assert json.loads((E47/f's{seed}/audit.json').read_text())['passed']
 p=D/'REPORT.md';s=p.read_text()+'''\n## 最终判断\n\n1. M16标签对齐的小集恢复34.90%/50.95%，低于原Q64监督37.86%/64.43%；局部MSE仍约0.669/0.665。未支持只改变监督Query范围即可恢复物理准确性和Q64决策。保持Q64收益尺度后，标签对齐本身不是当前充分修复。\n2. 原小集E43 P冻结到完整训练Risk恢复−1.04%/11.11%，Pset为2.81%/10.31%；开发分别14.28%/20.79%、17.26%/23.30%。小集可拟合不等于跨事件可靠，且已见训练事件也存在明显困难。不能将差距全部归因于跨轨迹泛化。\n3. Layer_split相对同预算Shared_T0：训练恢复5.36%→6.91%、11.39%→1.68%；开发13.89%→14.15%、17.17%→24.79%。收益和损害未跨种子一致改善，尤其seed29训练再次选中reef_fish/c5/l29/head9/t37的高损害动作，绝对NMSE增加约1.010389。追加直接求和核验确认其有害性。\n4. 因此按层解耦尚未解决泛化/可靠性，不能将结果称为确认“坐标失配是最后瓶颈”。B16共享、Value basis独立的事实与可学习性需要区分。也不根据本轮结果追加layer-head网络、学习率、clip或epoch搜索。\n\n两种子所有固定训练均完成；独立审计覆盖M16标签直接求和、稳定评分、count/own及每模型/划分最差动作。Test继续封存。完整事件级结果及最终模型在e46_review.tar.gz。\n''';p.write_text(s)
 p=E47/'REPORT.md';s=p.read_text()+'''\n## 最终判断\n\n- Pset在E45邻近留出Set恢复64.33%/75.41%；沿用Risk偏好为64.34%/75.70%。提高的小集决策能力能够延续到这个邻近集合，但重评分没有额外净收益，且损害略高。\n- trained2条件价值只占真实自适应收益0.081%/0.011%；all3为1.025%/0.255%。这说明协议本来就几乎不需要改变动作偏好，不能用它强力验证或否定复杂集合条件策略。\n- 原Pset在两个训练状态上Set相对Independent仍是一正一负；本轮没有把小集信息改善自动归为集合交互创新。\n- 不增加状态、Query或训练，不根据结果改接受规则。后续若需要检验更强条件学习，应另行预登记有代表性的状态范围，不能在本轮标签上追选“强交互”事件包装为泛化。\n\n审阅包e47_review.tar.gz已实际纳入仓库证据目录，含完整scores.npz和对应冻结模型，避免再次提供仅存在于本地artifacts的链接。\n''';p.write_text(s)
 # Source dependency closure for the exact implementation, without staging unrelated working changes.
 todo=[ROOT/'local_experiments/e46_supervision_bridge'/n for n in ['core.py','evaluate.py','audit.py','report.py','finish.py']]+[ROOT/'local_experiments/e47_conditional_value'/n for n in ['run.py','report.py']];seen=set()
 while todo:
  p=todo.pop()
  if p in seen or not p.exists():continue
  seen.add(p);mod=list(p.relative_to(ROOT).with_suffix('').parts)
  for n in ast.walk(ast.parse(p.read_text())):
   if isinstance(n,ast.ImportFrom):
    parts=mod[:-n.level] if n.level else []
    if n.module:parts+=n.module.split('.')
    q=ROOT.joinpath(*parts).with_suffix('.py')
    if q.exists():todo.append(q)
   elif isinstance(n,ast.Import):
    for a in n.names:
     if a.name.startswith('local_experiments.'):
      q=ROOT.joinpath(*a.name.split('.')).with_suffix('.py')
      if q.exists():todo.append(q)
 manifest=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(seen)]
 for root,ename in [(D,'e46_supervision_bridge_v1'),(E47,'e47_conditional_value_v1')]:
  out=ROOT/'docs/research/evidence'/ename;out.mkdir(parents=True,exist_ok=True);shutil.copy2(root/'REPORT.md',out/'REPORT.md');dump(root/'source_manifest.json',manifest)
  with tarfile.open(root/'source_snapshot.tar.gz','w:gz') as tf:
   for p in sorted(seen):tf.add(p,arcname=str(p.relative_to(ROOT)))
  audit=dict(passed=True,Test=False,source_files=len(seen),models={},events_per_seed=1440 if root==D else 144,interpretation='mechanism diagnostic, no deployment success')
  for seed in [17,29]:
   checkpoint_paths=[E29/f's{seed}/selected.pt',E43/f's{seed}/P_final.pt',E44/f's{seed}/P_set_final.pt',E39/f'T0_s{seed}/epoch1.pt']
   if root==D:checkpoint_paths+=[D/f's{seed}/aligned_final.pt',D/f's{seed}/layer_final.pt']
   for cp in checkpoint_paths:audit['models'][str(cp.relative_to(ROOT))]=sha(cp)
   for n in ['audit.json','audit_checks.csv']:
    p=root/f's{seed}'/n
    if p.exists():shutil.copy2(p,out/f's{seed}_{n}')
  dump(root/'audit.json',audit);shutil.copy2(root/'audit.json',out/'audit.json');shutil.copy2(root/'source_manifest.json',out/'source_manifest.json')
  package=root/('e46_review.tar.gz' if root==D else 'e47_review.tar.gz')
  files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.npz','.md','.pt']];fm=[dict(file=str(p.relative_to(root)),sha256=sha(p)) for p in files];dump(root/'package_manifest.json',fm)
  with tarfile.open(package,'w:gz') as tf:
   for p in files+[root/'package_manifest.json',root/'source_snapshot.tar.gz']:tf.add(p,arcname=str(p.relative_to(root)))
   for seed in [17,29]:
    tf.add(E43/f's{seed}/P_final.pt',arcname=f'reference_models/E43_P_s{seed}.pt');tf.add(E44/f's{seed}/P_set_final.pt',arcname=f'reference_models/E44_Pset_s{seed}.pt')
  shutil.copy2(package,out/package.name)
  with tarfile.open(package) as tf:
   for r in fm:assert hashlib.sha256(tf.extractfile(r['file']).read()).hexdigest()==r['sha256']
  print('PACKAGED',ename,len(files),package.stat().st_size,flush=True)
 print('FINISHED',flush=True)
if __name__=='__main__':finish()
