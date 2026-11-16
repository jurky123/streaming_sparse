from .run import *
import shutil,tarfile,hashlib,ast

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def finish():
 assert json.loads((D/'audit.json').read_text())['passed'];out=ROOT/'docs/research/evidence/e49_numeric_v1';out.mkdir(parents=True,exist_ok=True)
 p=D/'REPORT.md';s=p.read_text()+'''\n## 最终判定\n\n全部8640模型事件中，统一归约与交换增量路径均未改变最高动作；区间规则也没有拒绝任何历史正提议（Set/Independent均如此）。因此各基线训练/开发的净收益、H_gain、H_loss、退化与最差尾部全部保持原值。Full_Pset开发恢复仍26.00%/28.85%，退化仍69/73例，最大损害问题未被数值协议修复。\n\n参考事件中，旧两条归约路径对同一mask确有非零差，最大输出差范数约4.40e−16、单Query raw no-op收益绝对值约9.26e−16；统一路径的重复mask差为0。但不能由“存在差异”推出它改变了实际最高动作。全量排名没有变化，高精度确认所选微小正值不是纯底噪。\n\n两个原E29灾难动作的raw高精度gain约4.9190e−11、1.0500e−13，均落在严格正区间；它们应列为可分辨的模型误判。T0/Full_Pset在seed29灾难事件最高分为负而no-op，在seed17仍给可分辨小正分并保留错误动作。不能用一个统一归一化epsilon将这些归类为数值bug。\n\n本轮不扩大区间、不调实际收益门槛，也不启动温度/范数、pooling或Reader训练。数值规则只审计代理计算，不构成真实安全保证。下一步若研究接受策略或新表示，必须另立协议。\n''';p.write_text(s);shutil.copy2(p,out/'REPORT.md')
 for n in ['platform.json']:
  shutil.copy2(D/n,out/n)
 # Freeze exact input arrays for independent reference reconstruction without rerunning Reader.
 for seed in [17,29]:
  data=Data(seed);mpaths={'E29':E29/f's{seed}/selected.pt','T0':E39/f'T0_s{seed}/epoch1.pt','Full_Pset':E48/f's{seed}/fullset_final.pt'};models={n:model_from(cp) for n,cp in mpaths.items()};refs=[]
  for kind in ['tails','all']:refs+=read(D/f'reference_{seed}_{kind}.csv')
  seen=set()
  for ref in refs:
   key=(ref['model'],ref['event'])
   if key in seen:continue
   seen.add(key);d=data.load(ref['event'])
   with torch.no_grad():rp,bp=models[key[0]](d['s'],d['F'])
   fn=D/f'reference_input_{seed}_{len(seen):02d}.npz';np.savez_compressed(fn,model=key[0],event=key[1],rp=rp.cpu().numpy(),beta=bp.cpu().numpy(),logmass=d['lm'].cpu().numpy(),center=d['center'].cpu().numpy(),mask=d['mask'].cpu().numpy(),r=d['r'].cpu().numpy(),a=d['a'].cpu().numpy(),real_gains=d['g'].cpu().numpy())
 # Exact source closure, including uncommitted dependency versions, without modifying production files.
 todo=list((ROOT/'local_experiments/e49_numeric').glob('*.py'));seen=set()
 while todo:
  f=todo.pop()
  if f in seen or not f.exists():continue
  seen.add(f);mod=list(f.relative_to(ROOT).with_suffix('').parts)
  for n in ast.walk(ast.parse(f.read_text())):
   if isinstance(n,ast.ImportFrom):
    ps=mod[:-n.level] if n.level else []
    if n.module:ps+=n.module.split('.')
    q=ROOT.joinpath(*ps).with_suffix('.py')
    if q.exists():todo.append(q)
 seen.add(ROOT/'local_experiments/e49_numeric/interval.cpp');dump(D/'source_manifest.json',[dict(file=str(f.relative_to(ROOT)),sha256=sha(f)) for f in sorted(seen)])
 with tarfile.open(D/'source_snapshot.tar.gz','w:gz') as t:
  for f in sorted(seen):t.add(f,arcname=str(f.relative_to(ROOT)))
 files=[p for p in D.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.npz','.md']];common=out/'reference_review.tar.gz'
 with tarfile.open(common,'w:gz') as t:
  for f in files:t.add(f,arcname=f.name)
  t.add(D/'source_snapshot.tar.gz',arcname='source_snapshot.tar.gz')
 packages=[]
 for seed in [17,29]:
  for part in range(4):
   dest=D/f's{seed}_part{part}';package=out/f's{seed}_scores_part{part}.tar.gz';ff=list(dest.iterdir());manifest=[dict(file=f.name,sha256=sha(f)) for f in ff if f.is_file()];dump(dest/'package_manifest.json',manifest)
   with tarfile.open(package,'w:gz') as t:
    for f in dest.iterdir():
     if f.is_file():t.add(f,arcname=f.name)
   assert package.stat().st_size<95000000;packages.append(dict(file=package.name,sha256=sha(package),bytes=package.stat().st_size))
 packages.append(dict(file=common.name,sha256=sha(common),bytes=common.stat().st_size));dump(out/'packages.json',packages);shutil.copy2(D/'source_manifest.json',out/'source_manifest.json');print('PACKAGED',sum(r['bytes'] for r in packages),flush=True)
if __name__=='__main__':finish()
