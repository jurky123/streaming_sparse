"""Audit all required Full runs before producing a review and raw-evidence ZIP."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
import hashlib
import json
import math
import multiprocessing
import os
import shutil
from pathlib import Path
import statistics
import subprocess
import time
import zipfile
from .evaluation_journal import atomic_json
from .final_analysis import run as analyze

HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[1]
DEST=Path(os.environ.get('STREAMING_SPARSE_SCRATCH',str(HOME/'runtime')))/'review'


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit_evaluation(arguments):
    directory,expected_events,prompts,snapshot_info=arguments[:4]
    options=arguments[4] if len(arguments)>4 else {}
    partial=options.get('partial',False)
    directory=Path(directory);summary=None if partial else json.loads((directory/'summary.json').read_text())
    totals=defaultdict(lambda:[0,0.,0.]);heads=defaultdict(lambda:[0.,0.])
    h=hashlib.sha256();count=0;seen=set();current=None;keys=set();partition_hashes={};costs=None
    expected_keys={(step,tile,method,keep) for step in range(4) for tile in range(8)
                   for method in ('Ours','B4','B4_cost_capped','B0','B1','B2') for keep in (.3,.2,.1)}
    def committed_lines(stream):
        limit=options.get('raw_bytes',(directory/'per_tile.jsonl').stat().st_size)
        while stream.tell()<limit:
            line=stream.readline()
            if not line or stream.tell()>limit:raise AssertionError('Invalid committed byte boundary')
            yield line
    with (directory/'per_tile.jsonl').open('rb') as stream:
        for line in committed_lines(stream):
            if not line.endswith(b'\n'):raise AssertionError('Partial final result row')
            h.update(line);row=json.loads(line);count+=1
            version=row['cache_version']
            if version!=current:
                if current is not None and keys!=expected_keys:raise AssertionError('Snapshot tile/method/step coverage mismatch')
                if version in seen or version not in snapshot_info:raise AssertionError('Duplicate/unknown snapshot')
                seen.add(version);current=version;keys=set()
                import torch
                partition_path=directory/'partitions'/f'{version}.pt'
                partition=torch.load(partition_path,map_location='cpu',weights_only=True)
                labels=partition['labels']
                if (partition['cache_version']!=version or labels.ndim!=1
                    or len(labels)!=snapshot_info[version]['history_tokens']
                    or labels.dtype not in (torch.int32,torch.int64)
                    or int(labels.min())<0 or int(labels.max())>=256):
                    raise AssertionError('Invalid saved hard partition')
                costs=torch.bincount(labels.to(torch.int64)).tolist()
                if sum(costs)!=len(labels) or any(n==0 for n in costs):
                    raise AssertionError('Empty/missing hard partition membership')
                partition_hashes[partition_path.name]=sha256(partition_path)
            key=(row['step_id'],row['tile_index'],row['method'],row['requested_keep'])
            if key in keys:raise AssertionError('Duplicate tile result')
            keys.add(key);info=snapshot_info[version]
            if any(row[k]!=info[k] for k in ('prompt_id','seed','layer_id','head_id','history_tokens','chunk_id')):
                raise AssertionError('Snapshot metadata mismatch')
            start=info['query_tile_starts'][row['tile_index']]
            if row['query_indices']!=list(range(start,start+16)):raise AssertionError('Query position mismatch')
            if row['prompt_id'] not in prompts:raise AssertionError('Prompt split leakage')
            if not 0<=row['kept_tokens']<=row['token_budget']==int(row['requested_keep']*row['history_tokens']):
                raise AssertionError('Token budget mismatch')
            if abs(row['actual_keep']-row['kept_tokens']/row['history_tokens'])>1e-12:raise AssertionError('Keep ratio mismatch')
            selected=row['selected_blocks']
            if len(selected)!=len(set(selected)) or any(not isinstance(i,int) or i<0 for i in selected):
                raise AssertionError('Invalid selected block indices')
            if row['method'] in ('Ours','B4','B4_cost_capped'):
                if any(i>=len(costs) for i in selected) or sum(costs[i] for i in selected)!=row['kept_tokens']:
                    raise AssertionError('Selected hard partition does not match reported token cost')
            if not all(math.isfinite(row[k]) and row[k]>=0 for k in ('squared_error','dense_energy','error')):
                raise AssertionError('Nonfinite evaluation')
            key=row['method'],row['requested_keep'];t=totals[key]
            t[0]+=1;t[1]+=row['squared_error'];t[2]+=row['dense_energy']
            key=(*key,row['prompt_id'],row['layer_id'],row['head_id']);t=heads[key]
            t[0]+=row['squared_error'];t[1]+=row['dense_energy']
    if (snapshot_info and keys!=expected_keys) or seen!=set(snapshot_info):raise AssertionError('Incomplete snapshot coverage')
    expected_per_method_keep=expected_events*8
    if count!=expected_per_method_keep*18:raise AssertionError('Incomplete/duplicate result row count')
    recomputed={}
    for (method,keep),(n,se,energy) in totals.items():
        if n!=expected_per_method_keep:raise AssertionError('Method/keep coverage mismatch')
        metric=summary['metrics'][f'{method}/keep={keep}'] if summary else None
        if metric and not math.isclose(se/(energy+1e-8),metric['global_relative_error'],rel_tol=1e-8,abs_tol=1e-12):
            raise AssertionError('Raw global error disagrees with summary')
        per_prompt=defaultdict(list)
        for (m,k,p,l,head),(s,e) in heads.items():
            if (m,k)==(method,keep):per_prompt[p].append(s/(e+1e-8))
        if not partial and (set(per_prompt)!=set(prompts) or any(len(x)!=12 for x in per_prompt.values())):
            raise AssertionError('Layer/head coverage mismatch')
        macro=statistics.mean(statistics.mean(x) for x in per_prompt.values())
        recomputed[f'{method}/keep={keep}']=dict(global_relative_error=se/(energy+1e-8),observed_prompt_head_relative_error=macro,tiles=n)
        if metric and not math.isclose(macro,metric['macro_prompt_head_relative_error'],rel_tol=1e-8,abs_tol=1e-12):
            raise AssertionError('Raw prompt/head normalization disagrees with summary')
    return dict(path=str(directory),rows=count,sha256=h.hexdigest(),budget_ok=True,split_ok=True,summary_recomputed=not partial,recomputed=recomputed,
                hard_partition_labels_checked=True,same_partition_masks_checked=True,partition_sha256=partition_hashes)


def audit():
    protocol=json.loads((HOME/'protocol.json').read_text())
    coverage=json.loads((HOME/'data/full/coverage.json').read_text())
    assert coverage['events']==6912 and coverage['snapshots']==1728 and coverage['complete_cross_product']
    assert coverage['max_backend_error']<1e-4
    splits=json.loads((HOME/'data/full/splits.json').read_text())
    assert [len(splits[s]) for s in ('train','val','test')]==[16,4,4]
    assert len(set(sum(splits.values(),[])))==24
    p0=json.loads((HOME/'runs/p0/summary.json').read_text())
    assert {v['weight'] for v in p0.values()}=={0.,1e-4,1e-3}
    assert all(v['gradients_finite'] for v in p0.values())
    for variant in ('shared','conditioned'):
        for split,n in (('train',576),('val',144),('test',144)):
            pilot=json.loads((HOME/'runs/pilot'/variant/'seed0'/f'eval_{split}'/'summary.json').read_text())
            assert pilot['snapshots']==n
    reference=json.loads((HOME/'reference_v01_sha256.json').read_text())
    for name,digest in reference.items():assert sha256(ROOT/name)==digest,name
    subprocess.run(['git','diff','--exit-code'],cwd=ROOT,check=True,stdout=subprocess.PIPE)
    records=[json.loads(x) for x in (HOME/'data/full/events.jsonl').read_text().splitlines()]
    tensor_paths=sorted({r[key] for r in records for key in ('history','event')})
    def tensor_digest(relative):
        source=HOME/'data/full'/relative
        return relative,dict(bytes=source.stat().st_size,sha256=sha256(source))
    with ThreadPoolExecutor(max_workers=4) as executor:
        tensor_hashes=dict(executor.map(tensor_digest,tensor_paths))
    atomic_json(HOME/'reports/QKV_SHA256.json',tensor_hashes)
    metadata={r['cache_version']:r for r in records}
    jobs=[];training=[]
    for variant in protocol['variants']:
        for seed in protocol['router_seeds']:
            run=HOME/'runs/full'/variant/f'seed{seed}'
            marker=json.loads((run/'training_completed.json').read_text());assert marker['steps']==3000
            for name in ('best.pt','final.pt','latest.pt'):assert (run/name).stat().st_size>0
            rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows]==list(range(3000))
            assert all(math.isfinite(r['attention_loss']) and all(math.isfinite(x) for x in r['gradients'].values()) for r in rows)
            assert all(any(r['gradients'][name]>0 for r in rows) for name in ('content','aggregator','encoder','decoder'))
            if variant not in ('spatial','frozen'):assert any(r['gradients']['grouping']>0 for r in rows)
            training.append(dict(variant=variant,seed=seed,updates=len(rows),finite_gradients=True))
            for split in ('train','val','test'):
                path=run/f'eval_{split}';assert (path/'summary.json').exists(),path
                summary=json.loads((path/'summary.json').read_text())
                assert summary['snapshots']==len(splits[split])*72
                expected_oracle=72 if variant=='conditioned' and split!='train' else 0
                assert len(summary['oracle'])==expected_oracle
                versions={k:v for k,v in metadata.items() if v['prompt_id'] in splits[split]}
                jobs.append((str(path),len(splits[split])*288,splits[split],versions))
    results=[]
    with ProcessPoolExecutor(max_workers=7,mp_context=multiprocessing.get_context('spawn')) as executor:
        for future in as_completed([executor.submit(audit_evaluation,j) for j in jobs]):
            results.append(future.result())
            atomic_json(HOME/'reports/audit_progress.json',dict(completed=len(results),total=len(jobs)))
            print('Audited',len(results),'/',len(jobs),results[-1]['path'],flush=True)
    permanent=persist_archives((first,second))
    value=dict(complete=True,created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),coverage=coverage,
               frozen_v01_files=len(reference),upstream_unchanged=True,training=training,evaluations=results)
    atomic_json(HOME/'reports/FINAL_AUDIT.json',value)
    return value


def archive(path,members,text_members=None,expected_hashes=None,limits=None):
    inventory=[]
    temporary=path.with_suffix(path.suffix+'.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,source in sorted(members.items()):
            h=hashlib.sha256();size=0
            with Path(source).open('rb') as f,z.open(name,'w',force_zip64=True) as target:
                remaining=(limits or {}).get(name,Path(source).stat().st_size)
                while remaining:
                    block=f.read(min(4*1024*1024,remaining))
                    if not block:raise AssertionError('Source shorter than committed archive boundary: '+name)
                    h.update(block);size+=len(block);target.write(block);remaining-=len(block)
            if expected_hashes and name in expected_hashes and h.hexdigest()!=expected_hashes[name]:
                raise AssertionError('Source changed after audit: '+name)
            inventory.append(dict(name=name,size=size,sha256=h.hexdigest()))
        for name,text in (text_members or {}).items():
            data=text.encode();z.writestr(name,data)
            inventory.append(dict(name=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('INVENTORY.json',json.dumps(inventory,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temporary) as z:
        bad=z.testzip()
        if bad:raise AssertionError('ZIP CRC failure: '+bad)
    temporary.replace(path)
    digest=sha256(path)
    path.with_suffix(path.suffix+'.sha256').write_text(digest+'  '+path.name+'\n')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest,members=len(inventory)+1,crc_verified=True)



def persist_archives(archives):
    """Keep the delivered archives under the actual model tree, outside scratch."""
    destination=HOME.parent/'review_packages/v02'
    destination.mkdir(parents=True,exist_ok=True)
    needed=sum(item['bytes'] for item in archives)
    if shutil.disk_usage(destination).free<needed+2_000_000_000:
        raise OSError('Insufficient persistent storage for review delivery; verified scratch ZIPs preserved')
    for item in archives:
        source=Path(item['path']);target=destination/source.name
        if not target.exists() or sha256(target)!=item['sha256']:
            temporary=target.with_suffix(target.suffix+'.copying')
            shutil.copyfile(source,temporary)
            with temporary.open('rb+') as stream:os.fsync(stream.fileno())
            if sha256(temporary)!=item['sha256']:
                raise AssertionError('Persistent ZIP copy hash mismatch')
            temporary.replace(target)
        checksum=target.with_suffix(target.suffix+'.sha256')
        temporary=checksum.with_suffix(checksum.suffix+'.tmp')
        temporary.write_text(item['sha256']+'  '+target.name+'\n');temporary.replace(checksum)
        item.update(staging_path=str(source),path=str(target),persistent_copy_verified=True)
    return destination


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    result=audit();analyze()
    from .report import main as report
    report()
    core={};raw={}
    def add(name,path):
        path=Path(path)
        if path.is_file():core[name]=path
    for relative in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():
        p=ROOT/relative
        if p.suffix in ('.py','.md','.txt','.json','.yaml','.yml','.toml') or p.name=='LICENSE':
            add('code/Self-Forcing/'+relative,p)
    reference=json.loads((HOME/'reference_v01_sha256.json').read_text())
    for relative in reference:add('code/Self-Forcing/'+relative,ROOT/relative)
    for p in HOME.glob('*.py'):add('code/Self-Forcing/streaming_sparse/v02/'+p.name,p)
    for name in ('signal_audit.c','requirements.txt','README.md','DIRECTORY.md','RUNTIME.md','PARALLEL.md'):
        add('code/Self-Forcing/streaming_sparse/v02/'+name,HOME/name)
    add('code/Self-Forcing/streaming_sparse/INDEX.md',HOME.parent/'INDEX.md')
    for directory in ('configs','provenance','reports'):
        for p in (HOME/directory).glob('*'):
            if p.is_file() and p.suffix in ('.json','.md','.txt','.zip','.png'):
                add('results/v02/'+directory+'/'+p.name,p)
    for name in ('protocol.json','REPORT.md','results.json','pilot_full_gate.json','selected_regularizer.json'):
        add('results/v02/'+name,HOME/name)
    for stage in ('pilot','full'):
        for run in sorted((HOME/'runs'/stage).glob('*/seed*')):
            prefix='results/v02/runs/'+stage+'/'+run.parent.name+'/'+run.name+'/'
            for name in ('config.json','requested_config.json','training_completed.json','metrics.jsonl',
                         'validation_curve.jsonl','learning_curves.png','best.pt','train.log'):
                add(prefix+name,run/name)
            for split in ('train','val','test'):
                directory=run/f'eval_{split}'
                for name in ('summary.json','oracle.jsonl','progress.json','resume.json','legacy_adoption.json'):
                    add(prefix+f'eval_{split}/'+name,directory/name)
                source=directory/'per_tile.jsonl'
                if source.exists():raw[prefix+f'eval_{split}/per_tile.jsonl']=source
                for partition in sorted((directory/'partitions').glob('*.pt')):
                    raw[prefix+f'eval_{split}/partitions/'+partition.name]=partition
    for p in (HOME/'runs/p0').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.md','.png'):
            add('results/v02/runs/p0/'+str(p.relative_to(HOME/'runs/p0')),p)
    for stage in ('pilot','full'):
        for name in ('events.jsonl','splits.json','coverage.json'):
            add('results/v02/data/'+stage+'/'+name,HOME/'data'/stage/name)
    for name in ('parallel_commands.jsonl','commands.jsonl','workflow.log'):
        add('results/v02/logs/'+name,HOME/'logs'/name)
    for path in (HOME/'logs').glob('capture_*.log'):
        add('results/v02/logs/'+path.name,path)
    add('reference/v01_review.zip',HOME.parent/'artifacts/self_forcing_sparse_review_20261015.zip')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    readme=f'''# Self-Forcing Streaming Sparse v0.2 评审包

本包仅在 15 组 Full 训练和 45 次完整评估通过验收后生成。上游提交：{commit}。

## 阅读顺序

1. `results/v02/reports/FINAL_ANALYSIS.md`：三训练 Seed 主结果、配对 Prompt CI、模块消融和 Oracle Headroom。
2. `results/v02/REPORT.md`：完整增量报告及 P0/Pilot/Full 结果。
3. `results/v02/reports/FINAL_AUDIT.json`：完整性、预算、Prompt 划分、原始指标重算和代码冻结验收。
4. `results/v02/protocol.json` 与各 run 的配置、曲线、最佳 Router checkpoint。
5. `code/Self-Forcing/streaming_sparse/v02/`：本轮代码；基础可移植 Router/Attention 模块在同级目录。

主包包含代码、配置、报告、各次训练指标、所有汇总、Oracle 结果和最佳 Router 权重。
原始逐 Tile 结果和历史硬分区标签在独立的 `_raw_results.zip` 中；两包解压到同一目录，可从原始 SSE/Dense 能量重算报告，并核对选中 Block 的实际 Token 数。
`INVENTORY.json` 为本包其他成员的 SHA-256，包外 `.sha256` 校验 ZIP 本身；生成时已检查所有成员 CRC。

## 数据与运行

24 Prompts，按 Prompt 完全隔离 Train16/Val4/Test4；Generation Seeds0/1，Router Seeds0/1/2；3 Layers × 4 Heads × 3 History Lengths × 4 Steps；每 Event 8 个 16-token Tiles。
不使用 Oracle 标签、不更新 Backbone。Greedy 为启发式参考，负 Recovery 保留。
四个 Test Prompts 仅构成初步泛化证据。没有稀疏 Kernel 加速或完整视频质量结论。

Backbone 权重、原始 QKV 张量和 Python 环境未包含（体积较大）。索引、缓存版本及采集/训练命令均已保存，需按代码 README 下载官方权重并采集，或一起迁移原 QKV 存储。
所有代码保持在模型文件夹的 streaming_sparse 子目录；不包含代理凭据、用户配置或 .codex 日志。
'''
    stamp=time.strftime('%Y%m%d',time.gmtime())
    first=archive(DEST/f'self_forcing_sparse_v02_review_{stamp}.zip',core,{'README_REVIEW.md':readme})
    expected={}
    for entry in result['evaluations']:
        directory=Path(entry['path'])
        name='results/v02/'+str(directory.relative_to(HOME))+'/per_tile.jsonl'
        expected[name]=entry['sha256']
        for partition,digest in entry['partition_sha256'].items():
            expected['results/v02/'+str(directory.relative_to(HOME))+'/partitions/'+partition]=digest
    second=archive(DEST/f'self_forcing_sparse_v02_raw_results_{stamp}.zip',raw,
                   {'README_RAW_RESULTS.md':'# 原始结果\n\n与主评审包解压到同一目录。逐 Tile 结果与 FINAL_AUDIT.json 的 SHA-256 对应。\n'},expected_hashes=expected)
    permanent=persist_archives((first,second))
    value=dict(complete=True,created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),review=first,raw_results=second,
               audited_training_runs=len(result['training']),audited_evaluations=len(result['evaluations']))
    atomic_json(DEST/'REVIEW_COMPLETE.json',value)
    atomic_json(permanent/'REVIEW_COMPLETE.json',value)
    artifacts=HOME.parent/'artifacts'
    for item in (first,second):
        path=Path(item['path'])
        for source in (path,path.with_suffix(path.suffix+'.sha256')):
            link=artifacts/source.name
            if link.is_symlink():link.unlink()
            if link.exists():raise FileExistsError(link)
            link.symlink_to(source)
    for alias,item in (('v02_review_latest.zip',first),('v02_raw_results_latest.zip',second)):
        link=artifacts/alias
        if link.is_symlink():link.unlink()
        if link.exists():raise FileExistsError(link)
        link.symlink_to(Path(item['path']))
    marker=artifacts/'v02_REVIEW_COMPLETE.json'
    if marker.is_symlink():marker.unlink()
    if not marker.exists():marker.symlink_to(DEST/'REVIEW_COMPLETE.json')
    atomic_json(HOME/'reports/REVIEW_COMPLETE.json',value)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
