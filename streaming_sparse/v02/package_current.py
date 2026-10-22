"""Package a paused experiment, committed evidence, and portable resume assets."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
import gzip
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tarfile
import time

from .evaluation_journal import atomic_json
from .review_pack import archive, audit_evaluation, sha256

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
DEST = HOME.parent / 'review_packages/v02_paused_20261016'
PREFIX = 'Self-Forcing/'
OMIT = {'__pycache__', '.venv', 'interrupted', 'validation_replayed'}


def progress(phase, **values):
    value = dict(phase=phase, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), **values)
    atomic_json(HOME / 'reports/paused_pack_progress.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def add_tree(members, logical, source, suffixes=None):
    source = Path(source).resolve()
    if not source.exists():
        return
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if path.is_file() and not (set(rel.parts) & OMIT):
            if suffixes is None or path.suffix in suffixes:
                members[PREFIX + str(Path(logical) / rel)] = path


def audit_one(args):
    import torch
    torch.set_num_threads(1)
    return audit_evaluation(args)


def inspect_evaluations():
    records = [json.loads(x) for x in (HOME / 'data/full/events.jsonl').read_text().splitlines()]
    metadata = {r['cache_version']: r for r in records}
    splits = json.loads((HOME / 'data/full/splits.json').read_text())
    protocol = json.loads((HOME / 'protocol.json').read_text())
    jobs, states = [], []
    for variant in protocol['variants']:
        for seed in protocol['router_seeds']:
            run = HOME / 'runs/full' / variant / f'seed{seed}'
            assert json.loads((run / 'training_completed.json').read_text())['steps'] == 3000
            for split in ('train', 'val', 'test'):
                path = run / f'eval_{split}'
                commit = json.loads((path / 'resume.json').read_text())
                versions = sorted(v for v, r in metadata.items() if r['prompt_id'] in splits[split])
                done = commit['snapshots_done']
                complete = (path / 'summary.json').exists()
                assert 0 <= done <= len(versions)
                if complete:
                    assert done == len(versions)
                info = {v: metadata[v] for v in versions[:done]}
                options = dict(partial=not complete, raw_bytes=commit['raw_bytes'])
                jobs.append((str(path), done * 4, splits[split], info, options))
                states.append(dict(variant=variant, seed=seed, split=split, path=str(path),
                    complete=complete, committed_snapshots=done, total_snapshots=len(versions),
                    raw_bytes=commit['raw_bytes'], oracle_bytes=commit['oracle_bytes'],
                    omitted_uncommitted_bytes=(path / 'per_tile.jsonl').stat().st_size-commit['raw_bytes']))
    progress('audit_committed_results', evaluations=len(jobs))
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(audit_one, jobs))
    for state, result in zip(states, results):
        state['audit'] = result
    return states


class ParallelGzip:
    """Bounded parallel concatenated gzip members, readable by standard tar/gzip."""
    def __init__(self, target):
        self.target = target
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.pending = deque()
        self.buffer = bytearray()

    def write(self, data):
        self.buffer.extend(data)
        chunk = 16 * 1024 * 1024
        while len(self.buffer) >= chunk:
            block = bytes(self.buffer[:chunk])
            del self.buffer[:chunk]
            self.pending.append(self.pool.submit(gzip.compress, block, compresslevel=1, mtime=0))
            if len(self.pending) >= 16:
                self.target.write(self.pending.popleft().result())
        return len(data)

    def finish(self):
        if self.buffer:
            self.pending.append(self.pool.submit(gzip.compress, bytes(self.buffer), compresslevel=1, mtime=0))
            self.buffer.clear()
        while self.pending:
            self.target.write(self.pending.popleft().result())
        self.pool.shutdown()


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        return value


def migration_archive(path, members, texts, limits):
    temporary = path.with_suffix(path.suffix + '.partial')
    inventory = []
    total = sum(limits.get(n, p.stat().st_size) for n, p in members.items())
    consumed = 0
    last = 0
    with temporary.open('wb') as output:
        compressor = ParallelGzip(output)
        with tarfile.open(fileobj=compressor, mode='w|', bufsize=1024*1024, copybufsize=1024*1024) as tar:
            for name, source in sorted(members.items()):
                size = limits.get(name, source.stat().st_size)
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = size, source.stat().st_mode & 0o777, int(source.stat().st_mtime)
                with source.open('rb') as stream:
                    reader = DigestReader(stream)
                    tar.addfile(info, reader)
                    inventory.append(dict(name=name, size=size, sha256=reader.digest.hexdigest()))
                consumed += size
                if time.monotonic()-last > 15:
                    progress('writing_migration', bytes_done=consumed, bytes_total=total, files=len(inventory))
                    last = time.monotonic()
            for name, value in texts.items():
                data = value.encode()
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
                inventory.append(dict(name=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
            data = json.dumps(inventory, ensure_ascii=False, indent=2).encode()
            info = tarfile.TarInfo(PREFIX + 'MIGRATION_INVENTORY.json')
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
        compressor.finish()
        output.flush()
        os.fsync(output.fileno())
    progress('verify_migration', files=len(inventory), uncompressed_bytes=total)
    expected = {r['name']: r for r in inventory}
    seen = set()
    with gzip.open(temporary,'rb') as decoded, tarfile.open(fileobj=decoded,mode='r|') as tar:
        for item in tar:
            if not item.isfile() or item.name in seen:
                raise AssertionError('Migration archive has a symlink, non-file or duplicate')
            seen.add(item.name)
            stream = tar.extractfile(item)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(4*1024*1024), b''):
                h.update(block)
            if item.name in expected:
                r = expected[item.name]
                assert item.size == r['size'] and h.hexdigest() == r['sha256'], item.name
        while decoded.read(4*1024*1024):pass  # Verify CRC through the final gzip member.
    assert seen == set(expected) | {PREFIX+'MIGRATION_INVENTORY.json'}
    temporary.replace(path)
    digest = sha256(path)
    path.with_suffix(path.suffix+'.sha256').write_text(digest+'  '+path.name+'\n')
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest,
                members=len(seen), all_member_hashes_verified=True, no_symlinks=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert (HOME / 'supervisor.stop').exists(), 'Pause before packaging'
    assert json.loads((HOME / 'status.json').read_text())['state'] == 'paused'
    DEST.mkdir(parents=True, exist_ok=True)
    states = inspect_evaluations()
    complete = sum(s['complete'] for s in states)
    train_done = sum(s['committed_snapshots'] for s in states if s['split']=='train')
    state = dict(experiment_complete=False, paused=True, automatic_resume_disabled=True,
        training_completed=15, training_total=15, evaluations_completed=complete, evaluations_total=45,
        train_evaluation_snapshots_done=train_done, train_evaluation_snapshots_total=17280,
        created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()), evaluations=states,
        remaining='Finish the 15 committed-prefix Full Train evaluations, then final audit and final review package; await user instruction.')
    atomic_json(HOME/'reports/MIGRATION_STATE.json',state)
    reference=json.loads((HOME/'reference_v01_sha256.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in reference.items())
    subprocess.run(['git','diff','--exit-code'],cwd=ROOT,check=True,stdout=subprocess.PIPE)
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','matplotlib')}
    atomic_json(HOME/'reports/environment_selected.json',versions)
    (HOME/'requirements-offline.txt').write_text('\n'.join(name+'=='+version.split('+')[0] for name,version in versions.items())+'\n')
    (HOME/'reports/CURRENT_REVIEW.md').write_text(f'''# v0.2 阶段评审与迁移状态

本轮按用户要求暂停，GPU 已释放，自动续跑关闭。实验尚未全部结束。

- P0、Pilot、Full 数据采集完成：24 Prompts，6912 Events，1728 History Snapshots。
- Full 训练 15/15 完成，各 3000 次更新，5 Variants × 3 Router Seeds。
- 全部 Val/Test 30/30 完成；Full Train 仍为 15 项部分评估，已提交 {train_done}/17280 Snapshots。
- 本次对 45 组完整或部分评估进行已提交前缀审计，核验覆盖、Prompt 划分、预算、硬分区标签及选中 Block 成本；完整评估重算主指标。

## 已有测试结论

完整数值、四 Prompt 配对区间、分 Head 绝对误差与 Oracle 诊断见 [TEST_ANALYSIS.md](TEST_ANALYSIS.md)。

1. Layer/Head 条件化优于不含 ID 的共享 Router。
2. Learned Grouping 明显优于 Frozen 随机特征分组，但仍不如 Spatial Grouping + Learned Router。
3. 两层 Transformer 未优于参数量相近的独立 Block MLP；Keep 30%/20% 时独立 MLP 略好，10% 的配对区间跨零。
4. Conditioned 的 Ours / 同分区 Mass 在 Keep 30%/20%/10% 为 1.650×/1.742×/1.728×，尚未超过 Mass。
5. 抽样同分区 Greedy 存在改善空间：显著 Headroom 为 44/72、31/72、33/72 Cases，多数有效 Recovery 为负。它是启发式诊断，实际 Token 成本可能不同。

测试仅包含四个独立 Prompt，以上为初步泛化证据。部分 Train 评估不是完整训练集指标，不据此判断过拟合或 Decoder 的最终瓶颈。本轮不启动 P4、不训练 Backbone、不使用 Oracle Mask 标签。

## 交付

主评审 ZIP 包含代码、配置、报告、曲线、已完成汇总、逐步训练记录和最佳 Router 权重。
证据 ZIP 包含所有已提交逐 Tile 结果与硬分区；未提交尾部省略并记录字节数。
迁移 tar.gz 包含可移植模型代码树、现有 QKV、best/final/latest/initial、评估断点和历史诊断。所有目录链接展开为普通文件。
Backbone 权重和 Python 环境不包含；继续当前 Router 评估不需要 Backbone。环境版本与重建说明随包提供。

继续工作须等待用户明确通知；迁移和解压均不会启动 GPU 任务。详见 MIGRATION_GUIDE.md 和 MIGRATION_STATE.json。
''')
    guide='''# 平台迁移与续跑

## 当前状态

实验已暂停。15 组 Full 训练、全部 30 项 Val/Test 已完成；剩余是 15 项 Full Train 评估及最终审计打包。保留 supervisor.stop，等待用户明确通知后再继续。

## 迁移

1. 传输 migration.tar.gz 及对应 .sha256；先运行 sha256sum -c 对应校验文件。
2. 在目标目录运行 tar -xzf self_forcing_sparse_v02_migration_20261016.tar.gz，得到 Self-Forcing/。
3. cd Self-Forcing；可用 python3 streaming_sparse/v02/verify_migration.py 校验所有解压文件。该命令不使用 GPU。
4. 用 Python 3.10 重建 streaming_sparse/.venv，安装与目标 CUDA 驱动兼容的 PyTorch 2.6.0，再安装 streaming_sparse/v02/requirements-offline.txt 中其他依赖。原环境 torch 2.6.0+cu124；精确版本见 environment_selected.json。当前离线 Router 评估无需 FlashAttention 或视频 Backbone。
5. gcc -shared -fPIC -O2 -Wall -Wextra streaming_sparse/v02/signal_audit.c -o streaming_sparse/v02/libsignal_audit.so；还需要 strace、tmux。

包中 QKV 只保存 shared 一份，Pilot/Full 使用相对索引；不依赖原机器 /tmp 链接，不包含 .venv、代理凭据或用户全局配置。包含公共上游 Git 对象与干净源代码，后续最终审计可核验冻结 v0.1 与上游提交。Backbone 权重可按原下载脚本重新下载；继续剩余 15 项评估不需要这些权重。

## 等用户通知后再续跑

先用 nvidia-smi 确认可用物理 GPU，按实际卡号替换下列示例中的 0；可以提供多个卡号。保留原数据、checkpoint、配置和 resume.json，不手工改提交偏移。

```bash
# 以下命令只在用户明确要求继续后执行
rm streaming_sparse/v02/supervisor.stop
tmux new-session -d -s self-forcing-sparse-v02 -c "$PWD" \\
  "$PWD/streaming_sparse/.venv/bin/python -u -m streaming_sparse.v02.supervisor --gpu 0 --gpus 0 --max-restarts 2 >> '$PWD/streaming_sparse/v02/logs/supervisor.log' 2>&1"
tmux set-option -t self-forcing-sparse-v02 remain-on-exit on
```

现有完成标记会跳过全部训练和已完成 Val/Test；15 项 Train 从 resume.json 的 Snapshot 边界恢复。评估每卡最多两项，策略位于 configs/evaluation_concurrency.json。原机器 PID 文件不迁移，自动续跑服务不自动启动。全部 45 项评估完成后，正常工作流才生成最终审计和最终评审包。
'''
    (HOME/'reports/MIGRATION_GUIDE.md').write_text(guide)
    core={}
    def add(relative,path):
        p=Path(path)
        if p.is_file():core[PREFIX+relative]=p
    for rel in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():add(rel,ROOT/rel)
    for directory in ('capture','router','attention','train','evaluation','integration','scripts','experiments'):
        add_tree(core,'streaming_sparse/'+directory,HOME.parent/directory,{'.py','.md','.txt','.json','.yaml','.yml','.sh'})
    for p in HOME.parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.txt','.json'):add('streaming_sparse/'+p.name,p)
    for p in HOME.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.txt','.json','.c','.zip') and not p.name.endswith('.pid'):
            add('streaming_sparse/v02/'+p.name,p)
    add('streaming_sparse/v02/supervisor.stop',HOME/'supervisor.stop')
    for name in ('configs','provenance','reports'):
        add_tree(core,'streaming_sparse/v02/'+name,HOME/name)
    add_tree(core,'streaming_sparse/v02/runs/p0',HOME/'runs/p0',{'.json','.jsonl','.md','.png'})
    for stage in ('pilot','full'):
        for run in sorted((HOME/'runs'/stage).glob('*/seed*')):
            rel=str(run.relative_to(ROOT))
            for name in ('config.json','requested_config.json','training_completed.json','metrics.jsonl',
                         'validation_curve.jsonl','learning_curves.png','best.pt','train.log'):
                add(rel+'/'+name,run/name)
            for split in ('train','val','test'):
                for name in ('summary.json','resume.json','progress.json','oracle.jsonl','legacy_adoption.json'):
                    add(rel+'/eval_'+split+'/'+name,run/f'eval_{split}'/name)
    for stage in ('pilot','full'):
        for name in ('events.jsonl','splits.json','coverage.json'):add('streaming_sparse/v02/data/'+stage+'/'+name,HOME/'data'/stage/name)
    add('streaming_sparse/v02/data/shared/events.jsonl',HOME/'data/shared/events.jsonl')
    for name in ('parallel_commands.jsonl','commands.jsonl','workflow.log'):
        add('streaming_sparse/v02/logs/'+name,HOME/'logs'/name)
    old=HOME.parent/'artifacts/self_forcing_sparse_review_20261015.zip'
    add('streaming_sparse/artifacts/'+old.name,old)
    raw,limits,expected={},{},{}
    for item in states:
        path=Path(item['path']);base=PREFIX+str(path.relative_to(ROOT))+'/'
        name=base+'per_tile.jsonl';raw[name]=path/'per_tile.jsonl';limits[name]=item['raw_bytes'];expected[name]=item['audit']['sha256']
        for name,digest in item['audit']['partition_sha256'].items():
            key=base+'partitions/'+name;raw[key]=path/'partitions'/name;expected[key]=digest
    # Pilot is already complete, and remains useful for reproducing the gate.
    for run in (HOME/'runs/pilot').glob('*/seed*'):
        for path in run.glob('eval_*/per_tile.jsonl'):raw[PREFIX+str(path.relative_to(ROOT))]=path
        for path in run.glob('eval_*/partitions/*.pt'):raw[PREFIX+str(path.relative_to(ROOT))]=path
    for stage in ('pilot','full'):
        for path in (HOME/'runs'/stage).glob('*/seed*/assignments/*.pt'):raw[PREFIX+str(path.relative_to(ROOT))]=path
    texts={'README_REVIEW.md':'# v0.2 阶段评审包\n\n本包为暂停时的真实阶段结果，不表示全部实验完成。\n\n阅读 Self-Forcing/streaming_sparse/v02/reports/CURRENT_REVIEW.md、TEST_ANALYSIS.md、MIGRATION_GUIDE.md。原始证据在独立 evidence.zip，迁移资产在独立 migration.tar.gz。\n'}
    progress('writing_review_zip',files=len(core))
    review=archive(DEST/'self_forcing_sparse_v02_paused_review_20261016.zip',core,texts)
    progress('writing_evidence_zip',files=len(raw))
    evidence=archive(DEST/'self_forcing_sparse_v02_paused_evidence_20261016.zip',raw,expected_hashes=expected,limits=limits)
    migration=dict(core)
    for logical in ('data/shared','data/pilot','data/full','runs/pilot','runs/full','runs/p0'):
        add_tree(migration,'streaming_sparse/v02/'+logical,HOME/logical)
    for logical in ('data','runs'):
        add_tree(migration,'streaming_sparse/'+logical,HOME.parent/logical)
    add_tree(migration,'.git',ROOT/'.git')
    for name in list(migration):
        if name.endswith('.pid') or name.endswith('.lock') or name.startswith(PREFIX+'.git/hooks/') or name.startswith(PREFIX+'.git/logs/'):
            del migration[name]
    migration.pop(PREFIX+'.git/config',None)
    gitconfig='[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n[remote "origin"]\n\turl = https://github.com/guandeh17/Self-Forcing.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
    extra={PREFIX+'.git/config':gitconfig,PREFIX+'MIGRATION_GUIDE.md':guide}
    result=migration_archive(DEST/'self_forcing_sparse_v02_migration_20261016.tar.gz',migration,extra,limits)
    receipt=dict(packages_verified=True,experiment_complete=False,paused=True,automatic_resume_disabled=True,
        created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),training_completed=15,
        complete_evaluations=complete,total_evaluations=45,review=review,evidence=evidence,migration=result)
    atomic_json(DEST/'PAUSED_REVIEW_COMPLETE.json',receipt)
    atomic_json(HOME/'reports/PAUSED_REVIEW_COMPLETE.json',receipt)
    progress('paused_packages_ready',**receipt)


if __name__=='__main__':main()
