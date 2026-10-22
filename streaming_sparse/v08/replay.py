"""Prompt-sealed replay inputs, payload checks and balanced paired sampling."""
import hashlib
import json
import random
from pathlib import Path
import torch
from .common import HERE,ROOT,protocol,source_identity,atomic_json,save,sha


def prompt_id(text):return hashlib.sha256(text.encode()).hexdigest()[:20]


def prepare_registry():
    p=protocol();path=HERE/'configs/prompt_registry.json'
    if path.exists():
        registry=json.loads(path.read_text())
        if registry['protocol_sha256']!=sha(HERE/'configs/protocol.json'):
            raise ValueError('Existing prompt registry belongs to another protocol')
        return registry
    legacy=(ROOT/'streaming_sparse/v04/configs/capture_prompts.txt').read_text().splitlines()
    by_id={prompt_id(t):t for t in legacy}
    blocked=set(p['legacy_train_prompts']+p['legacy_val_prompts']+p['excluded_old_test'])
    # Every historical capture manifest in this project contributes exposed IDs;
    # never rename an already evaluated prompt as independent Test.
    for manifest in (ROOT/'streaming_sparse').glob('v*/data/*/events.jsonl'):
        with manifest.open() as f:
            for line in f:
                if line.strip():blocked.add(json.loads(line)['prompt_id'])
    candidates={prompt_id(t):t for t in (ROOT/p['new_prompt_source']).read_text().splitlines()
                if t.strip() and prompt_id(t) not in blocked}
    ids=sorted(candidates);random.Random(p['registry_seed']).shuffle(ids)
    splits={'train':list(p['legacy_train_prompts']),'val':list(p['legacy_val_prompts']),'test':[]}
    position=0
    for split in ('train','val','test'):
        needed=p['split_counts'][split]-len(splits[split]);chosen=ids[position:position+needed]
        if len(chosen)!=needed:raise ValueError('Not enough unexposed prompt candidates')
        position+=needed;splits[split]+=chosen
    for a in splits:
        for b in splits:
            if a!=b and set(splits[a])&set(splits[b]):raise ValueError('Prompt split leakage')
    registry=dict(protocol_sha256=sha(HERE/'configs/protocol.json'),splits=splits,
        texts={i:(by_id|candidates)[i] for i in sum(splits.values(),[])},
        excluded_exposed_prompts=sorted(blocked),source=protocol()['new_prompt_source'],
        test_status='sealed; no architecture/checkpoint selection on new Test')
    atomic_json(path,registry)
    return registry


class ReplayDataset:
    def __init__(self,split,root=None,lock=None,require_complete=False):
        p=protocol();self.split=split;self.root=Path(root or HERE/'data/replay')
        registry=prepare_registry();self.registry=registry
        if split not in ('train','val','test'):raise ValueError(split)
        if split=='test':
            if lock is None:raise ValueError('New Test is sealed; a final checkpoint lock is required')
            self.lock=json.loads(Path(lock).read_text())
            if self.lock['source_sha256']!=source_identity() or self.lock['registry_sha256']!=sha(HERE/'configs/prompt_registry.json'):
                raise ValueError('Test lock source/registry mismatch')
        manifest=self.root/'events.jsonl'
        rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()] if manifest.exists() else []
        allowed=set(registry['splits'][split]);self.rows=[r for r in rows if r['prompt_id'] in allowed]
        self.rows.sort(key=lambda r:(r['prompt_id'],r['generation_seed'],r['history_tokens'],r['step']))
        keys=[(r['prompt_id'],r['generation_seed'],r['history_tokens'],r['step']) for r in self.rows]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate replay observation')
        for r in self.rows:
            if r['split']!=split or r['teacher_sha256']!=p['teacher_sha256'] or r['parameterization']!='x0':
                raise ValueError('Replay identity/target mismatch')
            if r['timestep']!=p['native_timesteps'][r['step']]:raise ValueError('Wrong native timestep')
        if require_complete:
            expected={(i,g,h,s) for i in allowed for g in p['generation_seeds'] for h in p['histories'] for s in p['steps']}
            if set(keys)!=expected:raise ValueError(f'Incomplete {split} replay: {len(keys)}/{len(expected)} inputs')
        self._trajectory=None;self._payload=None

    def __len__(self):return len(self.rows)

    def load(self,index):
        r=self.rows[index];path=self.root/r['payload']
        if self._trajectory!=r['trajectory_id']:
            if sha(path)!=r['payload_sha256']:raise ValueError('Replay payload SHA mismatch')
            self._payload=torch.load(path,map_location='cpu',weights_only=True)
            self._trajectory=r['trajectory_id']
        x=self._payload;chunk=r['chunk'];step=r['step'];p=protocol()
        if x['teacher_sha256']!=r['teacher_sha256'] or x['scheduler_identity']!=r['scheduler_identity']:
            raise ValueError('Replay payload trajectory identity mismatch')
        sample=dict(r,history=x['latents'][:,:chunk*p['frames_per_chunk']],
            noisy=x['noisy'][chunk,step],target=x['teacher_x0'][chunk,step],
            prompt_embeds=x['prompt_embeds'],context_noise=x['context_noise'])
        expected=(1,p['frames_per_chunk'],*p['latent_shape'])
        if tuple(sample['noisy'].shape)!=expected or sample['target'].shape!=sample['noisy'].shape:
            raise ValueError('Replay is not a full original-resolution current chunk')
        if sample['history'].shape[1]*p['frame_tokens']!=r['history_tokens']:
            raise ValueError('Wrong historical latent coverage')
        return sample

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.rows,sort_keys=True).encode()).hexdigest()


class BalancedReplay:
    """Without-replacement balanced factorial passes; same seed => same inputs."""
    def __init__(self,ds,seed):
        if not len(ds):raise ValueError('No replay inputs; collect noisy states first')
        self.ds=ds;self.rng=random.Random(seed);self.pool=[];self.visits=[0]*len(ds)
    def sample(self,count):
        result=[]
        for _ in range(count):
            if not self.pool:self.pool=list(range(len(self.ds)));self.rng.shuffle(self.pool)
            i=self.pool.pop();self.visits[i]+=1;result.append(i)
        return result
    def state_dict(self):return dict(rng=self.rng.getstate(),pool=self.pool,visits=self.visits,fingerprint=self.ds.fingerprint())
    def load_state_dict(self,s):
        if s['fingerprint']!=self.ds.fingerprint():raise ValueError('Sampler data changed')
        self.rng.setstate(s['rng']);self.pool=s['pool'];self.visits=s['visits']
    def coverage(self):
        return dict(inputs=len(self.visits),seen=sum(v>0 for v in self.visits),min_visits=min(self.visits),
                    max_visits=max(self.visits),total_draws=sum(self.visits),visits=self.visits)


def inventory():
    p=protocol();examples=list((ROOT/'streaming_sparse/v04/data').glob('*/*.latents.pt'))
    report=dict(source_sha256=source_identity(),final_latent_files=len(examples),
        final_latent_example=str(examples[0]) if examples else None,
        legacy_capture_saves='final latents only; intermediate noisy states are not persisted by legacy capture_runner',
        reusable_qkv='warm-start and attention diagnostics only; cannot update backbone from fixed QKV',
        replay_manifest_exists=(HERE/'data/replay/events.jsonl').exists(),
        action='Replay original sampler and record actual intermediate noisy states; never infer them from final latents')
    if examples:
        x=torch.load(examples[0],map_location='cpu',weights_only=True)
        report['final_latent_shape']=list(x.shape)
    atomic_json(HERE/'reports/replay_inventory.json',report)
    return report
