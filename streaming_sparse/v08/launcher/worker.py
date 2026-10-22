"""Require the execution equivalence audit before running an optimized job."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime,timezone
from ..common import HERE,atomic_json,source_identity,sha
from ..execution import accelerate


def launcher_identity():
    digest=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def require_execution_audit():
    path=HERE/'reports/execution/benchmark.json'
    audit=json.loads(path.read_text())
    if not audit.get('passed') or not audit.get('checks',{}).get('full_input'):
        raise RuntimeError('Execution optimization requires the full-input equivalence audit')
    if audit['source_sha256']!=source_identity() or audit['execution_sha256']!=accelerate.identity():
        raise RuntimeError('Execution audit belongs to different source code')
    return audit


def checkpoint_instrumentation(train,execution):
    """Save optimizer-boundary recovery state, without changing update math.

    Original training only saves at validation boundaries. The instrumented
    optimizer delegates to exactly the same AdamW.step, verifies frozen state,
    and saves the existing snapshot closure every ten completed updates.
    """
    import torch
    original_optimizer=torch.optim.AdamW
    original_save=train.save
    def save(path,value):
        if isinstance(value,dict)and 'source_sha256'in value:
            value={**value,'execution':execution}
        return original_save(path,value)
    train.save=save
    class RecoveryAdamW(original_optimizer):
        def step(self,*args,**kwargs):
            if accelerate.identity()!=execution['kernel_sha256'] or launcher_identity()!=execution['launcher_sha256']:
                raise RuntimeError('Execution source changed while a worker was running')
            result=super().step(*args,**kwargs)
            frame=inspect.currentframe().f_back
            try:
                while frame is not None and frame.f_code is not train.run.__code__:
                    frame=frame.f_back
                if frame is not None:
                    local=frame.f_locals;step=local.get('step')
                    if isinstance(step,int)and step%10==0 and 'snapshot'in local:
                        local['verify_frozen']()
                        directory=local['directory']
                        save(directory/'recovery.pt',local['snapshot'](step))
                        atomic_json(directory/'recovery.json',dict(updates=step,source_sha256=source_identity(),
                            execution=execution,saved_at=datetime.now(timezone.utc).isoformat()))
            finally:
                del frame
            return result
    torch.optim.AdamW=RecoveryAdamW


def main():
    allowed={'streaming_sparse.v08.'+m for m in ('train','evaluate','rollout','warmstart')}
    if len(sys.argv)<2 or sys.argv[1]not in allowed:
        raise SystemExit('Expected a v08 train/evaluate/rollout/warmstart module')
    module=sys.argv[1];arguments=sys.argv[2:];audit=require_execution_audit()
    execution=dict(kernel_sha256=accelerate.identity(),launcher_sha256=launcher_identity(),
        benchmark_sha256=sha(HERE/'reports/execution/benchmark.json'),score_query_batch=accelerate.SCORE_BATCH,
        gate_vjp_query_batch=accelerate.GATE_BATCH,auxiliary_query_batch=16,
        ranking_guard_absolute=accelerate.GUARD_ABSOLUTE,ranking_guard_relative=accelerate.GUARD_RELATIVE,
        native_attention_layout='unchanged full-query twelve-head BF16',recovery_every_updates=10)
    accelerate.install()
    if module=='streaming_sparse.v08.train':
        from .. import train
        checkpoint_instrumentation(train,execution)
        # Call the imported module once; runpy would create another globals dict.
        entry=train.main
    else:
        entry=None
    atomic_json(HERE/f'reports/execution/workers/{os.getpid()}.json',dict(source_sha256=source_identity(),
        execution=execution,module=module,arguments=arguments,pid=os.getpid(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),started_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(event='audited_execution_enabled',**execution)),flush=True)
    sys.argv=[module,*arguments]
    if entry:entry()
    else:runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()
