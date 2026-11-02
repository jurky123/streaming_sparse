"""Held-out, separately rebuilt rollouts; initial versus both trained checkpoints."""
import gc,json,os,torch
from pathlib import Path
from .broad_common import Engine,cases,ROOT as DATA
from .rollout_suite import generate_chunk
from .all_sparse_runtime import AllSparseRuntime
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/sparse_expansion_20261031/all_sparse_train')
def main():
    lock=RunLock(ROOT/'validation')
    try:
        torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
        identity=dict(checkpoints={name:digest(ROOT/(name+'.pt')) for name in ['initial','epoch0','epoch1']},code={name:digest(Path('local_experiments')/name) for name in ['all_sparse_validate.py','all_sparse_runtime.py','all_sparse_model.py','all_sparse_attention.py']},cases=cases('val'))
        ip=ROOT/'validation/identity.json'
        if ip.exists():assert json.loads(ip.read_text())==identity
        else:atomic(ip,identity)
        engine=Engine();engine.wrapper.requires_grad_(False);base_forward=engine.forward
        for case in cases('val'):
            for label in ['initial','epoch0','epoch1']:
                dest=ROOT/'validation'/f'{case["id"]}_{label}.json'
                if dest.exists():continue
                disk_guard(ROOT,50,1);atomic(ROOT/'validation/status.json',dict(status='running',case=case['id'],checkpoint=label,pid=os.getpid()))
                rt=AllSparseRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt');rt.model.load_state_dict(torch.load(ROOT/(label+'.pt'),map_location='cpu',weights_only=False)['model']);rows=[]
                def forward(x,emb,chunk,step):
                    rt.sparse=True;out=base_forward(x,emb,chunk,step)
                    if chunk in [2,5] and step in [0,3]:
                        rt.sparse=False
                        try:teacher=base_forward(x,emb,chunk,step)
                        finally:rt.sparse=True
                        err=float((out.float()-teacher.float()).square().mean()/teacher.float().square().mean().clamp_min(1e-8))
                        assert torch.isfinite(out).all();rows.append(dict(chunk=chunk,step=step,nmse=err))
                    return out
                engine.forward=forward
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(6):generate_chunk(engine,emb,case['seed'],c)
                    assert len(rows)==4;atomic(dest,dict(rows=rows,mean_nmse=sum(r['nmse'] for r in rows)/4,scope='each checkpoint has its own rebuilt history and rollout; dense same-input teacher at four events; not matched-input across checkpoints, not RGB video acceptance'))
                    print(json.dumps(dict(case=case['id'],checkpoint=label,mean_nmse=sum(r['nmse'] for r in rows)/4)),flush=True)
                finally:engine.forward=base_forward;rt.close();del rt;gc.collect();torch.cuda.empty_cache()
        atomic(ROOT/'validation/status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'validation/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
