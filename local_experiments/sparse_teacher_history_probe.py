"""Diagnose same-student-history versus independent-dense-history targets."""
import gc,json,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .rollout_runtime import Runtime
from .all_sparse_runtime import AllSparseRuntime
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,RunLock,disk_guard,digest
ROOT=Path('artifacts/sparse_quality_line_20261031/history_probe');PREV=Path('artifacts/sparse_expansion_20261031')
@torch.no_grad()
def main():
    lock=RunLock(ROOT)
    try:
        torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
        paths={'initial':PREV/'all_sparse_train/initial.pt','horizon2':PREV/'horizon2_train/epoch0.pt'}
        identity=dict(code=digest(Path(__file__)),models={k:digest(v) for k,v in paths.items()},cases=['city_walk','flower_wind'],seed=0,chunks=[2,5],steps=[0,2])
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        student=Engine();teacher=Engine();student.wrapper.requires_grad_(False);teacher.wrapper.requires_grad_(False);dense=Runtime(teacher,'dense');base=student.forward
        for case in identity['cases']:
            emb=torch.load(DATA/'embeddings'/f'{case}.pt',weights_only=True).cuda();teacher_reference=None
            for label,path in paths.items():
                disk_guard(ROOT,50,1);atomic(ROOT/'status.json',dict(status='running',case=case,checkpoint=label))
                rt=AllSparseRuntime(student,'artifacts/full_route_grad_20261031/epoch1.pt');rt.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);teacher.reset();rows=[];dense_chunks=[]
                def forward(x,emb,chunk,step):
                    out=base(x,emb,chunk,step)
                    if chunk in [2,5] and step in [0,2]:
                        rt.sparse=False
                        try:shared=base(x,emb,chunk,step)
                        finally:rt.sparse=True
                        independent=teacher.forward(x,emb,chunk,step)
                        norm=independent.float().square().mean().clamp_min(1e-8)
                        error=lambda a,b:float((a.float()-b.float()).square().mean()/norm)
                        rows.append(dict(chunk=chunk,step=step,student_to_shared=error(out,shared),student_to_independent=error(out,independent),teacher_target_gap=error(shared,independent)))
                    return out
                student.forward=forward
                try:
                    for c in range(6):
                        clean=generate_chunk(teacher,emb,0,c);dense_chunks.append(clean.cpu())
                        generate_chunk(student,emb,0,c)
                        # Probes overwrite only the current dense KV slots. Restore
                        # the teacher's own clean chunk before advancing history.
                        teacher.forward(clean,emb,c,4)
                    if teacher_reference is None:teacher_reference=dense_chunks
                    else:assert all(torch.equal(a,b) for a,b in zip(teacher_reference,dense_chunks)),'Teacher trajectory contaminated by student probes'
                    assert len(rows)==4;result=dict(rows=rows,teacher_trajectory_exact=True,scope='train scenes only; dense teacher uses original chronological cache budget, student uses grouped338 pages. chunk5 target gap includes retention differences; no claim to isolate history drift alone.')
                    atomic(ROOT/f'{case}_{label}.json',result);print(json.dumps(dict(case=case,checkpoint=label,rows=rows)),flush=True)
                finally:student.forward=base;rt.close();del rt;gc.collect();torch.cuda.empty_cache()
        dense.close();atomic(ROOT/'status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
