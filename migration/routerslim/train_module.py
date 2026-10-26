"""Audited R1 opt-in entry point; original distributed loop remains immutable."""
import inspect
import json
from pathlib import Path
import sys
import textwrap
from migration.runtime import multigpu
from migration.routerslim import r1
from streaming_sparse.v08.common import sha,source_identity


def main():
    flag='--routerslim-audit';index=sys.argv.index(flag);path=Path(sys.argv[index+1]);del sys.argv[index:index+2]
    report=json.loads(path.read_text())
    assert report['passed']and report['source_sha256']==source_identity()
    assert report['r1_sha256']==r1.identity()
    assert report['entry_sha256']==sha(__file__)
    for name,digest in report['evidence'].items():assert sha(name)==digest,(name,'R1 evidence changed')
    # Install the predecessor first; its first install assigns _head. R1 must
    # remain the final head implementation when multigpu.main installs again.
    multigpu.accelerate.install();multigpu.router_fast.install()
    r1.install(**report['features'])
    entry_sha=sha(__file__);audit_sha=sha(path)
    old_start=multigpu.update_started
    def start():
        assert r1.identity()==report['r1_sha256']and sha(__file__)==entry_sha,'R1 source changed during training'
        frame=inspect.currentframe().f_back
        try:r1.configure_update(frame.f_locals['network'],frame.f_locals['step'])
        finally:del frame
        r1.stats.update(boundary_groups=0,score_queries=0)
        old_start()
    multigpu.update_started=start
    original_io=multigpu.install_checkpoint_io
    def install_io(train,execution):
        multigpu.run_info['routerslim']=dict(r1_sha256=r1.identity(),entry_sha256=entry_sha,audit_sha256=audit_sha,
            features=report['features'],
            hard_boundary_fallback='original equations and original 16-query grouping')
        return original_io(train,execution)
    multigpu.install_checkpoint_io=install_io
    # Keep the immediate caller of record_update as train.run for the existing
    # exact recovery snapshot mechanism. Do not interpose a Python wrapper.
    source=textwrap.dedent(inspect.getsource(multigpu.record_update))
    source=source.replace('    elapsed=time.perf_counter()-_step_started',"""    item['att_aux_computed']=item['updates']<501
    if not item['att_aux_computed']:item['losses']['att_aux']=None
    item['routerslim']=run_info['routerslim']
    item['routerslim_boundary_counts_rank0']=dict(_routerslim.stats)
    elapsed=time.perf_counter()-_step_started""",1)
    multigpu.__dict__['_routerslim']=r1
    exec(compile(source,'<R1 metrics metadata>','exec'),multigpu.__dict__)
    multigpu.main()


if __name__=='__main__':main()
