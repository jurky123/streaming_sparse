"""Run an experiment with same-process VRAM reservation; no external locks."""
import json,os,runpy,sys,gc,time
from pathlib import Path
import torch
from .gap_gpu_processes import inventory

def main():
    targets=sys.argv[1:]
    assert targets and all(target in ['gap_group_fused_features_perf','gap_staged_single_perf','gap_tail_fused_merge_q32_check','gap_staged_history_quality','gap_staged_history_perf','gap_staged_history_profile_entry'] for target in targets)
    peers=[p for ps in inventory(os.environ['CUDA_VISIBLE_DEVICES']).values() for p in ps if p['pid']!=os.getpid()]
    if peers and os.environ.get('GAP_WAIT_FOR_PEERS')=='1':
        # A tiny own CUDA context lets cooperative workers finish and yield.
        # No peer signals, external locks, scheduler changes, or busy GPU work.
        presence=torch.empty(1,dtype=torch.uint8,device='cuda')
        previous=None
        while peers:
            if peers!=previous:
                print('WAITING_FOR_PEERS '+json.dumps(peers),flush=True)
                previous=peers
            time.sleep(5)
            peers=[p for ps in inventory(os.environ['CUDA_VISIBLE_DEVICES']).values() for p in ps if p['pid']!=os.getpid()]
    if peers:raise RuntimeError(f'GPU occupied before reservation: {peers}')
    free,total=torch.cuda.mem_get_info()
    working=int(float(os.environ.get('GAP_WORKING_GIB','48'))*2**30)
    size=max(0,free-working)
    reservation=torch.empty(size,dtype=torch.uint8,device='cuda')
    metadata=dict(pid=os.getpid(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],reserved_bytes=size,working_headroom_bytes=working,total_bytes=total,scope='Untouched same-process VRAM reservation; included in process memory, not retained KV; does not guarantee exclusive GPU access.')
    print('VRAM_RESERVATION '+json.dumps(metadata),flush=True)
    if os.environ.get('GAP_HOT_PAIR_ROOT'):
        root=Path(os.environ['GAP_HOT_PAIR_ROOT']);root.mkdir(parents=True,exist_ok=True)
        (root/'vram_reservation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for target in targets:
        print('EXPERIMENT_START '+target,flush=True)
        sys.argv=[target]
        runpy.run_module('local_experiments.'+target,run_name='__main__')
        gc.collect();torch.cuda.empty_cache()
        print('EXPERIMENT_COMPLETE '+target,flush=True)
    del reservation
if __name__=='__main__':main()
