"""Own persistent experiment session with VRAM reservation, no external locks."""
import gc,json,os,re,runpy,signal,sys,time,traceback
from pathlib import Path
import torch
from .gap_gpu_processes import inventory
ROOT=Path(os.environ.get('GAP_SESSION_ROOT','artifacts/gap_gpu4_session_20261102'))
def atomic(name,obj):
    path=ROOT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def main():
    ROOT.mkdir(exist_ok=True)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','4')
    assert gpu.isdigit(), 'One physical GPU required'
    peers=[p for ps in inventory(gpu).values() for p in ps if p['pid']!=os.getpid()]
    if peers and os.environ.get('GAP_WAIT_FOR_PEERS')=='1':
        presence=torch.empty(1,dtype=torch.uint8,device='cuda')
        while peers:
            atomic('status.json',dict(status='waiting_for_peers',pid=os.getpid(),gpu=int(gpu),peers=peers,updated=time.time()))
            print('WAITING_FOR_PEERS '+json.dumps(peers),flush=True)
            time.sleep(10)
            peers=[p for ps in inventory(gpu).values() for p in ps if p['pid']!=os.getpid()]
    if peers:raise RuntimeError(f'GPU{gpu} occupied: {peers}')
    free,total=torch.cuda.mem_get_info();headroom=40*2**30
    reservation=torch.empty(max(0,free-headroom),dtype=torch.uint8,device='cuda')
    base=dict(pid=os.getpid(),gpu=int(gpu),reserved_bytes=reservation.numel(),working_headroom_bytes=headroom,total_bytes=total,policy='Own reservation retained until user release. No external locks or peer termination.')
    atomic('reservation.json',base);print(json.dumps(base),flush=True)
    if not (ROOT/'tasks.json').exists():atomic('tasks.json',[])
    stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    while not stop[0]:
        atomic('status.json',base|dict(status='reserved_waiting',updated=time.time()))
        for task in json.loads((ROOT/'tasks.json').read_text()):
            uid=task['id'];module=task['module']
            assert re.fullmatch(r'[a-zA-Z0-9_-]+',uid) and re.fullmatch(r'gap_[a-z0-9_]+',module)
            if (ROOT/(uid+'.result.json')).exists():continue
            if stop[0]:break
            atomic('status.json',base|dict(status='running',task=task,updated=time.time()))
            result=dict(task=task,started=time.time());saved=sys.argv
            try:
                peers=[p for ps in inventory(gpu).values() for p in ps if p['pid']!=os.getpid()]
                if peers:raise RuntimeError(f'GPU{gpu} shared before task: {peers}')
                sys.argv=[module]+task.get('args',[])
                runpy.run_module('local_experiments.'+module,run_name='__main__')
                result['status']='process_finished'
            except BaseException as exc:result.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
            finally:
                sys.argv=saved;gc.collect();torch.cuda.empty_cache()
                result['finished']=time.time();atomic(uid+'.result.json',result)
        time.sleep(2)
    atomic('status.json',base|dict(status='stopped',updated=time.time()))
if __name__=='__main__':main()
