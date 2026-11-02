"""Serial fresh-process official comparison under the existing idle reservoir."""
import gc,json,os,subprocess,sys,time
from pathlib import Path
import torch
from .learned_cache.io import atomic,digest,disk_guard,RunLock
from .stream_eval_common import ROOT,REPO
from .gap_gpu_processes import inventory
METHODS=['dense','ours','lightforcing','forcingkv','dasa']
CASES=['sailboat','skateboard','dog']
def main():
    ROOT.mkdir(exist_ok=True);lock=RunLock(ROOT)
    try:
        gc.collect();torch.cuda.empty_cache()
        manifest=dict(methods=METHODS,cases=CASES,height=512,width=768,frames=480,fps=16,chunks=41,latent_frames_per_chunk=3,history_chunk_cap=5,checkpoint_sha256=digest(REPO/'checkpoints/self_forcing_dmd.pt'),own_model_sha256=digest(REPO/'artifacts/four_gaps_20261031/shared/epoch1.pt'),quality_window=[21,101],repeats=2,full_warmup=True,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],reservation_pid=os.getpid(),exceptions=['Forcing-KV native smaller per-head context retained and reported, not padded to 5 chunks.','DASA 15 main-cache frames plus 3 hidden clean KV frames within common self-KV cap; auxiliary feature reuse reported separately.','Our summary metadata allows 16MiB above original raw KV allocation; raw history338 pages vs dense360 pages.','LightForcing same-backbone weight transfer excludes its paper-specific fine-tuning; native relative position policy remains.','Dense uses common PyTorch SDPA; official LightForcing sparse Triton path.','Official bounded-buffer compatibility patches retained and archived; no user rebase modification.'],scope='Small three-scene same-weight controlled comparison; not a published-paper quality reproduction or comprehensive VBench evaluation.')
        atomic(ROOT/'manifest.json',manifest)
        for repo in ['LightForcing','Forcing-KV','DASA']:
            (ROOT/(repo+'.compatibility.patch')).write_text(subprocess.check_output(['git','-C',str(REPO/'external_baselines'/repo),'diff'],text=True))
        for ci,case in enumerate(CASES):
            source=REPO/'artifacts/ar_official_baselines_20261029'/case/'shared_inputs.pt';assert source.is_file()
            folder=ROOT/case;folder.mkdir(exist_ok=True)
            (folder/'prompt.txt').write_text((source.parent/'prompt.txt').read_text())
            atomic(folder/'input_identity.json',dict(path=str(source),sha256=digest(source)))
            rest=METHODS[1:];rest=rest[ci%len(rest):]+rest[:ci%len(rest)]
            for method in ['dense']+rest:
                out=folder/method;out.mkdir(exist_ok=True)
                if (out/'result.json').exists():continue
                disk_guard(ROOT,50,1)
                peers=[r for ps in inventory(os.environ['CUDA_VISIBLE_DEVICES']).values() for r in ps if r['pid']!=os.getpid()];assert not peers,peers
                module='stream_eval_ours' if method=='ours' else 'stream_eval_official'
                cmd=[sys.executable,'-u','-m','local_experiments.'+module,'--inputs',str(source),'--output',str(out)]
                if method!='ours':cmd+=['--method',method]
                env=os.environ.copy();env.update(STREAM_RESERVATION_PID=str(os.getpid()),GAP_TAIL_MODE='1',PYTHONUNBUFFERED='1')
                atomic(ROOT/'status.json',dict(status='running',case=case,method=method,updated=time.time()))
                with (out/'worker.log').open('a') as log:
                    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                    atomic(out/'process.json',dict(pid=child.pid,parent=os.getpid(),started=time.time(),command=cmd))
                    code=child.wait()
                if code:raise RuntimeError(f'{case}/{method} exited {code}; see {out}/worker.log')
                assert (out/'result.json').exists()
        atomic(ROOT/'status.json',dict(status='complete'))
    except BaseException as e:
        atomic(ROOT/'status.json',dict(status='failed',error=repr(e)));raise
    finally:lock.close()
if __name__=='__main__':main()
