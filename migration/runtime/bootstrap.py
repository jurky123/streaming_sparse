"""Enable stack diagnostics before importing the GPU runtime."""
import faulthandler
import os
import functools
import importlib._bootstrap_external as import_external
import runpy
import signal
import sys
import time
faulthandler.enable()
faulthandler.register(signal.SIGUSR1,all_threads=True)
print(f'bootstrap pid={os.getpid()} rank={os.environ.get("RANK")} visible={os.environ.get("CUDA_VISIBLE_DEVICES")}',flush=True)
# Dependencies are fixed by requirements.lock and never edited by the worker.
# Avoid repeatedly stat-ing the same dependency directories on the shared FS.
# Source, replay data, checkpoints, and output paths are deliberately excluded.
dependency_root=os.path.abspath(os.path.join(os.path.dirname(sys.executable),'..'))+os.sep
original_path_stat=import_external._path_stat
cached_dependency_stat=functools.lru_cache(maxsize=32768)(original_path_stat)
def dependency_stat(path):
    return cached_dependency_stat(path)if str(path).startswith(dependency_root)else original_path_stat(path)
import_external._path_stat=dependency_stat
# PyTorch's registration-time inspect.getmodule repeatedly resolves identical
# paths. Cache only during import on LeoFS; restore normal filesystem behavior
# before executing experiment code. No package/version or model math changes.
original_realpath=os.path.realpath
cached_dependency_realpath=functools.lru_cache(maxsize=32768)(original_realpath)
def dependency_realpath(path,*args,**kwargs):
    return (cached_dependency_realpath(path,*args,**kwargs)
            if os.fspath(path).startswith(dependency_root) else original_realpath(path,*args,**kwargs))
os.path.realpath=functools.lru_cache(maxsize=16384)(original_realpath)
started=time.perf_counter()
try:import torch
finally:os.path.realpath=dependency_realpath
print(f'torch_import_seconds={time.perf_counter()-started:.3f}',flush=True)
module=sys.argv[1];sys.argv=[module,*sys.argv[2:]]
runpy.run_module(module,run_name='__main__')
