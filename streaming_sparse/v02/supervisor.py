"""Run under tmux; preserve exit/signal evidence and bound unexpected restarts."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .evaluation_journal import atomic_json


HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
LOGS = HOME / 'logs'


def event(kind, **details):
    value = dict(time_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                 supervisor_pid=os.getpid(), event=kind, **details)
    with (LOGS / 'supervisor.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(HOME / 'supervisor.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-restarts', type=int, default=2)
    parser.add_argument('--gpus',nargs='+',type=int)
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    with (LOGS / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (HOME / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
        child = None
        stopping = False
        stop_signal = None

        def stop(signum, frame):
            nonlocal stopping, stop_signal
            stopping = True
            stop_signal = signum
            event('supervisor_signal', signal=signum)
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, stop)
        library=ctypes.CDLL(str(HOME / 'libsignal_audit.so'))
        library.install_signal_audit.argtypes=[ctypes.c_char_p]
        library.install_signal_audit.restype=ctypes.c_int
        if library.install_signal_audit(os.fsencode(LOGS / 'signal_senders.log')) != 0:
            raise RuntimeError('Could not install sender-identity signal audit')
        for attempt in range(args.max_restarts + 1):
            if stopping or (HOME / 'supervisor.stop').exists():
                event('stopped_by_operator')
                return
            stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
            trace = LOGS / f'signals_{stamp}_attempt{attempt}.log'
            command = ['strace', '-f', '--seccomp-bpf', '-ttt', '-e', 'trace=none',
                       '-e', 'signal=SIGTERM,SIGKILL,SIGHUP,SIGINT,SIGSEGV,SIGABRT,SIGXCPU,SIGXFSZ',
                       '-o', str(trace), sys.executable, '-u', '-m',
                       'streaming_sparse.v02.workflow', '--gpu', str(args.gpu), '--through', 'full']
            if args.gpus:command+=['--gpus',*map(str,args.gpus)]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1')
            with (LOGS / 'workflow.log').open('a') as output:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True, close_fds=True)
                event('started', attempt=attempt, tracer_pid=child.pid,
                      signal_log=str(trace), command=command)
                while child.poll() is None:
                    if (HOME / 'supervisor.stop').exists() and not stopping:
                        stop(signal.SIGTERM, None)
                    time.sleep(2)
                code = child.returncode
            try:
                state = json.loads((HOME / 'status.json').read_text())
            except (OSError, ValueError):
                state = {}
            event('exited', attempt=attempt, returncode=code, status=state,
                  signal_log=str(trace), stopping=stopping)
            if stopping:
                explicit = (HOME / 'supervisor.stop').exists()
                state.update(state='paused' if explicit else 'failed',
                             detail='Supervisor stop marker requested' if explicit else
                             f'Supervisor received external signal {stop_signal}; see signal trace and supervisor log',
                             updated_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
                atomic_json(HOME / 'status.json', state)
                (HOME / 'STATUS.md').write_text('# v0.2 状态\n\n' + state['updated_utc'] + ' UTC\n\n' +
                                              state['state'] + '\n\n' + state['detail'] + '\n')
                return
            if code == 0 or state.get('state') in ('paused', 'needs_analysis', 'complete'):
                return
            # A Python/numerical error needs investigation. Only externally
            # interrupted runs with stale running state receive bounded retries.
            if state.get('state') != 'running' or not (code < 0 or code >= 128):
                return
            if attempt == args.max_restarts:
                event('restart_limit', returncode=code)
                return
            event('retry_after_signal', next_attempt=attempt + 1)
            time.sleep(5)


if __name__ == '__main__':
    main()
