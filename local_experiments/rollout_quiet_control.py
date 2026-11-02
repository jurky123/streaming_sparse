"""Opportunistic paired control on an initially empty GPU, with contention audit."""
import gc,json,os,statistics,subprocess,threading,time
from pathlib import Path
import torch
from . import rollout_dense_control as control
from .rollout_analyze import write,sha
from .rollout_runtime import ROOT

def main():
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];root=ROOT/'quiet_control';root.mkdir(exist_ok=True)
    def processes():
        out=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
        return [int(x) for x in out.splitlines() if x.strip().isdigit()]
    assert not [p for p in processes() if p!=os.getpid()], 'GPU is no longer empty; do not mix busy timing into quiet control'
    stop=threading.Event();errors=[];foreign=[]
    def monitor():
        while not stop.is_set():
            try:
                pids=processes();others=[p for p in pids if p!=os.getpid()];foreign.extend(others)
                row=dict(unix_time=time.time(),gpu=gpu,worker_pid=os.getpid(),pids=pids,foreign=others)
            except Exception as exc:errors.append(repr(exc));row=dict(unix_time=time.time(),error=repr(exc))
            with (root/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    control.ROOT=root
    try:control.main()
    finally:stop.set();thread.join()
    rows={}
    for method in ['dense','dense_host','mass_var']:
        results=[json.loads(p.read_text()) for p in sorted((root/'dense_control').glob(f'{method}.rep*.json'))]
        times=[r['total_generation_vae_seconds'] for r in results]
        rows[method]=dict(n=len(times),median=statistics.median(times),min=min(times),max=max(times),generation=statistics.median(r['generation_seconds'] for r in results))
    a=torch.load(root/'dense_control/mass_var.latents.pt',weights_only=True)
    b=torch.load(ROOT/'basketball_court/mass_var.latents.pt',weights_only=True)
    exact=torch.equal(a,b)
    result=dict(rows=rows,foreign_pids=sorted(set(foreign)),monitor_errors=errors,no_observed_contention=not foreign and not errors,mass_var_matches_primary_latents=exact,source_sha256=sha(Path(__file__)),gpu=gpu,
        speedup_vs_native=rows['dense']['median']/rows['mass_var']['median'],speedup_vs_host_dense=rows['dense_host']['median']/rows['mass_var']['median'])
    write(root/'summary.json',json.dumps(result,indent=2))
    lines=['# 空闲卡补充端到端对照','', 'basketball_court，seed606，同模型、原生RoPE、30秒含VAE。每方法3次完整生成，轮换顺序。dense_host保持原始KV顺序/搬移/attention，仅层1..29改用CPU索引；每次完整latent要求与native dense逐元素一致。', '', '| 方法 | n | 含VAE中位秒 [min,max] | 生成中位秒 |','|---|---:|---:|---:|']
    for m,r in rows.items():lines.append(f"| {m} | {r['n']} | {r['median']:.3f} [{r['min']:.3f},{r['max']:.3f}] | {r['generation']:.3f} |")
    lines+=['',f"mass_var 对 native dense 速度比：{result['speedup_vs_native']:.3f}×；对 host-index dense：{result['speedup_vs_host_dense']:.3f}×。",'',f"5秒间隔GPU进程采样是否未发现其他进程：{result['no_observed_contention']}；采样不能排除间隔内短暂任务。mass_var 与主实验latent一致：{exact}。",'', '只覆盖一个场景，不能替代四场景速度验收。质量仍沿用原生dense参考；不能因补测速度好而绕过已失败的保真门槛。']
    write(root/'report.zh.md','\n'.join(lines)+'\n')
    print(json.dumps(result),flush=True)
    gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
