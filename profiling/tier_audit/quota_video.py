import hashlib,json,sys,time,tempfile,subprocess
from pathlib import Path
import torch
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
import queue_policy
from local_experiments.broad_common import Engine
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from utils.wan_wrapper import WanTextEncoder
@torch.no_grad()
def main():
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=Path(__file__).parent
    cached=Path('/tmp/quota_prompt720.pt')
    if not cached.exists():
        enc=WanTextEncoder().to(device='cuda',dtype=torch.bfloat16)
        prompt=(repo/'prompts/vbench/all_dimension.txt').read_text().splitlines()[720]
        torch.save(enc([prompt])['prompt_embeds'].cpu(),cached)
        del enc
    embeddings=[torch.load(p,weights_only=True).cuda() for p in [cached,repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt']]
    cfg=config('intersection',local_assets=repo)
    configure(cfg)
    engine=Engine()
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    if sys.argv[1] in ('before','before-paired','after'):
        old=Path(tempfile.mkdtemp())
        (old/'graph.py').write_bytes(subprocess.check_output(['git','show','69c1eb15cbc7d38b821d50981fb7353100c3fd83:reproduction/vbench_current/graph.py'],cwd=repo))
        if sys.argv[1]=='after':
            graph=old/'graph.py'
            source=graph.read_text().replace('hr=risk[...,24:-72].argsort(dim=-1,descending=True,stable=True).argsort(dim=-1)', 'history=risk[...,24:-72]\n            order=history.argsort(dim=-1,descending=True,stable=True)\n            keep=torch.arange(history.shape[-1],device=risk.device)<(self.n-curr)\n            history_mask=torch.empty_like(history,dtype=torch.bool).scatter_(-1,order,keep.expand_as(history))').replace('hr<(self.n-curr),cr<curr','history_mask,cr<curr')
            graph.write_text(source)
        (old/'policies.npz').symlink_to(queue_policy.dest/'policies.npz')
        queue_policy.dest=old
    gen=queue_policy.attach(rt,engine,10)
    paired=sys.argv[1].endswith('-paired')
    rows=[]
    for i,seed in ([(0,701)]+[(i,s) for s in (701,702) for i in (0,1)]*(4 if paired else 2)):
        if paired and rows:
            print('READY',flush=True)
            input()
        rt.reset()
        torch.cuda.synchronize()
        t=time.perf_counter()
        z=gen(embeddings[i],seed,7)
        torch.cuda.synchronize()
        row=[i,seed,time.perf_counter()-t,hashlib.sha256(z.cpu().view(torch.uint8).numpy()).hexdigest()]
        print(row,flush=True)
        rows.append(row)
        (out/('quota_'+sys.argv[1]+'.json')).write_text(json.dumps(rows))
main()
