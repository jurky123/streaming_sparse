"""Real deployed high/low-AV exact attention, P432, for hardware counters."""
import torch,json,argparse
from .weights import ROOT
from ..gap_exact_round_kernel import _attn_fwd
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--fold',action='store_true');args=parser.parse_args()
 fn=_attn_fwd
 if args.fold:
  from .exact_fold import exact_fold
  fn=exact_fold
 torch.set_num_threads(4);s=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
 q,k,v=[s[x].cuda().contiguous() for x in ('q','k','v')];ids=s['ids'].cuda().contiguous()[None]
 out=torch.empty(q.shape,device='cuda',dtype=torch.float32);lse=torch.empty(12,4608,device='cuda')
 for _ in range(2):
  compiled=fn[(72,12)](q,k,v,128**-.5,43,ids,lse,out,12,4608,k.shape[1],72,128,64,64,HIGH_PRECISION_AV=True,num_warps=4,num_stages=2);torch.cuda.synchronize()
 root=ROOT/'artifacts/video_runtime_v1/exact_pipeline_v1';root.mkdir(exist_ok=True)
 (root/('fold_kernel.json' if args.fold else 'kernel.json')).write_text(json.dumps(dict(registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared),indent=2))
 for ext in ('ttgir','ptx','sass'):
  if ext in compiled.asm:(root/f"{'fold' if args.fold else 'baseline'}.{ext}").write_text(compiled.asm[ext])
if __name__=='__main__':main()
