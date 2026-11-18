import inspect
from pathlib import Path
from local_experiments.gap_exact_round_kernel import _attn_fwd
from local_experiments.review_v1.tuned_attention import _tail
p=Path(__file__).parent
s=inspect.getsource(_attn_fwd.fn).replace('def _attn_fwd(','def selected_online(').replace('    Q, K, V,','    TILE, Q, K, V,',1).replace('tl.program_id(0).to(tl.int32)','TILE.to(tl.int32)')
t=inspect.getsource(_tail.fn).replace('def _tail(Q,','def selected_tail(TILE,Q,').replace('t=tl.program_id(0);','t=TILE;').replace('_attn_fwd(Q,','selected_online(TILE,Q,')
s='''import triton
import triton as tr
import triton.language as tl
import math
from local_experiments.review_v1.tuned_attention import _av,_anchored_exact
'''+s+'\n'+t+'''
@tr.jit
def repair_group(FLAGS,Q,K,V,IDS,KEY,MEAN,DETAIL,BASIS,OUT,P:tl.constexpr,HP:tl.constexpr,C:tl.constexpr,G:tl.constexpr):
 h=tl.program_id(1)
 start=tl.program_id(0)*G
 slots=start+tl.arange(0,G)
 bad=tl.load(FLAGS+h*72+slots,slots<72,0)
 if tl.max(bad,0):
  for i in range(G):
   tile=start+i
   if tl.load(FLAGS+h*72+tile,tile<72,0):
    selected_tail(tile,Q,KEY,MEAN,DETAIL,None,None,None,None,P,4608,12,128,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=BASIS,OUT=OUT,HEAD_PAGES=HP,K=K,V=V,IDS=IDS,ANCHOR_SCALE=False,COLUMN_MASS=True,EXACT_COUNT=C,QK_SCALE=128**-.5)
'''
(p/'group_repair.py').write_text(s)
s=(p/'paired_k.py').read_text().replace('from repair import retry','from group_repair import repair_group\ngroup=int(os.environ["REPAIR_GROUP"])').replace('retry[(72,12)]','repair_group[(triton.cdiv(72,group),12)]').replace('basis,new,p,hp,count,num_warps=4','basis,new,p,hp,count,group,num_warps=4').replace("f'paired_k_{p}.json'","f'repair_{group}_{p}.json'")
s+='''
q.fill_(10);k.zero_();k[:,:64].fill_(10.)
repack(0);control();run()
r=dict(triggered=int(flags.sum()),bitwise=torch.equal(old.view(torch.uint8),new.view(torch.uint8)),finite=bool(new.isfinite().all()))
assert r['triggered']==864 and r['bitwise'] and r['finite'],r
(root/f'group{group}_stress_{p}.json').write_text(json.dumps(r,indent=2))
'''
(p/'bench_group.py').write_text(s)
