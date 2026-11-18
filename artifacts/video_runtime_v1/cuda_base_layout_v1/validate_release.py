"""Real-input module and guard parity for the installed base, including ring writes."""
from pathlib import Path
from types import SimpleNamespace
import json,sys
import torch
from local_experiments.video_runtime_v1.fixed15_attention import PackedAttention
from local_experiments.video_runtime_v1.page_prefetch import coeff_tail
from local_experiments.video_runtime_v1.fixed15_layout import pack_range

root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import q,k,v,ids,keys,mean,detail,basis,p,hp,count,old

class Workspace:
    def __init__(self):self.buffers={}
    def tensor(self,name,shape,device,dtype):
        if name not in self.buffers:self.buffers[name]=torch.empty(shape,device=device,dtype=dtype)
        return self.buffers[name]
class Store:pass

store=Store()
store.cache={'k':k,'v':v}
store.commits=0
store.write=0
w=Workspace()
s=SimpleNamespace(mean=mean,keys=keys,detail=detail)
attn=PackedAttention()
def verify():
    coeff_tail[(72,12)](q,keys,mean,detail,None,None,None,None,p,4608,12,128,
        BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=old,HEAD_PAGES=hp,
        K=k,V=v,IDS=ids,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=count,
        QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
    actual=attn(store,q,k,v,s,basis,ids,w)
    assert torch.equal(actual.view(torch.uint8),old.view(torch.uint8))
    return int(w.buffers['packed_guard'].sum())
actual=verify()
attn.reset()
q.fill_(10);k.zero_();k[:,:64].fill_(10)
extreme=verify()
assert extreme==864
(root/f'guard_{p}.json').write_text(json.dumps(dict(pages=p,actual_flagged=actual,
    extreme_flagged=extreme,bitwise=True),indent=2))
