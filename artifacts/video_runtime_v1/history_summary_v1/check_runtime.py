"""Verify cache invalidation and CUDA Graph replays against a full prepare."""
import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from local_experiments.video_runtime_v1.fixed15_attention import PackedAttention,library
from local_experiments.video_runtime_v1.attention import Workspace

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,k,v,pk,pv,ids,keys,mean,detail,basis,p
class Store:
    packed_writer=True
    commits=2
store=Store()
summary=SimpleNamespace(keys=keys,mean=mean,detail=detail)
candidate=PackedAttention()
control=PackedAttention()
old=ctypes.CDLL('/tmp/jzk_compact_half.so')
old.prepare_fixed15.argtypes=library().prepare_fixed15.argtypes
control.prepare_kernel=old.prepare_fixed15
shared=Workspace()
reference=Workspace()
for obj in (candidate,control):
    obj.buffers[store]=(pk,pv)
def run(obj,ws):
    return obj(store,q,k,v,summary,basis,ids,ws)
def check():
    assert torch.equal(run(candidate,shared).view(torch.uint8),run(control,reference).view(torch.uint8))
check()
mean[:,p-72:].neg_()
check()
assert candidate.summary_hits==1
store.commits+=1
mean[:,:p-72].neg_()
check()
assert candidate.summary_hits==1
candidate.reset()
check()
assert candidate.summary_hits==0
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    result=run(candidate,shared)
assert candidate.captured
for _ in range(2):
    detail[:,24:p-72].neg_()
    graph.replay()
    assert torch.equal(result.view(torch.uint8),run(control,reference).view(torch.uint8))
check()
assert candidate.summary_hits==0
(root/f'cache_contract_{p}.json').write_text(json.dumps(dict(eager_hit=True,commit=True,reset=True,graph_replays=2,bitwise=True)))
