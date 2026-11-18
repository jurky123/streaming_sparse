"""Validate every byte of the changed summary, including padded windows."""
import ctypes
import os
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
import json
from pathlib import Path
import sys

import torch

root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,k,v,pk,pv,keys,mean,detail,p,hp,summary

pk_test,pv_test=torch.empty_like(pk),torch.empty_like(pv)
packed=torch.empty_like(summary)
release=os.environ.get('CANDIDATE')=='release'
lib=library() if release else ctypes.CDLL('/tmp/jzk_tailmax_direct_mass.so')
lib.prepare_fixed15.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*2+[ctypes.c_void_p]
args=(k,v,pk_test,pv_test,keys,mean,detail,packed)
assert lib.prepare_fixed15(*(t.data_ptr() for t in args),p,hp,torch.cuda.current_stream().cuda_stream)==0
index=torch.arange(8192,device=q.device)
key_block=index//1024
lane=index//8%32
token=16*(index//256%4)+8*(index//4%2)+lane//4
page=8*(token//32)+2*((token%8)//2)+token%32//16
rep=4*page+2*((token//8)%2)+token%2
src=1024*key_block+256*(rep//16)+8*(4*(rep%8)+lane%4)+4*(rep//8%2)+index%4
assert src.unique().numel()==8192
expected=summary[:,:,:8192].index_select(-1,src)
assert torch.equal(expected,packed[:,:,:8192])
triton_summary=torch.empty_like(packed)
nt=(p+15)//16
encode[(12,nt,42)](keys,mean,detail,triton_summary,p,hp,nt,DIRECT_MASS=True,num_warps=4)
assert torch.equal(triton_summary.view(torch.uint8),packed.view(torch.uint8))
assert torch.equal(summary[:,:,8192:],packed[:,:,8192:])
for have,want in ((pk_test,pk),(pv_test,pv)):
    assert torch.equal(have.reshape(p,12,8192)[-72:],want.reshape(p,12,8192)[-72:])
(root/f"{'release_' if release else ''}layout_{p}.json").write_text(json.dumps(dict(p=p,key_elements=expected.numel(),same_current_kv=True,
                                                      checked_mean_detail=True,permutation=True,triton_matches=True)))
