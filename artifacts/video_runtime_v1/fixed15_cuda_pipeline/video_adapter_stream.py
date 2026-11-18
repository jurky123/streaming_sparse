import ctypes as C
import torch
from pack_kernel import pack_range
from summary_fragments import encode
from repair import retry
lib=C.CDLL('/tmp/jzk_fixed15_parity.so')
lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]
class Packed:
    def __init__(self,rt,base):
        self.rt=rt
        self.base=base
        self.history={}
        self.enabled=False
        self.buffers={}
        self.calls=0
        self.stream=torch.cuda.current_stream().cuda_stream
        self.stores={s.cache['k'].data_ptr():s for s in rt.stores.values()}
        stores=[rt.stores[layer] for layer in sorted(rt.stores)]
        self.nextstore={s.cache['k'].data_ptr():(i, stores[i+1] if i+1<len(stores) else None) for i,s in enumerate(stores)}
        self.copy_stream=torch.cuda.Stream()
        self.ready=[torch.cuda.Event() for _ in range(2)]
        self.consumed=[torch.cuda.Event() for _ in range(2)]
        self.enter=torch.cuda.Event()
        self.planned=None
    def reset(self):
        self.copy_stream.synchronize()
        self.planned=None
        self.history.clear()
        self.calls=0
    def __call__(self,q,k,v,s,basis,ids,workspace,**options):
        if not self.enabled:
            return self.base(q,k,v,s,basis,ids,workspace,**options)
        p=k.shape[1]//64
        hist=p-72
        count=ids.shape[-1]
        assert count==p*15//100
        key=k.data_ptr()
        i,next_store=self.nextstore[key]
        slot=i%2
        for idx in (0,1):
            if idx not in self.buffers:
                self.buffers[idx]=tuple(torch.empty((432,12,8192),device=k.device,dtype=k.dtype) for _ in range(2))
        pk,pv=self.buffers[slot]
        cur=torch.cuda.current_stream()
        self.enter.record(cur)
        if self.planned==(key,hist):
            cur.wait_event(self.ready[slot])
        else:
            assert i==0
            pack_range[(hist,12,16)](k,v,pk,pv,0)
        pack_range[(72,12,16)](k,v,pk,pv,hist)
        out=workspace.tensor('probe_out',q.shape,q.device,q.dtype)
        flags=workspace.tensor('probe_bad',(12,72),q.device,torch.int32)
        tiles=(p+15)//16
        summaries=workspace.tensor('probe_packed_summary',(12,tiles,10624),q.device,q.dtype)
        hp=s.mean.stride(0)//128
        encode[(12,tiles,42)](s.keys,s.mean,s.detail,summaries,p,hp,tiles,num_warps=4)
        basis=basis.bfloat16().contiguous()
        status=lib.run(*(x.data_ptr() for x in (q,pk,pv,ids,out,summaries,basis,flags)),p,count,self.stream)
        assert not status
        retry[(72,12)](flags,q,k,v,ids,s.keys,s.mean,s.detail,basis,out,p,hp,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
        self.consumed[slot].record(cur)
        if next_store is not None:
            alt=1-slot
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(self.enter)
                if i:self.copy_stream.wait_event(self.consumed[alt])
                nk,nv=next_store.cache['k'],next_store.cache['v']
                pack_range[(hist,12,16)](nk,nv,*self.buffers[alt],0)
                self.ready[alt].record(self.copy_stream)
            self.planned=(nk.data_ptr(),hist)
        else:
            self.planned=None
        self.calls+=1
        return out
