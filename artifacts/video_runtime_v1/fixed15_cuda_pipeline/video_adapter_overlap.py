import ctypes as C
import torch
from pack_kernel import pack_range
from summary_fragments import encode
from repair import retry
lib=C.CDLL('/tmp/jzk_fixed15_paired_k.so')
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
        self.copy_stream=torch.cuda.Stream()
        self.summary_graph=rt.summary_graph
        self.layer_for_model={id(rt.model[l]):l for l in rt.stores}
        rt.summary_graph=self.summary
    def summary(self,model,kg,vg,*args,**kwargs):
        if not self.enabled or 'out' not in kwargs:
            return self.summary_graph(model,kg,vg,*args,**kwargs)
        l=self.layer_for_model[id(model)]
        store=self.rt.stores[l];hist=store.n;p=hist+72
        k,v=store.cache['k'],store.cache['v'];key=k.data_ptr()
        cur=torch.cuda.current_stream()
        self.copy_stream.wait_stream(cur)
        with torch.cuda.stream(self.copy_stream):
            self.prepare(key,k,v,store,hist,p)
        result=self.summary_graph(model,kg,vg,*args,**kwargs)
        self.copy_stream.wait_stream(cur)
        with torch.cuda.stream(self.copy_stream):
            s=self.rt.slabs[l].slice(0,p);tiles=(p+15)//16
            summary=self.rt.workspace.tensor('probe_packed_summary',(12,tiles,10624),k.device,k.dtype)
            encode[(12,tiles,42)](s.keys,s.mean,s.detail,summary,p,s.mean.stride(0)//128,tiles,num_warps=4)
        return result
    def reset(self):
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
        store=self.stores[key]
        pk,pv=self.buffers[key]
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        out=workspace.tensor('probe_out',q.shape,q.device,q.dtype)
        flags=workspace.tensor('probe_bad',(12,72),q.device,torch.int32)
        tiles=(p+15)//16
        summaries=workspace.tensor('probe_packed_summary',(12,tiles,10624),q.device,q.dtype)
        hp=s.mean.stride(0)//128
        basis=basis.bfloat16().contiguous()
        status=lib.run(*(x.data_ptr() for x in (q,pk,pv,ids,out,summaries,basis,flags)),p,count,self.stream)
        assert not status
        retry[(72,12)](flags,q,k,v,ids,s.keys,s.mean,s.detail,basis,out,p,hp,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
        self.calls+=1
        return out
    def prepare(self,key,k,v,store,hist,p):
        if key not in self.buffers:self.buffers[key]=tuple(torch.empty((432,12,8192),device=k.device,dtype=k.dtype) for _ in range(2))
        pk,pv=self.buffers[key]
        def pack(a,b):
            if b>a:pack_range[(b-a,12,16)](k,v,pk,pv,a)
        if key not in self.history:pack(0,hist)
        else:
            commits,start=self.history[key]
            if commits!=store.commits:
                assert store.commits==commits+1
                end=min(start+72,360)
                pack(start,end)
                pack(24,24+72-(end-start))
        self.history[key]=store.commits,store.write
        pack(hist,p)
