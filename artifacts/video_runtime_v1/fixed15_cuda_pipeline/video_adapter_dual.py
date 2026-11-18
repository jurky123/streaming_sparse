import ctypes as C
import torch
from local_experiments.video_runtime_v1 import page_prefetch as pipeline
from pack_kernel import pack_range
from tail import complete
original=pipeline.stagger_copies
def apply_layout(ir):
    if 'public @complete(' in ir:
        aligned=pipeline.align_coefficients(ir)
        assert aligned!=ir
        return aligned
    return original(ir)
pipeline.stagger_copies=apply_layout
finish=pipeline.PagePipeline(complete.fn)
lib=C.CDLL('/tmp/jzk_fixed15_shared_overlap.so')
lib.run.argtypes=[C.c_void_p]*7+[C.c_int,C.c_void_p]
from summary_fragments import encode,combine
taillib=C.CDLL('/tmp/jzk_fixed15_packed_tail.so')
taillib.run.argtypes=[C.c_void_p]*8+[C.c_int]*3+[C.c_void_p]
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
    def reset(self):
        self.history.clear()
        self.calls=0
    def __call__(self,q,k,v,s,basis,ids,workspace,**options):
        if not self.enabled or ids.shape[-1]!=64:
            return self.base(q,k,v,s,basis,ids,workspace,**options)
        key=k.data_ptr()
        store=self.stores[key]
        if key not in self.buffers:self.buffers[key]=torch.empty_like(k),torch.empty_like(v)
        pk,pv=self.buffers[key]
        def pack(a,b):
            if b>a:pack_range[(b-a,12,16)](k,v,pk,pv,a)
        if key not in self.history:pack(0,360)
        else:
            commits,start=self.history[key]
            if commits!=store.commits:
                assert store.commits==commits+1
                end=min(start+72,360)
                pack(start,end)
                pack(24,24+72-(end-start))
        self.history[key]=store.commits,store.write
        pack(360,432)
        oe=workspace.tensor('probe_exact',q.shape,q.device,torch.float32)
        le=workspace.tensor('probe_lse',q.shape[:-1],q.device,torch.float32)
        mass=workspace.tensor('probe_mass',q.shape[:-1],q.device,torch.float32)
        out=workspace.tensor('probe_out',q.shape,q.device,q.dtype)
        status=lib.run(*(x.data_ptr() for x in (q,pk,pv,ids,oe,le,mass)),64,self.stream)
        assert not status
        hp=s.mean.stride(0)//128
        summaries=workspace.tensor('probe_summaries',(12,27,10624),q.device,q.dtype)
        mu=workspace.tensor('probe_mu',(12,4608,128),q.device,torch.float32)
        code=workspace.tensor('probe_code',(12,4608,8),q.device,torch.float32)
        lt=workspace.tensor('probe_lt',(12,4608),q.device,torch.float32)
        encode[(12,27,42)](s.keys,s.mean,s.detail,summaries,432,hp,27,num_warps=4)
        status=taillib.run(*(x.data_ptr() for x in (q,summaries,s.mean,s.detail,ids,mu,code,lt)),432,hp,64,self.stream)
        assert not status
        combine[(72,12)](q,k,v,ids,oe,le,mass,mu,code,lt,basis.bfloat16().contiguous(),out,432,64,num_warps=4,num_stages=2,enable_fp_fusion=False)
        self.calls+=1
        return out
